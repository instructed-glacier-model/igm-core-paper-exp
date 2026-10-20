"""Cold-temperate transition surface (CTS) diagnostics for the Drønbreen experiment.

Shared by the process module `user/code/processes/track_cts_err` and by
`make_figure_data.py`. The CTS is the level where the enthalpy `E` reaches the
pressure-melting enthalpy `E_pmp` (temperate below, cold above), located by linear
interpolation of `E - E_pmp` between the two layers that bracket the sign change.

Observation columns (Mannerfelt et al., 2025; doi:10.5281/zenodo.17882300):

* `temperate`            - thickness of the temperate layer above the bed, in m
* `temperate_elevation`  - CTS elevation, `bed_elevation + temperate`
* `temperate_upper/lower`- bounds on that thickness, with
  `temperate_upper <= temperate <= temperate_lower`.
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def sigma_midpoints(nz, vert_spacing):
    """Mid-layer positions of IGM's stretched vertical grid, as fractions of thickness.

    Same discretisation as the enthalpy module: `nz` and `vert_spacing` must come from
    `processes.enthalpy.numerics`.
    """

    zeta = np.arange(nz + 1) / nz
    edges = (zeta / vert_spacing) * (1.0 + (vert_spacing - 1.0) * zeta)

    return 0.5 * (edges[:-1] + edges[1:])


def layer_elevations(topg, thk, nz, vert_spacing):
    """Elevation of every layer midpoint. Returns shape `(nz,) + topg.shape`."""

    zeta_mid = sigma_midpoints(nz, vert_spacing)

    return topg[None, ...] + zeta_mid.reshape((-1,) + (1,) * topg.ndim) * thk[None, ...]


def cts_elevation(E, E_pmp, topg, thk, vert_spacing, min_thk=1.0):
    """Elevation of the cold-temperate transition surface.

    `E` and `E_pmp` have shape `(nz, ...)`; `topg` and `thk` have shape `(...)`.
    Returns an array of shape `(...)`, NaN only where there is no ice to speak of
    (`thk < min_thk`).

    Columns with no sign change are kept: the CTS of an entirely cold column is at the
    bed, that of an entirely temperate column at the surface.
    """

    E = np.asarray(E, dtype=float)
    E_pmp = np.asarray(E_pmp, dtype=float)
    topg = np.asarray(topg, dtype=float)
    thk = np.asarray(thk, dtype=float)

    nz = E.shape[0]
    delta = E - E_pmp
    z = layer_elevations(topg, thk, nz, vert_spacing)

    # Lowest pair of adjacent layers that straddle E == E_pmp.
    crossing = np.sign(delta[:-1]) != np.sign(delta[1:])
    found = crossing.any(axis=0)
    k = np.argmax(crossing, axis=0)

    take = lambda a: np.take_along_axis(a, k[None, ...], axis=0)[0]
    d1, d2 = take(delta[:-1]), take(delta[1:])
    z1, z2 = take(z[:-1]), take(z[1:])

    slope = d2 - d1
    interpolated = np.where(
        np.abs(slope) < 1e-12, z1, z1 - d1 * (z2 - z1) / np.where(slope == 0, 1.0, slope)
    )

    # No crossing: temperate throughout (CTS at the surface) or cold throughout (at the bed).
    uniform = np.where(delta[0] >= 0, topg + thk, topg)

    return np.where(thk < min_thk, np.nan, np.where(found, interpolated, uniform))


def enthalpy_grid(cfg):
    """`(Nz, vert_spacing)` of the enthalpy vertical grid, from a run configuration.

    Accepts either a live Hydra config or the `.hydra/config.yaml` of a finished run.
    """

    numerics = cfg["processes"]["enthalpy"]["numerics"]

    return int(numerics["Nz"]), float(numerics["vert_spacing"])


def sample_columns(field, x, y, easting, northing):
    """Bilinearly sample a model field at scattered points.

    `field` is `(ny, nx)` or `(nz, ny, nx)`; the result is `(npoints,)` or
    `(nz, npoints)`. Points outside the grid come back as NaN.
    """

    field = np.asarray(field, dtype=float)
    points = np.column_stack([np.asarray(northing), np.asarray(easting)])

    if field.ndim == 2:
        interp = RegularGridInterpolator(
            (y, x), field, bounds_error=False, fill_value=np.nan
        )
        return interp(points)

    interp = RegularGridInterpolator(
        (y, x), np.moveaxis(field, 0, -1), bounds_error=False, fill_value=np.nan
    )

    return interp(points).T


def load_cts_profiles(data_dir, glacier_name, year_obs):
    """Load the GPR profiles of one glacier and survey year.

    Returns a list of dicts, one per radar line, labelled T1, T2, ... by decreasing
    observed temperate area. Raises if no file matches.
    """

    data_dir = os.path.expanduser(str(data_dir))
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"CTS observation directory not found: {data_dir}")

    pattern = os.path.join(data_dir, f"cts_{glacier_name}-{year_obs}*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CTS observation file matches {pattern}")

    profiles = []
    for path in files:
        df = pd.read_csv(path)
        profiles.append(
            {
                "name": os.path.basename(path)[len("cts_") : -len(".csv")],
                "distance": df["distance"].to_numpy(),
                "easting": df["easting"].to_numpy(),
                "northing": df["northing"].to_numpy(),
                "surf_obs": df["elevation"].to_numpy(),
                "bed_obs": df["bed_elevation"].to_numpy(),
                "cts_obs": df["temperate_elevation"].to_numpy(),
                # Uncertainty band, converted from temperate-layer thickness to elevation.
                "cts_obs_low": df["bed_elevation"].to_numpy()
                + df["temperate_upper"].to_numpy(),
                "cts_obs_high": df["bed_elevation"].to_numpy()
                + df["temperate_lower"].to_numpy(),
                "is_temperate": df["temperate"].to_numpy() > 0,
                "bed_type": df["bed_type"].to_numpy(),
            }
        )

    # Labels T1, T2, ... by decreasing observed temperate area.
    profiles.sort(key=observed_temperate_area, reverse=True)
    for index, profile in enumerate(profiles, start=1):
        profile["label"] = f"T{index}"

    return profiles


def observed_temperate_area(profile):
    """Cross-sectional area of temperate ice the radar recorded on this transect (m²)."""

    return float(
        np.trapz(
            np.clip(profile["cts_obs"] - profile["bed_obs"], 0.0, None),
            profile["distance"],
        )
    )


def representative_profile(profiles, misfits, min_contrast=0.15):
    """Transect featured in the section panel (T3 for the paper run).

    Among transects with at least `min_contrast` of their ice area on the minority side
    (temperate or cold), the one whose misfit is closest to the median.
    """

    candidates = [
        (p, m)
        for p, m in zip(profiles, misfits)
        if np.isfinite(m) and _contrast_fraction(p) >= min_contrast
    ]
    if not candidates:
        return most_contrasted_profile(profiles)

    median = float(np.nanmedian([m for m in misfits if np.isfinite(m)]))

    return min(candidates, key=lambda c: abs(c[1] - median))[0]


def _contrast_fraction(profile):
    """Minority share (temperate or cold) of the observed ice cross-section."""

    thickness = profile["surf_obs"] - profile["bed_obs"]
    ice = np.trapz(np.clip(thickness, 0.0, None), profile["distance"])
    if ice <= 0:
        return 0.0

    temperate = np.trapz(
        np.clip(profile["cts_obs"] - profile["bed_obs"], 0.0, None), profile["distance"]
    )
    share = temperate / ice

    return float(min(share, 1.0 - share))


def most_contrasted_profile(profiles):
    """Profile with the most points on the minority side (fallback of the above)."""

    def contrast(p):
        temperate = int(p["is_temperate"].sum())
        return min(temperate, len(p["is_temperate"]) - temperate)

    return max(profiles, key=contrast)


def profile_rmse(cts_model, cts_obs):
    """RMSE between modelled and observed CTS, ignoring points where either is NaN.

    Returns NaN if nothing is comparable.
    """

    valid = np.isfinite(cts_model) & np.isfinite(cts_obs)
    if not np.any(valid):
        return np.nan

    return float(np.sqrt(np.mean((cts_model[valid] - cts_obs[valid]) ** 2)))
