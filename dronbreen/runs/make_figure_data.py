#!/usr/bin/env python3

"""Write the four data files that ../figure/make_figure.py draws, from the best run (step 3 of README.md).

    python make_figure_data.py --run outputs/step1

Reads the run's output.nc at its last saved time (2022) and its .hydra/config.yaml,
and the GPR transects in data/cts_gpr/. Every quantity goes through tools/cts.py, the
same CTS diagnostic the calibration uses:

    section.nc      the thermal section along the featured transect (T3 by default)
    transects.csv   per-transect temperate areas and CTS RMSE
    map.nc          ice thickness and bed on the model grid
    tracks.csv      the radar tracks, point by point

Copy them to ../figure/ to redraw the figure from a new run.
Requires numpy, pandas, scipy, xarray and omegaconf.
"""

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import OmegaConf

from tools.cts import (
    cts_elevation,
    enthalpy_grid,
    layer_elevations,
    load_cts_profiles,
    profile_rmse,
    representative_profile,
    sample_columns,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", required=True, help="Step 1 run directory (has output.nc)")
    p.add_argument("--data-dir", default=os.path.join(HERE, "data", "cts_gpr"))
    p.add_argument("--glacier", default="dronbreen")
    p.add_argument("--year", type=int, default=2022, help="Survey year")
    p.add_argument(
        "--section-line",
        help="Transect to feature in the section (e.g. T5); default: "
        "tools.cts.representative_profile, which picks T3",
    )
    p.add_argument("--out", default=os.path.join(HERE, "outputs", "figure_data"))

    return p.parse_args()


def cross_sections(profile, model):
    """Temperate and cold cross-sectional areas (m²), observed and modelled
    (trapezium rule along the profile)."""

    dist = profile["distance"]

    def areas(cts, bed, surface):
        temperate = np.clip(cts - bed, 0.0, None)
        cold = np.clip(surface - cts, 0.0, None)
        ok = np.isfinite(temperate) & np.isfinite(cold)
        if ok.sum() < 2:
            return 0.0, 0.0
        return (
            float(np.trapz(temperate[ok], dist[ok])),
            float(np.trapz(cold[ok], dist[ok])),
        )

    return (
        areas(profile["cts_obs"], profile["bed_obs"], profile["surf_obs"]),
        areas(model["cts"], model["topg"], model["usurf"]),
    )


def main():
    args = parse_args()
    run = os.path.abspath(args.run)

    cfg = OmegaConf.load(os.path.join(run, ".hydra", "config.yaml"))
    nz, vert_spacing = enthalpy_grid(cfg)

    snapshot = xr.open_dataset(os.path.join(run, "output.nc")).isel(time=-1)
    year = int(round(float(snapshot["time"])))
    x, y = snapshot["x"].values, snapshot["y"].values
    profiles = load_cts_profiles(args.data_dir, args.glacier, args.year)

    rows, sampled = [], []
    for profile in profiles:
        column = lambda name: sample_columns(
            snapshot[name].values, x, y, profile["easting"], profile["northing"]
        )
        topg, thk, usurf = column("topg"), column("thk"), column("usurf")
        E, E_pmp = column("E"), column("E_pmp")
        cts = cts_elevation(E, E_pmp, topg, thk, vert_spacing)
        (t_obs, c_obs), (t_mod, _) = cross_sections(
            profile, {"topg": topg, "usurf": usurf, "cts": cts}
        )
        rows.append(
            {
                "transect": profile["label"],
                "radar_line": profile["name"],
                "temperate_obs_m2": t_obs,
                "temperate_mod_m2": t_mod,
                "total_ice_m2": t_obs + c_obs,
                "cts_rmse_m": profile_rmse(cts, profile["cts_obs"]),
            }
        )
        sampled.append(
            {"topg": topg, "thk": thk, "usurf": usurf, "E": E, "E_pmp": E_pmp}
        )

    misfits = [r["cts_rmse_m"] for r in rows]
    if args.section_line:
        featured = next(
            p for p in profiles if args.section_line in (p["label"], p["name"])
        )
    else:
        featured = representative_profile(profiles, misfits)
    i = profiles.index(featured)
    m = sampled[i]

    os.makedirs(args.out, exist_ok=True)

    df = pd.DataFrame(rows).sort_values("temperate_obs_m2", ascending=False)
    df.to_csv(os.path.join(args.out, "transects.csv"), index=False, float_format="%g")

    ice_type = np.where(m["E"] >= m["E_pmp"], 1, 0).astype(np.int8)
    ice_type[:, m["thk"] < 1.0] = -1
    elevation = layer_elevations(m["topg"], m["thk"], nz, vert_spacing)
    section = xr.Dataset(
        {
            "ice_type": (("layer", "point"), ice_type),
            "elevation": (("layer", "point"), elevation),
            "surface": ("point", m["usurf"]),
            "bed": ("point", m["topg"]),
            "cts_observed": (
                "point",
                np.where(featured["is_temperate"], featured["cts_obs"], np.nan),
            ),
        },
        coords={"distance_km": ("point", featured["distance"] / 1000.0)},
        attrs={
            "transect": featured["label"],
            "radar_line": featured["name"],
            "cts_rmse_m": round(misfits[i], 3),
            "model_year": year,
            "ice_type_key": "-1 no ice, 0 cold, 1 temperate",
        },
    )
    section.to_netcdf(os.path.join(args.out, "section.nc"))

    grid = xr.Dataset(
        {
            "thk": (
                ("y", "x"),
                snapshot["thk"].values.astype(np.float32),
                {"units": "m", "long_name": "ice thickness"},
            ),
            "topg": (
                ("y", "x"),
                snapshot["topg"].values.astype(np.float32),
                {"units": "m", "long_name": "bed elevation"},
            ),
        },
        coords={"x": ("x", x, {"units": "m"}), "y": ("y", y, {"units": "m"})},
        attrs={
            "crs": "EPSG:32633 (UTM 33N)",
            "model_year": year,
            "source": "output.nc of a Step 1 run, last saved time. The run keeps its "
            "geometry fixed, so this is the input geometry: the 2024 NPI surface DEM "
            "minus a bed kriged from the radar.",
        },
    )
    grid.to_netcdf(os.path.join(args.out, "map.nc"))

    label = {p["name"]: p["label"] for p in profiles}
    tracks = pd.concat(
        [
            pd.DataFrame(
                {
                    "transect": label[p["name"]],
                    "easting": p["easting"],
                    "northing": p["northing"],
                }
            )
            for p in profiles
        ]
    )
    order = {name: k for k, name in enumerate(df["transect"])}
    tracks = tracks.sort_values("transect", key=lambda s: s.map(order), kind="stable")
    tracks.to_csv(
        os.path.join(args.out, "tracks.csv"), index=False, float_format="%.2f"
    )

    cost = float(np.mean([r for r in misfits if np.isfinite(r)]))
    print(f"cost_cts (mean CTS RMSE over transects) = {cost:.2f} m")
    print(f"section: {featured['label']} ({featured['name']}), model year {year}")
    print(f"wrote section.nc, transects.csv, map.nc, tracks.csv to {args.out}")


if __name__ == "__main__":
    main()
