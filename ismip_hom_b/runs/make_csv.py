#!/usr/bin/env python3

"""Extract the ISMIP-HOM experiment B profiles read by ../figure/make_figure.py.

For each length L, writes to outputs/:
  exp_b_unified_<L>km_identity_lbfgs_igm.csv   IGM surface speed along y = 0
  exp_b_unified_<L>km_identity_lbfgs_ref.csv   Elmer/Ice full-Stokes ('oga')
and prints the relative L2 error of IGM against the reference.
"""

from pathlib import Path

import numpy as np
import xarray as xr

LENGTHS = [20, 40, 80, 160]


def main():
    here = Path(__file__).resolve().parent
    path_outputs = here / "outputs"

    for length in LENGTHS:
        with xr.open_dataset(path_outputs / f"{length}km" / "results.nc") as ds:
            v_igm = ds["velsurf_mag"].values[0, 0]
        x_igm = np.linspace(0.0, 1.0, len(v_igm))

        ref = np.loadtxt(here / "data" / "oga" / f"oga1b{length:03d}.txt")
        x_ref = ref[:, 0]
        v_ref = ref[:, 1]

        stem = f"exp_b_unified_{length}km_identity_lbfgs"
        for suffix, x, v in [("igm", x_igm, v_igm), ("ref", x_ref, v_ref)]:
            np.savetxt(
                path_outputs / f"{stem}_{suffix}.csv",
                np.column_stack([x, v]),
                delimiter=",",
                header="x,v",
                comments="",
            )

        error = np.linalg.norm(np.interp(x_ref, x_igm, v_igm) - v_ref)
        error /= np.linalg.norm(v_ref)
        print(f"L = {length:3d} km: relative L2 error = {100 * error:.1f} %")


if __name__ == "__main__":
    main()
