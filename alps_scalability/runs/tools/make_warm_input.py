#!/usr/bin/env python3
"""
Build a WARM-START input (topg + thk + usurf) from a reference run's snapshot: the accuracy runs
restart from the reference geometry at t = 300 yr and run 300 -> 500 yr.

Usage:
  ./tools/make_warm_input.py outputs/ref_local/bench_r06 --time 300 -o data/warm_local_r06.nc
"""
from __future__ import annotations

import argparse
from pathlib import Path

import netCDF4
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path, help="reference run dir holding output.nc")
    p.add_argument("--time", type=float, default=300.0, help="snapshot to branch from (yr)")
    p.add_argument("-o", "--out", type=Path, required=True)
    args = p.parse_args()

    with netCDF4.Dataset(args.run_dir / "output.nc") as src:
        times = np.asarray(src["time"][:], float)
        k = int(np.argmin(np.abs(times - args.time)))
        if abs(times[k] - args.time) > 1e-3:
            raise SystemExit(f"no snapshot at t={args.time} (available: {times.tolist()})")
        x, y = np.asarray(src["x"][:]), np.asarray(src["y"][:])
        thk = np.asarray(src["thk"][k], "f4")
        usurf = np.asarray(src["usurf"][k], "f4")
        topg = (np.asarray(src["topg"][k], "f4") if src["topg"].ndim == 3
                else np.asarray(src["topg"][:], "f4"))
        with netCDF4.Dataset(args.out, "w") as dst:
            dst.createDimension("x", len(x)); dst.createDimension("y", len(y))
            for name, dims, arr, t in (("x", ("x",), x, "f8"), ("y", ("y",), y, "f8"),
                                       ("topg", ("y", "x"), topg, "f4"),
                                       ("thk", ("y", "x"), thk, "f4"),
                                       ("usurf", ("y", "x"), usurf, "f4")):
                v = dst.createVariable(name, t, dims, zlib=(name not in ("x", "y")))
                v[:] = arr
            dst.history = (f"warm start at t={times[k]:g} yr from {args.run_dir}/output.nc, "
                           "built by tools/make_warm_input.py")
    print(f"wrote {args.out}  ({len(x)}x{len(y)}, t={times[k]:g} yr, "
          f"ice {float(thk.sum()*abs(x[1]-x[0])**2/1e9):.0f} km3)")


if __name__ == "__main__":
    main()
