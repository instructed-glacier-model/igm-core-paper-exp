#!/usr/bin/env python3
"""
Build the benchmark's own input file  data/input_bench.nc  from the master grid.

Master = data/input.nc (Alps, 9020x6020 @ 100 m, variable `topg` only; not shipped, its `topg` is the one of
data/input_bench.nc). Without it this script keeps data/input_bench.nc and only rebuilds data/input_crop.nc.
An all-zero `thk` is added (ice-free start; IGM's local input module needs more than one variable).

Also builds  data/input_crop.nc : the 160 x 134 km sub-domain of the accuracy check
(x 520.1-681.2 km, y 5112.1-5245.9 km, 1611x1338 @ 100 m), small enough for a reference solve.

Usage:  conda activate igm && ./tools/make_input.py
"""
from pathlib import Path

import netCDF4
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
SRC, DST = DATA / "input.nc", DATA / "input_bench.nc"

if not SRC.exists():
    print(f"{SRC} not found: keeping {DST}")
else:
    with netCDF4.Dataset(SRC) as src, netCDF4.Dataset(DST, "w") as dst:
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
        for name, dim in src.dimensions.items():
            dst.createDimension(name, dim.size)
        for name, var in src.variables.items():
            out = dst.createVariable(name, var.dtype, var.dimensions,
                                     zlib=(name == "topg"), complevel=1)
            out.setncatts({k: var.getncattr(k) for k in var.ncattrs()
                           if k != "_FillValue"})
            out[:] = var[:]
        thk = dst.createVariable("thk", "f4", ("y", "x"), zlib=True, complevel=1)
        thk.long_name = "ice thickness (ice-free start; see tools/make_input.py)"
        thk.units = "m"
        thk[:] = np.zeros((src.dimensions["y"].size, src.dimensions["x"].size), "f4")
        dst.history = f"built by tools/make_input.py from data/{SRC.name}"
    print(f"wrote {DST}  ({DST.stat().st_size/1e6:.0f} MB)")

CROP = DATA / "input_crop.nc"
X0, X1, Y0, Y1 = 520100.0, 681200.0, 5112100.0, 5245900.0      # cell EDGES of the sub-domain
with netCDF4.Dataset(DST) as src, netCDF4.Dataset(CROP, "w") as dst:
    x, y = src["x"][:], src["y"][:]
    ix = np.where((x > X0) & (x < X1))[0]; iy = np.where((y > Y0) & (y < Y1))[0]
    dst.createDimension("x", len(ix)); dst.createDimension("y", len(iy))
    for name, dims, arr in (("x", ("x",), x[ix]), ("y", ("y",), y[iy]),
                            ("topg", ("y", "x"), src["topg"][iy[0]:iy[-1]+1, ix[0]:ix[-1]+1]),
                            ("thk", ("y", "x"), np.zeros((len(iy), len(ix)), "f4"))):
        v = dst.createVariable(name, "f8" if name in ("x", "y") else "f4", dims, zlib=(name != "x"))
        v[:] = arr
    dst.history = f"accuracy sub-domain of {DST.name}, built by tools/make_input.py"
print(f"wrote {CROP}  ({len(ix)}x{len(iy)} @ 100 m)")
