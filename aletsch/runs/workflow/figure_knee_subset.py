#!/usr/bin/env python
"""Copy of the knee run read by the figure (../figure/data/knee_run/output.nc): the DEM dates and 2020 only, and the
four variables the figure uses (thk, usurf, uvelsurf, vvelsurf); values unchanged.
Usage: python workflow/figure_knee_subset.py (called by workflow/05_figure.sh)"""
from pathlib import Path

import netCDF4
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "outputs/knee_run/output.nc"
DST = ROOT.parent / "figure/data/knee_run/output.nc"
YEARS = [1880, 1926, 1957, 1980, 1999, 2009, 2017, 2020]
VARS = ["thk", "usurf", "uvelsurf", "vvelsurf"]

DST.parent.mkdir(parents=True, exist_ok=True)
with netCDF4.Dataset(SRC) as s, netCDF4.Dataset(DST, "w") as d:
    t = np.asarray(s["time"][:])
    k = [int(np.argmin(abs(t - y))) for y in YEARS]
    d.setncatts({a: s.getncattr(a) for a in s.ncattrs()})
    d.source = f"outputs/knee_run/output.nc, years {', '.join(map(str, YEARS))}"
    d.createDimension("time", len(k))
    for dim in ("y", "x"):
        d.createDimension(dim, len(s.dimensions[dim]))
    for name in ["time", "y", "x"] + VARS:
        v = s[name]
        o = d.createVariable(name, v.dtype, v.dimensions, zlib=True, complevel=6, shuffle=True,
                             fill_value=getattr(v, "_FillValue", None))
        o.setncatts({a: v.getncattr(a) for a in v.ncattrs() if a != "_FillValue"})
        o[:] = v[k] if v.dimensions[0] == "time" else v[:]
print(f"wrote {DST}")
