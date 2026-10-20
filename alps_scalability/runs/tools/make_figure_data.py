#!/usr/bin/env python
"""
Write the data read by the figure script (../figure/fig_scalability.py) into ../figure/data/, so that the
figure can be redrawn without the raw runs:
  data/rtx4090_local/scaling*.csv       online sweep (copy of results/rtx4090_local/scaling*.csv)
  data/rtx4090_offline/scaling*.csv     offline sweep (copy of results/rtx4090_offline/scaling*.csv)
  data/rtx4090_local_nz10/scaling*.csv  online sweep with Nz = 10 (copy of results/rtx4090_local_nz10/scaling*.csv)
  data/isoL_b1/accuracy.csv             accuracy runs, online  (copy of results/isoL_b1/accuracy.csv)
  data/warm_local_offline/accuracy.csv  accuracy runs, offline (copy of results/warm_local_offline/accuracy.csv)
  data/map_offline_200m_t500.nc         t = 500 yr slice of outputs/rtx4090_offline/bench_r02/output.nc
                                        (x, y, time, thk, usurf, velsurf_mag; values unchanged), rewritten
                                        only if that run exists.
Usage (from anywhere): ./tools/make_figure_data.py
"""
import shutil
from pathlib import Path

import netCDF4
import numpy as np

B = Path(__file__).resolve().parent.parent          # runs/
FIG = B.parent / "figure" / "data"
T_MAP = 500.0

for tag, pattern in (("rtx4090_local", "scaling*.csv"), ("rtx4090_offline", "scaling*.csv"),
                     ("rtx4090_local_nz10", "scaling*.csv"), ("isoL_b1", "accuracy.csv"),
                     ("warm_local_offline", "accuracy.csv")):
    (FIG / tag).mkdir(parents=True, exist_ok=True)
    for f in sorted((B / "results" / tag).glob(pattern)):
        shutil.copy2(f, FIG / tag / f.name)
        print(f"wrote {FIG / tag / f.name}")

src = B / "outputs/rtx4090_offline/bench_r02/output.nc"
dst = FIG / "map_offline_200m_t500.nc"
if not src.exists():
    print(f"{src.relative_to(B)} not found: keeping {dst}")
    raise SystemExit(0)
with netCDF4.Dataset(src) as s, netCDF4.Dataset(dst, "w") as d:
    k = int(np.argmin(abs(np.asarray(s["time"][:]) - T_MAP)))
    d.setncatts({a: s.getncattr(a) for a in s.ncattrs()})
    d.source = f"{src.relative_to(B)}, time index {k}"
    d.createDimension("time", 1)
    for dim in ("y", "x"):
        d.createDimension(dim, len(s.dimensions[dim]))
    for name in ("time", "y", "x", "thk", "usurf", "velsurf_mag"):
        v = s[name]
        o = d.createVariable(name, v.dtype, v.dimensions, zlib=True, complevel=6, shuffle=True,
                             fill_value=getattr(v, "_FillValue", None))
        o.setncatts({a: v.getncattr(a) for a in v.ncattrs() if a != "_FillValue"})
        o[:] = v[k:k + 1] if v.dimensions[0] == "time" else v[:]
print(f"wrote {dst}")
