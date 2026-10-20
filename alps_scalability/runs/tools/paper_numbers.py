#!/usr/bin/env python
"""
Print the numbers quoted in the paper section "Scalability test", computed from the result files.

  ./tools/paper_numbers.py                  # reads results/{rtx4090_local,rtx4090_offline,rtx4090_local_nz10,cpu55_local,
                                            #               isoL_b1,warm_local_offline}

"paper:" = the value as written in the paper; "data:" = the value computed from the result files.
Definitions: see README.md, section "Metrics".
"""
import csv
import glob
from pathlib import Path

import netCDF4
import numpy as np

B = Path(__file__).resolve().parent.parent
R = B / "results"


def sweep(tag):
    f = glob.glob(str(R / tag / "scaling.*.csv"))[0]
    rows = [r for r in csv.DictReader(open(f))]
    return {int(r["ratio"]): r for r in rows}


def f(r, k):
    return float(r[k])


def acc(tag, ratio=3, t=500.0):
    for r in csv.DictReader(open(R / tag / "accuracy.csv")):
        if int(r["ratio"]) == ratio and float(r["time_yr"]) == t:
            return r


on, off, cpu, nz10 = sweep("rtx4090_local"), sweep("rtx4090_offline"), sweep("cpu55_local"), sweep("rtx4090_local_nz10")
ok = lambda s: {k: r for k, r in s.items() if r["status"] == "ok"}
run = lambda r: int(r["n_iter"]) * f(r, "iceflow_mean_s")
lines = []


def P(claim, paper, value):
    """One number: its value in the paper and in the data."""
    lines.append(f"{claim:<58s} paper: {paper:<14s} data: {value}")

floor = [f(r, "iceflow_mean_s") * 1e3 for k, r in ok(on).items() if int(r["cells"]) < 1e5]
P("latency floor, online, N < 1e5 cells", "flat", f"{min(floor):.1f}-{max(floor):.1f} ms/step")
big = [r for r in ok(on).values() if int(r["cells"]) >= 2e6]
x, y = np.log([int(r["cells"]) for r in big]), np.log([f(r, "iceflow_mean_s") for r in big])
P("log-log slope, online, N >= 2e6", "1.00", f"{np.polyfit(x, y, 1)[0]:.2f}")
P("throughput, online, N >= 2e6", "31 M cells/s",
  ", ".join(f"{f(r, 'throughput_cells_per_s') / 1e6:.1f}" for r in big) + " M cells/s")
P("per-step at 300 m (6.0e6 cells), online / offline", "~0.2 s",
  f"{f(on[3], 'iceflow_mean_s'):.3f} / {f(off[3], 'iceflow_mean_s'):.3f} s")
P("memory per M cells, online (300 m)", "3.3 GiB",
  f"{f(on[3], 'tf_peak_mib') / 1024 / (int(on[3]['cells']) / 1e6):.2f} GiB ({f(on[3], 'tf_peak_mib') / 1024:.1f} GiB peak)")
P("online at 200 m (1.4e7 cells) on 24 GB", "does not fit", on[2]["status"])
P("memory per M cells, offline (200 m)", "1.1 GiB",
  f"{f(off[2], 'tf_peak_mib') / 1024 / (int(off[2]['cells']) / 1e6):.2f} GiB ({f(off[2], 'tf_peak_mib') / 1024:.1f} GiB peak)")
P("cost x per halving, online 1200->600 m", "8.1", f"{run(on[6]) / run(on[12]):.2f}")
P("cost x per halving, online 600->300 m", "7.9", f"{run(on[3]) / run(on[6]):.2f}")
P("step count x per halving, online 1200->600->300 m", "2.8, 2.0",
  f"{int(on[6]['n_iter']) / int(on[12]['n_iter']):.2f}, {int(on[3]['n_iter']) / int(on[6]['n_iter']):.2f}")
P("cost x per halving, online, coarse grids (3200->1600, 2400->1200)", "1.4-2.0",
  f"{run(on[16]) / run(on[32]):.2f}, {run(on[12]) / run(on[24]):.2f}")
P("500-yr run at 400 m, online, wall time", "~6 min", f"{f(on[4], 'wall_s') / 60:.1f} min")
P("CPU (32 cores) / GPU at 400 m, whole step", "~120",
  f"{f(cpu[4], 'step_mean_s') / f(on[4], 'step_mean_s'):.0f} (ice flow only: "
  f"{f(cpu[4], 'iceflow_mean_s') / f(on[4], 'iceflow_mean_s'):.0f})")
P("500-yr run at 200 m, offline, wall time", "< 1 h", f"{f(off[2], 'wall_s') / 60:.0f} min")
a_on, a_off = acc("isoL_b1"), acc("warm_local_offline")
P("L2 depth-averaged speed at 300 m, t=500, online / offline", "5.2 / 2.3 m/yr",
  f"{f(a_on, 'l2_velbar'):.1f} / {f(a_off, 'l2_velbar'):.1f} m/yr")
P("ice volume vs reference after 200 yr, online / offline", "within 1.0 %",
  f"{f(a_on, 'volume_rel_pct'):+.2f} / {f(a_off, 'volume_rel_pct'):+.2f} %")
for dx_, ratio_ in ((600, 6), (1200, 12)):
    o_, f_ = acc("isoL_b1", ratio_), acc("warm_local_offline", ratio_)
    P(f"L2 at {dx_} m, t=500, online / offline", "2.3 -> 99 (offline)" if dx_ == 1200 else "(fig. only)",
      f"{f(o_, 'l2_velbar'):.1f} / {f(f_, 'l2_velbar'):.1f} m/yr")
z = ok(nz10); zf = min(z)
big_z = [k for k in z if k in ok(on) and int(z[k]["cells"]) >= 1e6]
P("online Nz=10 (Lagrange) / Nz=2 (MOLHO), per step, N >= 1e6", "~10 % more",
  f"x{np.mean([f(z[k], 'iceflow_mean_s') / f(on[k], 'iceflow_mean_s') for k in big_z]):.2f}")
P("online Nz=10: finest grid that fits on 24 GB", "(fig. only)",
  f"{f(z[zf], 'dx_m'):.0f} m ({f(z[zf], 'tf_peak_mib') / 1024:.1f} GiB, {f(z[zf], 'tf_peak_mib') / 1024 / (int(z[zf]['cells']) / 1e6):.1f} GiB per M cells)")
m = netCDF4.Dataset(B.parent / "figure/data/map_offline_200m_t500.nc")
thk, dx = np.asarray(m["thk"][0]), float(m["x"][1] - m["x"][0])
ice = thk > 1.0
cols = np.where(ice.any(axis=0))[0]
P("ice area at t=500 yr, 200 m offline (thk > 1 m)", "~35 000 km2", f"{ice.sum() * dx**2 / 1e6:,.0f} km2")
P("west-east extent of the ice", "nearly 700 km", f"{(cols[-1] - cols[0]) * dx / 1e3:.0f} km")
P("maximum ice thickness", "up to 1400 m", f"{thk.max():.0f} m")
print("\n".join(lines))
