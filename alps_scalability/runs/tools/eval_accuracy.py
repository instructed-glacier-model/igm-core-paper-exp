#!/usr/bin/env python3
"""
Accuracy of an online/offline run against the REFERENCE (identity-mapping) run, per grid and per
snapshot -> results/<test_tag>/accuracy.csv  (+ printed table).

Speed metrics:
  mask      = cells glaciated in the REFERENCE (thk_ref > 1 m)
  RMSE      = sqrt(mean((speed_test - speed_ref)^2)) over the mask          [m/yr]
  rel       = 100 * RMSE / mean(speed_ref over the mask)                     [%]
  vecRMSE   = sqrt(mean((u_t-u_r)^2 + (v_t-v_r)^2))                          [m/yr]
plus the geometry drift of the transient: ice-volume relative difference [%] and
thickness RMSE [m] (over cells glaciated in either run), and the metric quoted in the paper:
l2_velbar = sqrt(mean((velbar_test-velbar_ref)^2)) over ALL cells of the grid (no mask).

Both runs must have the same grids and snapshots:
  ./tools/eval_accuracy.py isoL_b1 ref_local          # tags under outputs/
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import netCDF4
import numpy as np

BENCH = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("test_tag")
    p.add_argument("ref_tag")
    args = p.parse_args()

    rows = []
    for tdir in sorted((BENCH / "outputs" / args.test_tag).glob("bench_r*/"), reverse=True):
        rdir = BENCH / "outputs" / args.ref_tag / tdir.name
        if not (tdir / "output.nc").exists() or not (rdir / "output.nc").exists():
            continue
        with netCDF4.Dataset(tdir / "output.nc") as T, netCDF4.Dataset(rdir / "output.nc") as R:
            dx = float(abs(R["x"][1] - R["x"][0]))
            # Match snapshots by TIME VALUE, not by index: a warm-started test run begins at
            # t = 300 yr while the reference begins at 0, so index k means different times.
            t_test = [float(v) for v in T["time"][:]]
            for k, t in enumerate(R["time"][:]):
                t = float(t)
                kt = next((i for i, v in enumerate(t_test) if abs(v - t) <= 1e-3), None)
                if kt is None:
                    continue
                hr, ht = np.asarray(R["thk"][k], float), np.asarray(T["thk"][kt], float)
                m = hr > 1.0
                if m.sum() == 0:
                    continue
                sr, st = np.asarray(R["velsurf_mag"][k], float), np.asarray(T["velsurf_mag"][kt], float)
                ok = m & np.isfinite(sr) & np.isfinite(st)
                rmse = float(np.sqrt(np.mean((st - sr)[ok] ** 2)))
                mean_ref = float(np.mean(sr[ok]))
                vec = float(np.sqrt(np.mean(((np.asarray(T["uvelsurf"][kt], float) - np.asarray(R["uvelsurf"][k], float)) ** 2
                                            + (np.asarray(T["vvelsurf"][kt], float) - np.asarray(R["vvelsurf"][k], float)) ** 2)[ok])))
                either = (hr > 1.0) | (ht > 1.0)
                l2b = (float(np.sqrt(np.mean((np.asarray(T["velbar_mag"][kt], float) - np.asarray(R["velbar_mag"][k], float)) ** 2)))
                       if "velbar_mag" in T.variables and "velbar_mag" in R.variables else np.nan)
                rows.append({
                    "ratio": int(tdir.name[-2:]), "dx_m": dx, "cells": hr.size, "time_yr": float(t),
                    "glaciated_cells": int(m.sum()), "mean_ref_speed": mean_ref,
                    "speed_rmse": rmse, "speed_rel_pct": 100 * rmse / mean_ref if mean_ref else np.nan,
                    "vector_rmse": vec,
                    "volume_rel_pct": 100 * (ht.sum() - hr.sum()) / hr.sum(),
                    "thk_rmse_m": float(np.sqrt(np.mean((ht - hr)[either] ** 2))),
                    "l2_velbar": l2b,
                })
    if not rows:
        raise SystemExit("no comparable run pairs found")
    out = BENCH / "results" / args.test_tag / "accuracy.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{'dx[m]':>7} {'t[yr]':>6} {'ice cells':>10} {'ref speed':>10} {'RMSE':>9} {'rel%':>7} {'vecRMSE':>9} {'vol%':>7} {'thkRMSE':>8} {'L2velbar':>9}")
    for r in rows:
        print(f"{r['dx_m']:>7.0f} {r['time_yr']:>6.0f} {r['glaciated_cells']:>10d} {r['mean_ref_speed']:>10.2f} "
              f"{r['speed_rmse']:>9.2f} {r['speed_rel_pct']:>7.1f} {r['vector_rmse']:>9.2f} "
              f"{r['volume_rel_pct']:>7.2f} {r['thk_rmse_m']:>8.2f} {r['l2_velbar']:>9.2f}")
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        by[r["dx_m"]].append(r["l2_velbar"])
    print("mean L2 velbar over the saves: " + ", ".join(
        f"dx {d:.0f} m: {np.nanmean(v):.2f} m/yr" for d, v in sorted(by.items(), reverse=True)))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
