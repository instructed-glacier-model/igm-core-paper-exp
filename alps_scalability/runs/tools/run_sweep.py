#!/usr/bin/env python3
"""
Scalability sweep driver — full-frame IGM compute time vs grid size.

Runs `igm_run +experiment=<experiment>` once per coarsening ratio, sequentially on one dedicated GPU,
samples the peak GPU memory, and writes one row per grid to results/<tag>/scaling.<gpuN|cpu>.csv
(rewritten after each ratio). Provenance (GPU, IGM commit, TF/Keras versions) goes to results/<tag>/meta.json.

Per-step ice-flow time: iceflow + the bench_sync_flow probe that follows it (GPU-synchronised), median of
steps[1:] (step 1 carries the one-time XLA compile), plus p10/p90. A run that runs out of memory is recorded
as status=OOM. The GPU must be idle; a previous run dir of the same name is moved to output_save/.

Usage (conda env igm):
  ./tools/run_sweep.py --gpu 0 --tag rtx4090_local
  ./tools/run_sweep.py --gpu 0 --tag test --ratios 60,32,16,8
  ./tools/run_sweep.py --cpu   --tag cpu_test --ratios 32,16 --extra processes.time.end=20
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent          # runs/
EXPERIMENT = "params_bench"
STATS_NAME = "computational-statistics.txt"
IGM_DIR = BENCH_DIR / "igm"      # project-local IGM (made by repro/00_env.sh)

GPU_IDLE_MIB = 500               # above this the card is considered busy

# Base input grid (regional domain). Coarsening ratio r -> ~base // r cells.
BASE_NX, BASE_NY, BASE_DX_M = 9020, 6020, 100.0
DEFAULT_RATIOS = [60, 40, 32, 24, 16, 12, 8, 6, 5, 4, 3, 2]

OOM_PAT = re.compile(r"ResourceExhausted|OOM|out of memory|CUDA_ERROR_OUT_OF_MEMORY",
                     re.IGNORECASE)
# "   <module>  |  mean time per it : 0.1234  |  total : 5.6789"
STAT_PAT = re.compile(r"^\s*(\S+)\s*\|\s*mean time per it\s*:\s*([0-9.eE+-]+)\s*\|"
                      r"\s*total\s*:\s*([0-9.eE+-]+)")

COLS = ["ratio", "status", "grid_measured", "nx", "ny", "dx_m", "cells", "cells_M", "n_iter",
        "warmup_excluded", "gpu_synced", "iceflow_mean_s", "iceflow_unsynced_s", "iceflow_avg_s", "iceflow_p10_s", "iceflow_p90_s",
        "step_mean_s", "throughput_cells_per_s", "first_solve_s",
        "peak_gpu_mib", "tf_peak_mib", "wall_s"]


def sh(cmd: list[str], **kw) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20,
                              **kw).stdout.strip()
    except Exception:
        return ""


def grid_for_ratio(r: int) -> tuple[int, int, float]:
    """(Nx, Ny, dx_m) after coarsening with boundary=trim."""
    return BASE_NX // r, BASE_NY // r, BASE_DX_M * r


def parse_stats(path: Path) -> dict:
    out: dict = {}
    if path.exists():
        for line in path.read_text().splitlines():
            m = STAT_PAT.match(line)
            if m:
                out[m.group(1)] = {"mean": float(m.group(2)),
                                   "total": float(m.group(3))}
    return out


def parse_steptimes(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _body(xs) -> list[float]:
    """Per-iteration times without the INIT step (index 0)."""
    xs = [float(x) for x in xs]
    return xs[1:] if len(xs) > 1 else xs


WARMUP_STEPS = 10      # skipped by the MEAN: init + first inference/retrain compiles


def _steady(xs) -> float:
    """Steady-state per-step time: median of steps[1:] — robust to the single
    XLA-compile outlier at the first solve, whatever index it lands on. Right for
    a config whose steps are all alike (retrain every step, or never)."""
    body = _body(xs)
    return statistics.median(body) if body else float("nan")


def _steady_mean(xs) -> float:
    """MEAN over steps[WARMUP_STEPS:] (column iceflow_avg_s; diagnostic, not used in the paper)."""
    xs = [float(x) for x in xs][WARMUP_STEPS:]
    return sum(xs) / len(xs) if xs else float("nan")


def _quantile(xs, q: float) -> float:
    body = sorted(_body(xs))
    if not body:
        return float("nan")
    return body[min(len(body) - 1, max(0, int(round(q * (len(body) - 1)))))]


def read_actual_grid(run_out: Path) -> tuple[int | None, int | None]:
    try:
        import netCDF4
        with netCDF4.Dataset(run_out / "output.nc") as ds:
            return ds.dimensions["x"].size, ds.dimensions["y"].size
    except Exception:
        return None, None


def gpu_used_mib(gpu: int) -> int:
    out = sh(["nvidia-smi", "--query-gpu=memory.used",
              "--format=csv,noheader,nounits", "-i", str(gpu)])
    try:
        return int(out.splitlines()[0])
    except Exception:
        return -1


class GpuMemSampler(threading.Thread):
    """Poll nvidia-smi used memory on one (dedicated) GPU; keep the peak (MiB)."""

    def __init__(self, gpu: int, period: float = 0.5):
        super().__init__(daemon=True)
        self.gpu, self.period, self.peak = gpu, period, -1
        self._stop_evt = threading.Event()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            self.peak = max(self.peak, gpu_used_mib(self.gpu))
            self._stop_evt.wait(self.period)

    def stop(self) -> int:
        self._stop_evt.set()
        self.join(timeout=3)
        return self.peak


def collect_meta(args, ratios, extra) -> dict:
    igm_dir = args.igm_dir.resolve() if args.igm_dir else IGM_DIR
    meta = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": "cpu" if args.cpu else "gpu",
        "tag": args.tag, "ratios": ratios, "extra_overrides": extra,
        "experiment": args.experiment,
        "igm_dir": os.path.relpath(igm_dir, BENCH_DIR),
        "igm_branch": sh(["git", "-C", str(igm_dir), "rev-parse", "--abbrev-ref", "HEAD"]),
        "igm_commit": sh(["git", "-C", str(igm_dir), "describe", "--tags", "--always", "--long"]),
        "igm_dirty_files": sh(["git", "-C", str(igm_dir), "status", "--porcelain",
                               "--untracked-files=no"]).splitlines(),
        "cpu_model": next((l.split(":", 1)[1].strip() for l in
                           Path("/proc/cpuinfo").read_text().splitlines()
                           if l.startswith("model name")), ""),
        "cpu_count": os.cpu_count(),
    }
    if not args.cpu:
        q = sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits", "-i", str(args.gpu)])
        parts = [p.strip() for p in q.split(",")] if q else []
        if len(parts) == 3:
            meta.update(gpu_index=args.gpu, gpu_name=parts[0],
                        gpu_mem_mib=int(parts[1]), driver=parts[2])
    vers = sh([sys.executable, "-c",
               "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3';"
               "import tensorflow as tf, keras; print(tf.__version__, keras.__version__)"],
              env=dict(os.environ, CUDA_VISIBLE_DEVICES="", TF_USE_LEGACY_KERAS="0"))
    if vers:
        meta["tensorflow"], meta["keras"] = vers.split()[-2:]
    return meta


def worker_id(args) -> str:
    return "cpu" if args.cpu else f"gpu{args.gpu}"


def run_one(ratio: int, args, extra: list[str], out_root: Path,
            res_dir: Path) -> dict | None:
    nx_est, ny_est, dx = grid_for_ratio(ratio)
    run_out = out_root / f"bench_r{ratio:02d}"
    log_path = out_root / f"bench_r{ratio:02d}.log"
    cmd = ["igm_run", f"+experiment={args.experiment}",
           f"inputs.local.coarsening.ratio={ratio}",
           f"hydra.run.dir={run_out}", *extra]
    env = dict(os.environ, TF_USE_LEGACY_KERAS="0", TF_CPP_MIN_LOG_LEVEL="3",
               PYTHONUNBUFFERED="1",
               CUDA_VISIBLE_DEVICES="" if args.cpu else str(args.gpu))
    igm_dir = args.igm_dir.resolve() if args.igm_dir else IGM_DIR   # takes precedence over the install
    env["PYTHONPATH"] = str(igm_dir) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    print(f"\n[bench] ratio={ratio:>3}  {nx_est}x{ny_est} @ {dx:.0f} m "
          f"({nx_est*ny_est/1e6:.2f} M cells)  "
          f"{'cpu' if args.cpu else f'gpu={args.gpu}'}", flush=True)
    print("        " + " ".join(cmd), flush=True)
    if args.dry_run:
        return {"ratio": ratio, "status": "dry-run"}

    # keep previous outputs: park any previous run of the same name
    if run_out.exists() or log_path.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        park = BENCH_DIR / "output_save" / out_root.name / f"bench_r{ratio:02d}_{stamp}"
        park.parent.mkdir(parents=True, exist_ok=True)
        if run_out.exists():
            shutil.move(str(run_out), str(park))
        if log_path.exists():
            shutil.move(str(log_path), str(park) + ".log")

    sampler = None if args.cpu else GpuMemSampler(args.gpu)
    if sampler:
        sampler.start()
    t0 = time.monotonic()
    limit = args.timeout_min * 60 if args.timeout_min else None
    timed_out = False
    with log_path.open("w") as log:
        try:
            proc = subprocess.run(cmd, cwd=BENCH_DIR, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc = subprocess.CompletedProcess(cmd, returncode=-9)
    wall = time.monotonic() - t0
    peak_mib = sampler.stop() if sampler else -1

    oom = OOM_PAT.search(log_path.read_text(errors="replace")[-20000:]) is not None
    status = "ok" if proc.returncode == 0 else ("TIMEOUT" if timed_out else "OOM" if oom else "FAIL")

    stats = parse_stats(run_out / STATS_NAME)
    if stats:
        shutil.copy(run_out / STATS_NAME, res_dir / f"stats_r{ratio:02d}.txt")
    nx, ny = read_actual_grid(run_out)
    # grid_measured=False: no output.nc, nx/ny estimated from the full-domain grid
    grid_measured = nx is not None and ny is not None
    nx, ny = nx or nx_est, ny or ny_est
    cells = nx * ny

    tcomp = parse_steptimes(run_out / "bench_steptimes.json")
    # GPU-synchronized timing: fold each sync probe into the module it follows
    # (per iteration), so iceflow = its host time + the GPU work it left in flight.
    unsynced = _steady(tcomp["iceflow"]) if len(tcomp.get("iceflow", [])) > 1 else float("nan")
    synced = False
    for probe, owner in (("bench_sync_flow", "iceflow"), ("bench_sync_thk", "thk")):
        if probe in tcomp and owner in tcomp and len(tcomp[probe]) == len(tcomp[owner]):
            tcomp[owner] = [a + b for a, b in zip(tcomp[owner], tcomp.pop(probe))]
            synced = synced or owner == "iceflow"
    flow = tcomp.get("iceflow", [])
    nan = float("nan")
    if len(flow) > 1:
        shutil.copy(run_out / "bench_steptimes.json",
                    res_dir / f"steptimes_r{ratio:02d}.json")
        warm, n_iter = True, len(flow)
        iceflow = _steady(flow)
        iceflow_avg = _steady_mean(flow)
        p10, p90 = _quantile(flow, 0.10), _quantile(flow, 0.90)
        first_solve = max(flow[:3])
        module_means = {m: _steady(v) for m, v in tcomp.items() if len(v) > 1}
        step_mean = sum(module_means.values())
    else:       # fallback: IGM's all-steps mean (includes the compile step)
        warm = False
        f = stats.get("iceflow", {})
        iceflow, p10, p90, first_solve = f.get("mean", nan), nan, nan, nan
        iceflow_avg = nan
        n_iter = int(round(f["total"] / f["mean"])) if f.get("mean", 0) > 0 else 0
        module_means = {m: v["mean"] for m, v in stats.items()}
        step_mean = sum(module_means.values()) if stats else nan
    if status != "ok":          # a partial run has no trustworthy steady state
        iceflow = iceflow_avg = p10 = p90 = step_mean = nan

    thr = cells / iceflow if iceflow == iceflow and iceflow > 0 else nan
    row = {
        "ratio": ratio, "status": status, "grid_measured": grid_measured,
        "nx": nx, "ny": ny, "dx_m": dx,
        "cells": cells, "cells_M": round(cells / 1e6, 4), "n_iter": n_iter,
        "warmup_excluded": warm, "gpu_synced": synced, "iceflow_mean_s": iceflow,
        "iceflow_unsynced_s": unsynced, "iceflow_avg_s": iceflow_avg,
        "iceflow_p10_s": p10, "iceflow_p90_s": p90, "step_mean_s": step_mean,
        "throughput_cells_per_s": thr, "first_solve_s": first_solve,
        "peak_gpu_mib": peak_mib,
        "tf_peak_mib": round(parse_steptimes(run_out / "bench_memory.json")
                             .get("tf_peak_mib", float("nan")), 1),
        "wall_s": round(wall, 1),
    }
    if status == "ok":
        for mod, v in module_means.items():
            row[f"mean__{mod}"] = v

    print(f"        -> status={status}  iceflow/step={iceflow:.4g}s  "
          f"thr={thr:,.0f} cells/s  peakGPU={peak_mib} MiB (TF peak {row['tf_peak_mib']})  "
          f"({n_iter} its, {wall:.0f}s)", flush=True)
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    extra = sorted({k for r in rows for k in r if k not in COLS})
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS + extra)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, default=0, help="dedicated GPU id (default 0)")
    p.add_argument("--cpu", action="store_true", help="CPU-only run (hides the GPUs)")
    p.add_argument("--tag", required=True,
                   help="name of this sweep -> results/<tag>/, outputs/<tag>/")
    p.add_argument("--ratios", default=",".join(map(str, DEFAULT_RATIOS)),
                   help="coarsening ratios, coarsest-first (default: %(default)s)")
    p.add_argument("--igm-dir", type=Path, default=None,
                   help="IGM checkout to run (via PYTHONPATH) and to record in meta.json "
                        "(default: the project-local igm/)")
    p.add_argument("--experiment", default=EXPERIMENT,
                   help="experiment yaml name (default: %(default)s)")
    p.add_argument("--extra", default="",
                   help="extra hydra overrides for every igm_run, space-separated")
    p.add_argument("--continue-after-oom", action="store_true",
                   help="keep sweeping finer grids after an OOM (default: stop)")
    p.add_argument("--timeout-min", type=float, default=0,
                   help="kill a single run after this many minutes (status=TIMEOUT); 0 = none")
    p.add_argument("--dry-run", "-n", action="store_true")
    args = p.parse_args()

    if not args.cpu and not args.dry_run:
        used = gpu_used_mib(args.gpu)
        if used > GPU_IDLE_MIB:
            sys.exit(f"[bench] GPU {args.gpu} is busy ({used} MiB in use): "
                     "the benchmark needs a DEDICATED card. Not starting.")

    extra = args.extra.split() if args.extra else []
    ratios = [int(x) for x in args.ratios.split(",") if x.strip()]
    res_dir = BENCH_DIR / "results" / args.tag
    out_root = BENCH_DIR / "outputs" / args.tag
    who = worker_id(args)
    csv_path = res_dir / f"scaling.{who}.csv"
    if not args.dry_run:
        res_dir.mkdir(parents=True, exist_ok=True)
        out_root.mkdir(parents=True, exist_ok=True)
        meta = json.dumps(collect_meta(args, ratios, extra), indent=1)
        (res_dir / "meta.json").write_text(meta)

    rows = []
    if csv_path.exists() and not args.dry_run:      # relaunch of the same worker: keep its rows
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
    for r in ratios:
        row = run_one(r, args, extra, out_root, res_dir)
        if args.dry_run or row is None:
            continue
        rows.append(row)
        write_csv(rows, csv_path)
        if rows and rows[-1]["status"] == "OOM" and not args.continue_after_oom:
            print("[bench] OOM reached — finer grids skipped.")
            break
    if not args.dry_run:
        print(f"\n[bench] wrote {csv_path}")


if __name__ == "__main__":
    main()
