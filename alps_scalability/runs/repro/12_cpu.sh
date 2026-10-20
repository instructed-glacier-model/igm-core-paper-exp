#!/usr/bin/env bash
# GPU-vs-CPU ratio (text: "about 100 times" at 400 m): online config, 20 yr only, 4 grids, CPU only.
# Usage: repro/12_cpu.sh [tag]      (~8 min on a 32-core Xeon E5-2683 v4)
set -e; source "$(dirname "$0")/_common.sh"
./tools/run_sweep.py --cpu --tag "${1:-cpu55_local}" --igm-dir "$IGM_DIR" --ratios 32,16,8,4 \
  --extra "processes.time.end=20"
