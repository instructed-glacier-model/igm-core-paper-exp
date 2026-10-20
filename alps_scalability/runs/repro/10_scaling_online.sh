#!/usr/bin/env bash
# Fig. 7b, online: full Alps, 500 yr from ice-free, ratios 60..2 (6 km .. 200 m; 200 m OOMs on 24 GB).
# Usage: repro/10_scaling_online.sh <gpu_id> [tag] [extra run_sweep.py args, e.g. --ratios 12,6]
#        (~30 min on an RTX 4090; GPU must be idle)
set -e; source "$(dirname "$0")/_common.sh"
./tools/run_sweep.py --gpu "${1:?gpu id}" --tag "${2:-rtx4090_local}" --igm-dir "$IGM_DIR" \
  --experiment params_bench "${@:3}"
