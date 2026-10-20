#!/usr/bin/env bash
# Fig. 7b, offline (frozen pretrained dahunet_mini) + Fig. 7a (its 200 m map), same grids.
# Usage: repro/11_scaling_offline.sh <gpu_id> [tag] [extra run_sweep.py args, e.g. --ratios 12,6]
#        (~1 h 30 on an RTX 4090; GPU must be idle)
set -e; source "$(dirname "$0")/_common.sh"
./tools/run_sweep.py --gpu "${1:?gpu id}" --tag "${2:-rtx4090_offline}" --igm-dir "$IGM_DIR" \
  --experiment params_bench_offline "${@:3}"
