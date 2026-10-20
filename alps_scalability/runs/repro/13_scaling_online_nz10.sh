#!/usr/bin/env bash
# Fig. 7b, third curve: the online sweep with the vertical fully resolved (Lagrange basis, Nz = 10) instead of
# MOLHO (Nz = 2); cost and memory only (no accuracy run). Same grids as repro/10.
# Usage: repro/13_scaling_online_nz10.sh <gpu_id> [tag]     (~20 min on an RTX 4090; GPU must be idle)
set -e; source "$(dirname "$0")/_common.sh"
./tools/run_sweep.py --gpu "${1:?gpu id}" --tag "${2:-rtx4090_local_nz10}" --igm-dir "$IGM_DIR" \
  --experiment params_bench --extra "processes.iceflow.numerics.Nz=10 processes.iceflow.numerics.basis_vertical=Lagrange" "${@:3}"
