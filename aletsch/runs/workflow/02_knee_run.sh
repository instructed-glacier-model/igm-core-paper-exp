#!/usr/bin/env bash
# Step 2 — replay of the knee trial #259 with annual outputs, ~35 s on an RTX 4090.
#
#   in : experiment/params.yaml (its values are the knee parameters of analysis/optuna.db)
#   out: outputs/knee_run/{output.nc, output_ts.nc, _igm_score.json}
#        -> maps of panels (b, c) of the figure, and the mean absolute bias quoted in the text.
#
# Usage (from anywhere): workflow/02_knee_run.sh [GPU=0] [NAME=knee_run]
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${1:-0}; NAME=${2:-knee_run}
[ -e "outputs/$NAME" ] && { echo "outputs/$NAME exists — pick another name" >&2; exit 2; }
CUDA_VISIBLE_DEVICES=$GPU TF_CPP_MIN_LOG_LEVEL=3 igm_run +experiment=params hydra.run.dir=outputs/$NAME \
  > "outputs/$NAME.log" 2>&1
cat "outputs/$NAME/_igm_score.json"; echo
