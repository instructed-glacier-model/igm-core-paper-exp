#!/usr/bin/env bash
# Step 1 — calibration sweep (NSGA-II, 6 parameters, 2 objectives), ~100 min on one RTX 4090.
#
#   in : experiment/params.yaml, optuna/optuna_params.yaml, user/, data/
#   out: analysis/optuna_rerun.db (study aletsch_rerun); per-trial scores in outputs/sweep_rerun/wK/
#
# NWORKERS independent workers on one GPU share one Optuna study (n_trials is per worker: 10 x 60).
# No fixed seed: a rerun gives an equivalent, not identical, front.
#
# Usage (from anywhere): workflow/01_sweep.sh [NWORKERS=10] [GPU=0]
set -euo pipefail
cd "$(dirname "$0")/.."
NW=${1:-10}; GPU=${2:-0}; NAME=sweep_rerun; CFG=optuna/optuna_params.yaml
[ -e "outputs/$NAME" ] && { echo "outputs/$NAME exists" >&2; exit 2; }
python - "$CFG" <<'PY'   # create the study first, so the workers do not race on an empty DB
import sys, yaml, optuna
c = yaml.safe_load(open(sys.argv[1]))
assert c.get("n_jobs", 1) == 1
optuna.create_study(storage=c["storage"], study_name=c["study_name"],
                    directions=[o["direction"] for o in c["objectives"]], load_if_exists=True)
PY
mkdir -p "outputs/$NAME"
for ((k = 0; k < NW; k++)); do
  CUDA_VISIBLE_DEVICES=$GPU TF_CPP_MIN_LOG_LEVEL=3 igm_run -m +experiment=params hydra/sweeper=igm_optuna \
    hydra.sweeper.optuna_config=$CFG hydra.sweep.dir=outputs/$NAME/w$k \
    'outputs.local.file_format_list=[]' outputs.local.write_ts=False > "outputs/${NAME}_w$k.log" 2>&1 &
  sleep 2
done
wait
echo "done: analysis/optuna_rerun.db"
