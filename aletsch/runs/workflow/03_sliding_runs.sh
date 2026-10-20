#!/usr/bin/env bash
# Step 3 — replays of the three witness trials saving the basal speed, ~35 s each.
#
#   in : experiment/params.yaml + the parameters of the witness trials of analysis/optuna.db
#        (high τ_ref #224, knee #259, low τ_ref #279)
#   out: outputs/sliding_{224,259,279}/output.nc (thk, usurf, velsurf_mag, velbase_mag)
#        -> glacier-wide sliding fraction u_b/u_s in 2020 quoted in the text (step 4).
#
# Usage (from anywhere): workflow/03_sliding_runs.sh [GPU=0]
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${1:-0}
P=processes.smb_accmelt; T=processes.iceflow.physics.sliding.tau_ref
SAVE="outputs.local.vars_to_save=[thk,usurf,velsurf_mag,velbase_mag] outputs.local.write_ts=False"
declare -A RUN=(
  [224]="$T=0.408838297729456 $P.weight_accumulation=1.29608830386264 $P.weight_ablation=1.57705539117014 $P.weight_Aletschfirn=1.54509655480356 $P.weight_Jungfraufirn=1.07113818837321 $P.weight_Ewigschneefeld=1.30669789119262"
  [259]="$T=0.378424390845927 $P.weight_accumulation=1.29608830386264 $P.weight_ablation=1.51426221885031 $P.weight_Aletschfirn=1.51060580888852 $P.weight_Jungfraufirn=1.07113818837321 $P.weight_Ewigschneefeld=1.08716385787098"
  [279]="$T=0.310820421557892 $P.weight_accumulation=1.29608830386264 $P.weight_ablation=1.51426221885031 $P.weight_Aletschfirn=1.51060580888852 $P.weight_Jungfraufirn=1.07113818837321 $P.weight_Ewigschneefeld=1.22549712528879"
)
for n in 224 259 279; do
  [ -e "outputs/sliding_$n" ] && { echo "outputs/sliding_$n exists — skipped"; continue; }
  CUDA_VISIBLE_DEVICES=$GPU TF_CPP_MIN_LOG_LEVEL=3 igm_run +experiment=params ${RUN[$n]} $SAVE \
    hydra.run.dir=outputs/sliding_$n > "outputs/sliding_$n.log" 2>&1
  cat "outputs/sliding_$n/_igm_score.json"; echo
done
