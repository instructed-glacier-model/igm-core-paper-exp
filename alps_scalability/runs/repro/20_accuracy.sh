#!/usr/bin/env bash
# "Same accuracy" statement: 160 x 134 km sub-domain, references (identity mapping + L-BFGS) 0->500 yr at
# 1200/600/300 m, warm inputs at t=300, online/offline warm runs 300->500 yr, L2 vs reference.
# Usage: repro/20_accuracy.sh <gpu_id>          (~3 h on an RTX 4090, mostly the 300 m reference)
set -e; source "$(dirname "$0")/_common.sh"
export CUDA_VISIBLE_DEVICES="${1:?gpu id}"
run(){ local exp=$1 r=$2 od=$3; shift 3
       mkdir -p "$(dirname "$od")"
       igm_run +experiment="$exp" inputs.local.coarsening.ratio="$r" "$@" hydra.run.dir="$od" > "$od.log" 2>&1; }
for r in 12 6 3; do rr=$(printf %02d $r)
  run params_crop_ref $r "$ROOT/outputs/ref_local/bench_r$rr"
  ./tools/make_warm_input.py outputs/ref_local/bench_r$rr --time 300 -o data/warm_local_r$rr.nc
done
# online: 1 Adam iteration per step after 2000 initial iterations = isoL_b1; offline = warm_local_offline
for rr in 12 06 03; do
  run params_warm_online 1 "$ROOT/outputs/isoL_b1/bench_r$rr" \
      inputs.local.filename="$ROOT/data/warm_local_r$rr.nc"
  run params_warm_offline 1 "$ROOT/outputs/warm_local_offline/bench_r$rr" \
      inputs.local.filename="$ROOT/data/warm_local_r$rr.nc"
done
for t in isoL_b1 warm_local_offline; do mkdir -p results/$t
  ./tools/eval_accuracy.py $t ref_local > results/$t/accuracy.txt; done
# paper: rows dx = 300 m, t = 500 yr: column vol% (~+1 %) and last column L2velbar (~7 m/yr)
grep -E "^ +300 +500" results/isoL_b1/accuracy.txt results/warm_local_offline/accuracy.txt
