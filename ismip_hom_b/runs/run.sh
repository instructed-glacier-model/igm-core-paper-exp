#!/usr/bin/env bash
# ISMIP-HOM experiment B with IGM: one diagnostic ice-flow solve per length L,
# then extraction of the CSV files read by ../figure/make_figure.py.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

for L in 20 40 80 160; do
  igm_run +experiment=params \
    inputs.ismip_hom_b.L=${L}e3 \
    hydra.run.dir=outputs/${L}km
done

python make_csv.py
