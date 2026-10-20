#!/usr/bin/env bash
# Step 5 — Fig. 6 of the paper (aletsch_pareto), ~5 s (CPU).
#
#   in : analysis/optuna.db, outputs/knee_run/output.nc, data/input.nc, data/past_surf.nc
#        (copied to ../figure/data/; the knee run reduced to the DEM dates and 2020 by figure_knee_subset.py)
#   out: ../figure/aletsch_pareto.{pdf,svg,png} (Fig. 6 of the paper)
#
# Usage (from anywhere): workflow/05_figure.sh
set -euo pipefail
cd "$(dirname "$0")/.."
D=../figure/data; mkdir -p $D
python workflow/figure_knee_subset.py
for pair in analysis/optuna.db:$D/optuna.db data/input.nc:$D/input.nc data/past_surf.nc:$D/past_surf.nc; do
  src=${pair%%:*}; dst=${pair#*:}
  cmp -s "$src" "$dst" || { cp "$src" "$dst"; echo "updated $dst"; }
done
python ../figure/plot_aletsch_pareto_figure.py
