#!/usr/bin/env bash
# Paper Fig. 7 (../figure/fig_scalability.{pdf,png,svg}) and the numbers of the text, from results/. CPU, ~1 min.
# tools/make_figure_data.py copies the figure's data to ../figure/data/ (the 200 m map is re-extracted only if
# outputs/rtx4090_offline/bench_r02/output.nc exists). The figure runs in the conda env FIG_ENV (default igm-fig).
set -e; source "$(dirname "$0")/_common.sh"
conda run -n "${FIG_ENV:-igm-fig}" python tools/make_figure_data.py
conda run -n "${FIG_ENV:-igm-fig}" python ../figure/fig_scalability.py
./tools/paper_numbers.py
