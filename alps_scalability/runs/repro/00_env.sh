#!/usr/bin/env bash
# Once: IGM and the two conda envs.
#   igm/      project-local IGM v3.2.1 (cloned if missing), put first on PYTHONPATH by every script
#   igm       conda env of the runs (python 3.12, TF 2.17.0, Keras 3.14.1)  <- env/conda_igm.yml
#   igm-fig   conda env of the figure (cartopy, cmocean)                      <- env/conda_igm-fig.yml
set -e; cd "$(dirname "$0")/.."
[ -d igm/.git ] || { git clone https://github.com/instructed-glacier-model/igm.git igm && git -C igm checkout v3.2.1; }
git -C igm describe --tags
conda env create -n "${IGM_ENV:-igm}" -f env/conda_igm.yml
conda env create -n "${FIG_ENV:-igm-fig}" -f env/conda_igm-fig.yml
