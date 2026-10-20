#!/usr/bin/env bash
# Inputs: data/input_bench.nc (full Alps, 9020x6020 @ 100 m, topg + zero thk; rebuilt only if the master bed
# data/input.nc is present) and data/input_crop.nc (160 x 134 km accuracy sub-domain). Both are shipped in data/.
set -e; source "$(dirname "$0")/_common.sh"
./tools/make_input.py
