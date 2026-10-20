#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
Surface-speed misfit of Aletsch at the final time step (2020), on a fixed observation mask.

The mask is every valid observed pixel (finite, |u_obs| >= velsurfobs_thr), all inside the
2020 observed ice; the modelled speed counts as 0 where the model has no ice (thk <= 1), so a
retreating glacier is penalised.

Writes to state.score:
  cost_velsurf_union : RMSE(|u|_mod - |u|_obs) on the fixed observation mask  (m/yr)  calibration objective
  cost_velsurf       : same RMSE restricted to the modelled ice               (m/yr)  diagnostic
"""

import numpy as np


def initialize(cfg, state):
    pass


def update(cfg, state):
    pass


def finalize(cfg, state):
    for name in ("uvelsurf", "vvelsurf", "uvelsurfobs", "vvelsurfobs", "thk"):
        if not hasattr(state, name):
            print(f" track_velsurf_union: missing state.{name}, skipping velocity score")
            return

    thr = float(cfg.processes.track_velsurf_union.velsurfobs_thr)
    uv, vv = np.asarray(state.uvelsurf), np.asarray(state.vvelsurf)
    uo, vo = np.asarray(state.uvelsurfobs), np.asarray(state.vvelsurfobs)
    thk = np.asarray(state.thk)

    speed_obs = np.sqrt(uo * uo + vo * vo)
    speed_mod = np.where(thk > 1, np.sqrt(uv * uv + vv * vv), 0.0)

    obs_mask = np.isfinite(speed_obs) & (speed_obs >= thr)
    modelice_mask = obs_mask & (thk > 1)
    if obs_mask.sum() == 0:
        print(" track_velsurf_union: no valid velocity observations, skipping score")
        return

    rmse = float(np.sqrt(np.mean((speed_mod - speed_obs)[obs_mask] ** 2)))
    modelice = float(np.sqrt(np.mean((speed_mod - speed_obs)[modelice_mask] ** 2))) if modelice_mask.any() else float("inf")
    print(" Final surface speed RMSE, fixed obs mask : %8.2f m/yr (n=%d) | model ice only: %8.2f m/yr (n=%d)"
          % (rmse, int(obs_mask.sum()), modelice, int(modelice_mask.sum())))

    if not hasattr(state, "score"):
        state.score = {}
    state.score["cost_velsurf_union"] = rmse
    state.score["cost_velsurf"] = modelice
