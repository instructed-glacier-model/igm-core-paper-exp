#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
DEM misfit of Aletsch against the observed surfaces of data/past_surf.nc.

At each DEM year after the start (1926, 1957, 1980, 1999, 2009, 2017), on the union of the
observed ice (surf_obs - topg > 1) and the modelled ice (thk > 1):
  rmse = sqrt(mean(diff^2)),  bias = mean(diff),  diff = usurf_mod - surf_obs.

Writes to state.score:
  cost_usurf_rmse : mean over the DEM years of rmse                         (m)  calibration objective
  cost_usurf_bias : mean over the DEM years of |bias|                       (m)  diagnostic
  cost_usurf      : mean over all 7 years of std(diff) on modelled ice only (m)  diagnostic
"""

import numpy as np
import os
import tensorflow as tf
from netCDF4 import Dataset

OBS_YEARS = [1880, 1926, 1957, 1980, 1999, 2009, 2017]


def initialize(cfg, state):

    nc = Dataset(os.path.join(state.original_cwd, "data", 'past_surf.nc'), "r")
    for y in OBS_YEARS:
        vars(state)['surf_' + str(y)] = np.squeeze(nc.variables['surf_' + str(y)]).astype("float32")
    nc.close()

    state.usurf = tf.constant(vars(state)['surf_' + str(int(cfg.processes.time.start))])
    # the 1880 surface can lie a few metres below the bed: clamp thk to >= 0
    state.thk = tf.maximum(state.usurf - state.topg, 0.0)
    state.usurf = state.topg + state.thk

    state.track_stds, state.track_rmses, state.track_biases = [], [], []
    state.obs_next_idx = 0


def update(cfg, state):

    while state.obs_next_idx < len(OBS_YEARS):
        y = OBS_YEARS[state.obs_next_idx]
        if state.t < y:
            break
        state.obs_next_idx += 1

        obs = vars(state)['surf_' + str(y)]
        usurf, thk, topg = state.usurf.numpy(), state.thk.numpy(), np.asarray(state.topg)
        diff = usurf - obs

        std = float(np.std(diff[thk > 1]))
        union = (thk > 1) | ((obs - topg) > 1)
        rmse = float(np.sqrt(np.mean(diff[union] ** 2)))
        bias = float(np.mean(diff[union]))
        vol = float(np.sum(thk) * (state.dx ** 2) / 10 ** 9)
        print("\n DEM %4.0f : RMSE %6.2f  bias %+6.2f  (union mask) | Std %6.2f (model ice) | Volume %6.2f km3"
              % (state.t, rmse, bias, std, vol))

        state.track_stds.append(std)
        if y != int(cfg.processes.time.start):
            state.track_rmses.append(rmse)
            state.track_biases.append(bias)


def finalize(cfg, state):
    if len(getattr(state, 'track_rmses', [])) == 0:
        return
    if not hasattr(state, 'score'):
        state.score = {}
    state.score['cost_usurf_rmse'] = float(np.mean(state.track_rmses))
    state.score['cost_usurf_bias'] = float(np.mean(np.abs(state.track_biases)))
    state.score['cost_usurf'] = float(np.mean(state.track_stds))
    print(" DEM mean RMSE %8.2f | mean |bias| %8.2f | mean STD (model ice) %8.2f"
          % (state.score['cost_usurf_rmse'], state.score['cost_usurf_bias'], state.score['cost_usurf']))
