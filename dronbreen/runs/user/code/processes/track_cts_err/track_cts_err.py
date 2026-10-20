#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Misfit between the modelled and the radar-observed cold-temperate transition surface.

Input : GPR profiles in `data/cts_gpr/cts_<glacier>-<year>*.csv`, and the enthalpy
        fields `E` / `E_pmp` of the model state at the end of the run.
Output: `state.score["cost_cts"]`, the mean per-profile RMSE in metres, read by the
        Optuna sweeper. Used in Step 2 only.
"""

import sys

import numpy as np

# tools/cts.py is shared with make_figure_data.py. It is imported from the launch
# directory (runs/), since the run itself executes inside a Hydra output directory.
def _load_cts_tools(state):
    root = str(state.original_cwd)
    if root not in sys.path:
        sys.path.insert(0, root)

    from tools import cts

    return cts


def initialize(cfg, state):
    tcfg = cfg.processes.track_cts_err
    cts = _load_cts_tools(state)

    data_dir = state.original_cwd.joinpath(cfg.core.folder_data, tcfg.data_dir)
    state.cts_profiles = cts.load_cts_profiles(
        data_dir, tcfg.glacier_name, tcfg.year_obs
    )

    npoints = sum(len(p["distance"]) for p in state.cts_profiles)
    if hasattr(state, "logger"):
        state.logger.info(
            f"track_cts_err: loaded {len(state.cts_profiles)} radar lines "
            f"({npoints} points) from {data_dir}"
        )


def update(cfg, state):
    pass


def finalize(cfg, state):
    cts = _load_cts_tools(state)
    nz, vert_spacing = cts.enthalpy_grid(cfg)

    x, y = state.x.numpy(), state.y.numpy()
    E, E_pmp = np.asarray(state.E), np.asarray(state.E_pmp)
    topg, thk = np.asarray(state.topg), np.asarray(state.thk)

    rmse_per_profile = []
    for profile in state.cts_profiles:
        east, north = profile["easting"], profile["northing"]

        cts_model = cts.cts_elevation(
            cts.sample_columns(E, x, y, east, north),
            cts.sample_columns(E_pmp, x, y, east, north),
            cts.sample_columns(topg, x, y, east, north),
            cts.sample_columns(thk, x, y, east, north),
            vert_spacing,
        )

        rmse = cts.profile_rmse(cts_model, profile["cts_obs"])
        if np.isfinite(rmse):
            rmse_per_profile.append(rmse)
            if hasattr(state, "logger"):
                state.logger.info(f"track_cts_err: {profile['name']} RMSE {rmse:.1f} m")

    # No comparable point: failed run, given a large finite cost.
    cost = float(np.mean(rmse_per_profile)) if rmse_per_profile else 1.0e6

    if hasattr(state, "logger"):
        state.logger.info(
            f"track_cts_err: mean CTS RMSE over {len(rmse_per_profile)} "
            f"profiles = {cost:.2f} m"
        )

    if not hasattr(state, "score"):
        state.score = {}
    state.score["cost_cts"] = cost
