#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Surface air-temperature forcing for the Drønbreen experiment.

1. Builds the monthly cycle the enthalpy surface boundary condition needs: a cosine
   cycle between the CARRA mean annual and mean summer fields of the input file,
   rolled onto the hydrological year.
2. Above `ela`, replaces the air temperature by the constant `T_ela`, a proxy for the
   latent heat of meltwater refreezing in the firn, which IGM does not model.
   `ela` and `T_ela` are calibrated in Step 2.

No mass balance is produced: the geometry is fixed.
"""

import numpy as np
import tensorflow as tf
import xarray as xr

from igm.utils.math.interp1d_tf import interp1d_tf

MONTHS_PER_YEAR = 12

# Start of the hydrological year: 1 November (NH) / 1 May (SH), as a fraction of a year.
HYDRO_YEAR_START = {False: 2 / 12, True: 8 / 12}


def initialize(cfg, state):
    ccfg = cfg.processes.clim_carra_ela

    if ccfg.climate_change_array == []:
        state.climatepar = None
    else:
        state.climatepar = np.array(ccfg.climate_change_array[1:]).astype(np.float32)

    produce_climate_data(cfg, state)

    state.air_temp = tf.Variable(state.air_temp_ref, dtype="float32")
    state.air_temp_sd = tf.Variable(state.air_temp_sd_ref, dtype="float32")

    state.tlast_clim_update = tf.Variable(-np.inf, dtype="float32")


def update(cfg, state):
    """Refresh the monthly air temperature fields driving the enthalpy surface BC."""

    ccfg = cfg.processes.clim_carra_ela

    if (state.t - state.tlast_clim_update) < ccfg.update_freq:
        return

    # Always rebuild from the reference cycle, so offsets never accumulate.
    state.air_temp.assign(state.air_temp_ref)

    if ccfg.time_dependent_climate and state.climatepar is not None:
        temp_offset = interp1d_tf(
            state.climatepar[:, 0], state.climatepar[:, 1], state.t
        )
        state.air_temp.assign_add(
            tf.broadcast_to(temp_offset, tf.shape(state.air_temp))
        )

    # The refreezing proxy: a constant temperature everywhere above the ELA.
    above_ela = tf.cast(state.usurf > ccfg.ela, state.air_temp.dtype)
    state.air_temp.assign(
        state.air_temp * (1.0 - above_ela) + ccfg.T_ela * above_ela
    )

    state.tlast_clim_update.assign(state.t)


def finalize(cfg, state):
    pass


def produce_climate_data(cfg, state):
    """Expand the 2-D CARRA fields into the monthly cycle stored in `air_temp_ref`."""

    ccfg = cfg.processes.clim_carra_ela

    air_temp = tf.repeat(
        tf.expand_dims(state.air_temp, axis=0), MONTHS_PER_YEAR, axis=0
    )
    air_temp_sd = tf.repeat(
        tf.expand_dims(
            tf.fill(tf.shape(state.thk), ccfg.air_temperature_stdev), axis=0
        ),
        MONTHS_PER_YEAR,
        axis=0,
    )

    if ccfg.cosine_yearly_cycle_temp:
        months = np.arange(MONTHS_PER_YEAR)
        phase = 0 if ccfg.southern_hemisphere_climate else 6  # peak in January / July
        seasonal_cycle = np.cos(2 * np.pi * (months - phase) / MONTHS_PER_YEAR)
        seasonal_cycle = tf.reshape(
            tf.convert_to_tensor(seasonal_cycle, dtype=tf.float32), (12, 1, 1)
        )
        amplitude = tf.expand_dims(state.air_temp_summer - state.air_temp, axis=0)
        air_temp = air_temp + amplitude * seasonal_cycle

    if ccfg.export_climate_ref:
        ds = xr.Dataset(
            {
                "air_temp": (["time", "y", "x"], air_temp.numpy()),
                "air_temp_sd": (["time", "y", "x"], air_temp_sd.numpy()),
            },
            coords={
                "time": np.arange(MONTHS_PER_YEAR),
                "x": state.x.numpy(),
                "y": state.y.numpy(),
            },
        )
        ds.to_netcdf("climate_ref.nc")

    # Shift the calendar year onto the hydrological year.
    shift = HYDRO_YEAR_START[bool(ccfg.southern_hemisphere_climate)]
    shift_steps = int(MONTHS_PER_YEAR * (1 - shift))

    state.air_temp_ref = tf.constant(
        tf.roll(air_temp, shift=shift_steps, axis=0), dtype="float32"
    )
    state.air_temp_sd_ref = tf.constant(
        tf.roll(air_temp_sd, shift=shift_steps, axis=0), dtype="float32"
    )
