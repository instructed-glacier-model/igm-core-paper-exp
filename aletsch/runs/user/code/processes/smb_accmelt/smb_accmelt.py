#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""

Mass balance forced by climate with accumulation and temperature-index melt model [Hock, 1999; Huss et al., 2009].
It is a TensorFlow re-implementation of the model used in simulations reported in
(Jouvet and al., Numerical simulation of Rhone's glacier from 1874 to 2100, JCP, 2009)
(Jouvet and al., Modelling the retreat of Grosser Aletschgletscher in a changing climate, JOG, 2011)
(Jouvet and Huss, Future retreat of great Aletsch glacier, JOG, 2019)
(Jouvet and al., Mapping the age of ice of Gauligletscher ..., TC, 2020)
The model is described in the last reference.

==============================================================================

Input: data/massbalance.nc, data/bassin.nc, data/mbparameter.dat, state.air_temp, state.precipitation
Output: state.smb
"""

import os
import time

import numpy as np
import tensorflow as tf
from netCDF4 import Dataset
from scipy.interpolate import interp1d


def initialize(cfg, state):
    """
        load smb data to run the Aletsch Glacier simulation
    """

    filepath = state.original_cwd.joinpath(os.path.join("data", 'massbalance.nc'))
    nc = Dataset(filepath, "r")

    state.snow_redistribution = np.squeeze(nc.variables["snow_redistribution"]
    ).astype("float32")
    state.direct_radiation = np.squeeze(nc.variables["direct_radiation"]).astype(
        "float32"
    )
    nc.close()


    nc = Dataset(state.original_cwd.joinpath(os.path.join("data", 'bassin.nc')), "r" )
    x = np.squeeze(nc.variables["x"]).astype("float32")   # grid of the basin masks, checked below
    y = np.squeeze(nc.variables["y"]).astype("float32")

    state.Aletschfirn = np.squeeze( nc.variables["Aletschfirn"] ).astype("float32")
    state.Ewigschneefeld = np.squeeze( nc.variables["Ewigschneefeld"] ).astype("float32")
    state.Jungfraufirn = np.squeeze( nc.variables["Jungfraufirn"] ).astype("float32")

    nc.close()


    if not hasattr(state, "x"):
        state.x = tf.constant(x, dtype="float32")
        state.y = tf.constant(y, dtype="float32")
    else:
        assert x.shape == state.x.shape
        assert y.shape == state.y.shape

    # resample at daily resolution the direct radiation field,
    da = np.arange(20, 351, 30)
    db = np.arange(1, 366)
    state.direct_radiation = interp1d(
        da, state.direct_radiation, kind="linear", axis=0, fill_value="extrapolate")(db)

    state.direct_radiation = tf.Variable(state.direct_radiation, dtype="float32")

    # read mass balance parameters
    state.mb_parameters = tf.Variable(
        np.loadtxt(
            state.original_cwd.joinpath(os.path.join("data", 'mbparameter.dat')),
            skiprows=2,
            dtype=np.float32,
        )
    )

    # Weight the snow redistribution in the three accumulation basins: factor (2 + w_b)/3 inside basin b
    if not ((cfg.processes.smb_accmelt.weight_Aletschfirn==1.0)&(cfg.processes.smb_accmelt.weight_Jungfraufirn==1.0)&(cfg.processes.smb_accmelt.weight_Ewigschneefeld==1.0)):
        state.snow_redistribution *= (1.0/3.0)* \
            ( cfg.processes.smb_accmelt.weight_Aletschfirn    * state.Aletschfirn     + (1-state.Aletschfirn) \
            + cfg.processes.smb_accmelt.weight_Jungfraufirn   * state.Jungfraufirn    + (1-state.Jungfraufirn) \
            + cfg.processes.smb_accmelt.weight_Ewigschneefeld * state.Ewigschneefeld  + (1-state.Ewigschneefeld) )

    # snow_redistribution must have shape like precipitation (365,ny,nx)
    state.snow_redistribution = tf.expand_dims(state.snow_redistribution, axis=0)
    state.snow_redistribution = tf.tile(state.snow_redistribution, (365, 1, 1))

    # define the year-corresponding indice in the mb_parameters file
    state.IMB = np.zeros((221), dtype="int32")
    for tt in range(1880, 2101):
        state.IMB[tt - 1880] = np.argwhere(
            (state.mb_parameters[:, 0] <= tt) & (tt < state.mb_parameters[:, 1])
        )[0]
    state.IMB = tf.Variable(state.IMB)

    if "time" not in cfg.processes:
        raise ValueError("The 'time' module is required to use the 'smb_accmelt' module.")


    # Initial SMB, used before the first update
    state.smb = tf.ones_like(state.topg) * (cfg.processes.smb_accmelt.wat_density / cfg.processes.smb_accmelt.ice_density)

    state.tlast_smb = tf.Variable(cfg.processes.time.start)
    state.tcomp_smb = []

def update(cfg, state):

    if ((state.t - state.tlast_smb) >= cfg.processes.smb_accmelt.update_freq):

        if hasattr(state, "logger"):
            state.logger.info("update smb at time : " + str(state.t.numpy()))

        state.tcomp_smb.append(time.time())

        # melt parameters of the current year
        Fm = state.mb_parameters[state.IMB[int(state.t) - 1880], 2] * 10 ** (-3)
        ri = state.mb_parameters[state.IMB[int(state.t) - 1880], 3] * 10 ** (-5)
        rs = state.mb_parameters[state.IMB[int(state.t) - 1880], 4] * 10 ** (-5)

        # keep solid precipitation when temperature < thr_temp_snow
        # with linear transition to 0 between thr_temp_snow and thr_temp_rain
        accumulation = tf.where(
            state.air_temp <= cfg.processes.smb_accmelt.thr_temp_snow,
            state.precipitation,
            tf.where(
                state.air_temp >= cfg.processes.smb_accmelt.thr_temp_rain,
                0.0,
                state.precipitation
                * (cfg.processes.smb_accmelt.thr_temp_rain - state.air_temp)
                / (cfg.processes.smb_accmelt.thr_temp_rain - cfg.processes.smb_accmelt.thr_temp_snow),
            ),
        )

        # unit to [  kg * m^(-2) * y^(-1) water eq  ] -> [ m water eq / d]
        accumulation /= (accumulation.shape[0] * cfg.processes.smb_accmelt.wat_density)

        # correct for snow re-distribution
        accumulation *= state.snow_redistribution  # unit to [ m water eq. / d ]

        accumulation *= cfg.processes.smb_accmelt.weight_accumulation  # calibration factor w_acc

        pos_temp = tf.where(state.air_temp > 0.0, state.air_temp, 0.0)  # unit is [°C]

        # daily snow-depth / ablation recursion over the hydrological year (starts Oct. 1)
        shift = int(state.air_temp.shape[0] * cfg.processes.smb_accmelt.shift_hydro_year)
        ablation = _annual_ablation_sum(
            accumulation, pos_temp, state.direct_radiation, Fm, ri, rs,
            tf.constant(cfg.processes.smb_accmelt.weight_ablation, dtype=tf.float32),
            tf.constant(shift),
        )  # [ unit : water-eq. m / y ]

        # sum accumulation and ablation over the year, and conversion to ice equivalent
        state.smb = (tf.math.reduce_sum(accumulation, axis=0) - ablation) * (
            cfg.processes.smb_accmelt.wat_density / cfg.processes.smb_accmelt.ice_density
        )

        if hasattr(state,'icemask'):  # -10 m ice eq./yr outside the ice mask
            state.smb  = tf.where((state.smb<0)|(state.icemask>0.5),state.smb,-10)

        state.tlast_smb.assign(state.t)

        state.tcomp_smb[-1] -= time.time()
        state.tcomp_smb[-1] *= -1


@tf.function
def _annual_ablation_sum(accumulation, pos_temp, direct_radiation, Fm, ri, rs, w_abl, shift):
    """Yearly sum of the daily ablation [m w.e.]: days visited from `shift` (hydro year), snow
    depth carried day to day; melt factor for ice (ri) when snow depth is 0, else snow (rs)."""
    n = tf.shape(accumulation)[0]
    idx = (tf.range(n) + shift) % n
    acc = tf.gather(accumulation, idx)
    pt = tf.gather(pos_temp, idx)
    dr = tf.gather(direct_radiation, idx)

    def step(carry, x):
        snow_depth, abl_sum = carry
        a, p, d = x
        # add accumulation to the snow depth
        snow_depth = snow_depth + a
        # ablation = positive temp x melt factor (Fm + r * direct radiation) [m ice eq./(°C d)]
        abl = tf.where(snow_depth == 0, p * (Fm + ri * d) * 0.91, p * (Fm + rs * d) * 0.91) * w_abl
        # remove melt from the snow depth, which can not be negative
        snow_depth = tf.clip_by_value(snow_depth - abl, 0.0, 1.0e10)
        return snow_depth, abl_sum + abl

    zero = tf.zeros_like(acc[0])
    _, abl_sum = tf.foldl(step, (acc, pt, dr), initializer=(zero, zero))
    return abl_sum


def finalize(cfg, state):
    pass
