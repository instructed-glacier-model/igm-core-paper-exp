#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Enthalpy benchmark, experiment A of Kleiner et al. (2015).

A 1000 m thick, motionless ice slab with geothermal heat flux 0.042 W m-2,
forced by a surface temperature of -30 C (0-100 kyr), -5 C (100-150 kyr) and
-30 C again (150-300 kyr). The IGM enthalpy and subglacial hydrology (till
storage) modules are stepped directly with dt = 200 yr and 50 enthalpy levels.

Writes outputs/kleiner_exp_a_results.csv (time, basal temperature, basal melt
rate, till water thickness), read by ../figure/make_figure.py.

Same setup as tests/test_enthalpy/kleiner/exp_a in the IGM repository.
"""

import os

import numpy as np
import tensorflow as tf

import igm
from igm.common import State
from igm.common.runner.configuration.loader import load_yaml_recursive
from igm.processes import subglacial_hydrology
from igm.processes.enthalpy import compute_variables_enthalpy_state, enthalpy

DT = 200.0
T_END = 300000.0
NZ_U = 10
NZ_E = 50
NY, NX = 2, 2


def setup():
    cfg = load_yaml_recursive(os.path.join(igm.__path__[0], "conf"))

    cfg.processes.iceflow.numerics.Nz = NZ_U
    cfg.processes.iceflow.numerics.vert_spacing = 1
    cfg.processes.enthalpy.numerics.Nz = NZ_E
    cfg.processes.enthalpy.numerics.vert_spacing = 1

    cfg.processes.enthalpy.thermal.K_ratio = 1e-5
    cfg.processes.enthalpy.drainage.drain_ice_column = False
    cfg.processes.enthalpy.solver.allow_basal_refreezing = True
    cfg.processes.enthalpy.solver.correct_w_for_melt = False

    # Till storage with a large cap and no drainage, as in the benchmark
    cfg.processes.subglacial_hydrology.mode = "till_storage"
    cfg.processes.subglacial_hydrology.till_storage.h_water_till_max = 200.0
    cfg.processes.subglacial_hydrology.till_storage.drainage_rate = 0.0

    state = State()
    state.thk = tf.Variable(1000.0 * tf.ones((NY, NX)), trainable=False)
    state.topg = tf.Variable(tf.zeros((NY, NX)), trainable=False)
    state.usurf = state.topg + state.thk
    state.t = tf.Variable(0.0, trainable=False)
    state.dt = tf.Variable(DT, trainable=False)
    state.air_temp = tf.Variable(-30.0 * tf.ones((1, NY, NX)), trainable=False)
    state.dx = tf.Variable(1000.0, trainable=False)
    state.dX = tf.Variable(1000.0 * tf.ones((NY, NX)), trainable=False)
    state.h_water_till = tf.zeros((NY, NX))
    state.tau_ref = tf.zeros((NY, NX))
    state.basal_heat_flux = 0.042 * tf.ones((NY, NX))
    state.U = tf.Variable(tf.zeros((NZ_U, NY, NX)), trainable=False)
    state.V = tf.Variable(tf.zeros((NZ_U, NY, NX)), trainable=False)
    state.W = tf.Variable(tf.zeros((NZ_U, NY, NX)), trainable=False)

    # Initial state: cold ice at -30 C
    enthalpy.initialize(cfg, state)
    c_ice = cfg.processes.enthalpy.thermal.c_ice
    T_ref = cfg.processes.enthalpy.thermal.T_ref
    state.E = c_ice * (243.15 * tf.ones((NZ_E, NY, NX)) - T_ref)

    return cfg, state


def run(cfg, state):
    times = np.arange(DT, T_END + 1.0, DT)
    results = {"time": [], "T_base": [], "melt_rate": [], "till_water": []}

    for it, t in enumerate(times):
        state.t.assign(t)

        if 100000.0 <= t < 150000.0:
            T_surface = -5.0
        else:
            T_surface = -30.0
        state.air_temp.assign(T_surface * tf.ones((1, NY, NX)))

        enthalpy.update(cfg, state)
        subglacial_hydrology.update(cfg, state)
        compute_variables_enthalpy_state(cfg, state)

        results["time"].append(t)
        results["T_base"].append(state.T[0, 0, 0].numpy() - 273.15)
        results["melt_rate"].append(state.basal_melt_rate[0, 0].numpy())
        results["till_water"].append(state.h_water_till[0, 0].numpy())

        if it % 100 == 0:
            print(f"t = {t / 1000:5.0f} kyr, T_base = {results['T_base'][-1]:6.2f} C")

    return results


def main():
    cfg, state = setup()
    results = run(cfg, state)

    path_outputs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(path_outputs, exist_ok=True)
    path_csv = os.path.join(path_outputs, "kleiner_exp_a_results.csv")
    np.savetxt(
        path_csv,
        np.column_stack(
            [
                results["time"],
                results["T_base"],
                results["melt_rate"],
                results["till_water"],
            ]
        ),
        delimiter=",",
        header="time_yr,T_base_C,melt_rate_m_per_yr,till_water_m",
        comments="",
    )
    print(f"Written {path_csv}")


if __name__ == "__main__":
    main()
