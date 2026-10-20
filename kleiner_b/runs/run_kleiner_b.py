#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Enthalpy benchmark, experiment B of Kleiner et al. (2015).

Polythermal parallel-sided slab (H = 200 m, slope 4 deg, surface temperature
-3 C, accumulation 0.2 m/yr) with a prescribed velocity field and strain
heating. The IGM enthalpy and subglacial hydrology (till storage) modules are
stepped directly (dt = 1 yr, 500 enthalpy levels) until steady state.

Writes to outputs/, read by ../figure/make_figure.py:
  kleiner_exp_b_results.csv      IGM enthalpy, temperature, water content
  kleiner_exp_b_analytical.csv   analytical solution on the same vertical grid

Same setup as tests/test_enthalpy/kleiner/exp_b in the IGM repository.
"""

import os

import numpy as np
import tensorflow as tf

import igm
from analytical_solutions import validate_exp_b
from igm.common import State
from igm.common.runner.configuration.loader import load_yaml_recursive
from igm.processes import subglacial_hydrology
from igm.processes.enthalpy import compute_variables_enthalpy_state, enthalpy

H = 200.0
GAMMA = 4.0 * np.pi / 180.0
A = 5.3e-24
RHO, G = 910.0, 9.81
SPY = 31556926.0
A_PERP = 0.2
NZ_U = 10
NZ_E = 500
NY, NX = 2, 2
MAX_ITER = 1500
TOL = 1e-3


def setup():
    cfg = load_yaml_recursive(os.path.join(igm.__path__[0], "conf"))

    cfg.processes.iceflow.numerics.Nz = NZ_U
    cfg.processes.iceflow.numerics.vert_spacing = 1
    cfg.processes.enthalpy.numerics.Nz = NZ_E
    cfg.processes.enthalpy.numerics.vert_spacing = 1

    cfg.processes.enthalpy.thermal.K_ratio = 1e-5
    cfg.processes.enthalpy.drainage.drain_ice_column = False
    cfg.processes.enthalpy.solver.allow_basal_refreezing = False
    cfg.processes.enthalpy.solver.correct_w_for_melt = False

    # Till storage with a large cap and no drainage, as in the benchmark
    cfg.processes.subglacial_hydrology.mode = "till_storage"
    cfg.processes.subglacial_hydrology.till_storage.h_water_till_max = 200.0
    cfg.processes.subglacial_hydrology.till_storage.drainage_rate = 0.0

    # Table A1 of Kleiner et al. (2015): no Clausius-Clapeyron, L = 3.35e5
    cfg.processes.enthalpy.thermal.beta = 0.0
    cfg.processes.enthalpy.thermal.L_ice = 3.35e5

    state = State()
    state.thk = tf.Variable(H * tf.ones((NY, NX)), trainable=False)
    state.topg = tf.Variable(tf.zeros((NY, NX)), trainable=False)
    state.usurf = state.topg + state.thk
    state.t = tf.Variable(0.0, trainable=False)
    state.dt = tf.Variable(1.0, trainable=False)
    state.air_temp = tf.Variable(-3.0 * tf.ones((1, NY, NX)), trainable=False)
    state.dx = tf.Variable(1000.0, trainable=False)
    state.dX = tf.Variable(1000.0 * tf.ones((NY, NX)), trainable=False)
    state.h_water_till = tf.zeros((NY, NX))
    state.tau_ref = tf.zeros((NY, NX))
    state.basal_heat_flux = tf.zeros((NY, NX))

    # Prescribed velocity field (Eqs. 13-15 of Kleiner et al., 2015)
    z = tf.reshape(tf.linspace(0.0, H, NZ_U), (NZ_U, 1, 1))
    z = tf.tile(z, [1, NY, NX])
    tau = RHO * G * tf.sin(GAMMA)
    vx = A * (tau**3) / 2 * (H**4 - (H - z) ** 4) * SPY
    state.U = tf.Variable(vx, trainable=False)
    state.V = tf.Variable(tf.zeros((NZ_U, NY, NX)), trainable=False)
    state.W = tf.Variable(-A_PERP * tf.ones((NZ_U, NY, NX)), trainable=False)

    # Initial state: ice slightly below the melting point
    c_ice = cfg.processes.enthalpy.thermal.c_ice
    T_ref = cfg.processes.enthalpy.thermal.T_ref
    state.E = c_ice * ((273.15 - 1.5) * tf.ones((NZ_E, NY, NX)) - T_ref)
    state.arrhenius = tf.constant(A * 1e18 * SPY, dtype=tf.float32) * tf.ones((NY, NX))
    enthalpy.initialize(cfg, state)

    return cfg, state


def run(cfg, state):
    dt = state.dt.numpy()
    prev_E = state.E.numpy().copy()

    for it in range(MAX_ITER):
        state.t.assign(it * dt)
        state.arrhenius = (A * 1e18 * SPY) * tf.ones_like(state.arrhenius)
        enthalpy.update(cfg, state)
        subglacial_hydrology.update(cfg, state)

        if it % 10 == 0 and it > 0:
            max_change = np.max(np.abs(state.E.numpy() - prev_E))
            if max_change < TOL:
                print(f"Converged at iteration {it}")
                break
            prev_E = state.E.numpy().copy()
            if it % 100 == 0:
                compute_variables_enthalpy_state(cfg, state)
                T_base = state.T[0, 0, 0].numpy() - 273.15
                print(f"Iteration {it}: T_base = {T_base:.2f} C")

    compute_variables_enthalpy_state(cfg, state)


def extract(state):
    z = np.linspace(0, state.thk[0, 0].numpy(), state.T.shape[0])
    E = state.E[:, 0, 0].numpy()
    T = state.T[:, 0, 0].numpy() - 273.15
    omega = state.omega[:, 0, 0].numpy() * 100

    cts_idx = np.where(omega > 0.01)[0]
    if len(cts_idx) > 0:
        cts_position = z[cts_idx[-1]]
    else:
        cts_position = 0.0

    return {"z": z, "E": E, "T": T, "omega": omega, "cts_position": cts_position}


def main():
    cfg, state = setup()
    run(cfg, state)
    results = extract(state)
    validation = validate_exp_b(results)

    print(
        f"CTS: IGM {validation['cts']['numerical']:.2f} m, "
        f"analytical {validation['cts']['analytical']:.2f} m"
    )
    print(f"RMSD enthalpy:      {validation['enthalpy']['rmsd']:.1f} J/kg")
    print(f"RMSD temperature:   {validation['temperature']['rmsd']:.3f} C")
    print(f"RMSD water content: {validation['water_content']['rmsd']:.3f} %")

    path_outputs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(path_outputs, exist_ok=True)
    columns = {
        "results": [results["z"], results["E"], results["T"], results["omega"]],
        "analytical": [
            results["z"],
            validation["enthalpy"]["analytical"],
            validation["temperature"]["analytical"],
            validation["water_content"]["analytical"],
        ],
    }
    for name, data in columns.items():
        path_csv = os.path.join(path_outputs, f"kleiner_exp_b_{name}.csv")
        np.savetxt(
            path_csv,
            np.column_stack(data),
            delimiter=",",
            header="z_m,E_J_per_kg,T_C,omega_pct",
            comments="",
        )
        print(f"Written {path_csv}")


if __name__ == "__main__":
    main()
