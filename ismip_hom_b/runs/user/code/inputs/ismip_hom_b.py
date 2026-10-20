#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Geometry of ISMIP-HOM experiment B (Pattyn et al., 2008).

Flowline (two rows in y) of length L with a 0.5 deg surface slope and a
sinusoidal bed, frozen to the bed, periodic in x. Same setup as
ExperimentB in the test suite of the IGM repository (tests/test_iceflow/ismip_hom).
"""

import numpy as np
import tensorflow as tf


def run(cfg, state):

    L = float(cfg.inputs.ismip_hom_b.L)
    n = int(cfg.inputs.ismip_hom_b.n)
    dtype = tf.float32

    x = np.linspace(0.0, L, n)
    dx = x[1] - x[0]
    y = np.array([0.0, dx])
    X, Y = np.meshgrid(x, y)

    alpha = np.deg2rad(0.5)
    omega = 2 * np.pi / L
    usurf = -X * np.tan(alpha)
    thk = 1000.0 - 500.0 * np.sin(omega * X)

    state.x = tf.constant(x, dtype=dtype)
    state.y = tf.constant(y, dtype=dtype)
    state.X = tf.constant(X, dtype=dtype)
    state.Y = tf.constant(Y, dtype=dtype)
    state.dx = tf.Variable(dx, dtype=dtype)
    state.dX = tf.Variable(dx * np.ones_like(X), dtype=dtype)
    state.thk = tf.Variable(thk, dtype=dtype)
    state.topg = tf.Variable(usurf - thk, dtype=dtype)
    state.usurf = tf.Variable(usurf, dtype=dtype)
    state.tau_ref = tf.Variable(np.ones_like(X), dtype=dtype)
    state.arrhenius = tf.Variable(100.0 * np.ones_like(X), dtype=dtype)
