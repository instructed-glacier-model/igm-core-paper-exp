"""
bench_seed user module: reproducible online runs (network built from scratch).

At initialize: tf.config.experimental.enable_op_determinism() + tf.keras.utils.set_random_seed(seed).
Must come before `iceflow` in the processes list, so that the network weights are drawn after seeding.
Op-determinism is part of the timed configuration. Used by the online configs only.
"""
import tensorflow as tf


def initialize(cfg, state):
    p = cfg.processes.bench_seed
    if p.op_determinism:
        tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(int(p.seed))


def update(cfg, state):
    pass


def finalize(cfg, state):
    pass
