"""
bench_sync_* probe module (bench_sync_flow and bench_sync_thk are identical copies;
IGM loads a user module by file name, and the position in the processes list is what matters).

IGM times each module with time.time() around module.update() and never
synchronizes the GPU, while TensorFlow launches kernels asynchronously: a module
"ends" when its kernels are QUEUED. The remaining GPU work is paid by the next
module that needs a value on the host (e.g. `time`: python `if` on a tensor).
A bench_sync_* module does nothing but block until the GPU is idle, so
  * its own tcomp = the GPU work the PREVIOUS module left in flight,
  * the modules after it are timed cleanly.
"""
import tensorflow as tf


def sync():
    try:
        tf.test.experimental.sync_devices()
    except AttributeError:                      # older TF: fetch forces a stream sync
        tf.constant(0.0).numpy()


def initialize(cfg, state):
    pass


def update(cfg, state):
    sync()


def finalize(cfg, state):
    pass
