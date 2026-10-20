"""
bench_timer user module: per-iteration module times and GPU memory peak, written at finalize.

Writes the per-iteration timing lists (state.tcomp) to bench_steptimes.json, from which tools/run_sweep.py
takes the median over steps[1:] (step 1 carries the one-time XLA compile), and the TensorFlow allocator peak
to bench_memory.json.
"""
import json


def initialize(cfg, state):
    pass


def update(cfg, state):
    pass


def finalize(cfg, state):
    tcomp = {m: [float(x) for x in v]
             for m, v in getattr(state, "tcomp", {}).items()}
    out_file = cfg.processes.bench_timer.out_file
    with open(out_file, "w") as f:
        json.dump(tcomp, f, indent=0)
    print(f"[bench_timer] wrote {out_file} "
          f"({len(tcomp.get('iceflow', []))} iceflow iterations)")

    # TF allocator peak, read once at finalize
    try:
        import tensorflow as tf
        info = tf.config.experimental.get_memory_info("GPU:0")
        mem = {"tf_peak_mib": info["peak"] / 2**20, "tf_current_mib": info["current"] / 2**20}
    except Exception as e:                      # CPU run, or allocator without stats
        mem = {"error": str(e)}
    with open("bench_memory.json", "w") as f:
        json.dump(mem, f)
