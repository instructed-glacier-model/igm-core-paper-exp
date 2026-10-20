# Alps scalability test (IGM GMD paper, Sect. "Scalability test", Fig. 7)

The same 500-yr run of the European Alps (900 × 600 km, ice-free start, constant SMB with the ELA lowered to
2300 m) on grids coarsened from a 100 m bed by integer ratios 60 … 2 (Δx = 6 km … 200 m, 1.5 × 10⁴ → 1.4 × 10⁷
cells). Ice-flow time, GPU memory and accuracy are measured for two ice-flow regimes:

| regime | network | trainable params | per step | config |
|---|---|---|---|---|
| online | dahunet 6 layers × 24 filters, kernel 3, random initialisation (seed 1) | 3.2 × 10⁴ | 1 Adam iteration | `experiment/params_bench.yaml` |
| offline | `dahunet_mini.keras` shipped with IGM, 8 × 48, kernel 5, frozen | 4.6 × 10⁵ | inference only | `experiment/params_bench_offline.yaml` |

Common to every run: MOLHO vertical basis with Nz = 2, basal sliding tau_ref = 0.2154 at u_ref = 100 m/yr (IGM
defaults). Horizontal quadrature: cell-centred (IGM default) for the online network; 2 × 2 Gauss on Q1 elements for
the offline network (the configuration it was trained in) and for the reference. A third sweep
(`repro/13_scaling_online_nz10.sh`) repeats the online sweep with a Lagrange vertical basis and Nz = 10 (vertical
fully resolved), for cost and memory only.

The cost (Fig. 7) is measured on the full 0 → 500 yr runs of the whole domain. The online network starts from
random weights at t = 0 (`nbit_init: 0`), when there is no ice yet, and learns as the ice cover grows.

## Accuracy check

The accuracy of the two regimes is measured separately, on a 160 × 134 km sub-domain (`data/input_crop.nc`) at
Δx = 1200, 600 and 300 m, with the same forcing and time stepping as the sweep:

1. Reference (`params_crop_ref.yaml`): IGM's direct solver, i.e. the Blatter–Pattyn energy minimised on the
   velocity field without network (identity mapping, L-BFGS, 50 iterations per step, MOLHO Nz = 2), from 0 to 500 yr.
2. Its state at t = 300 yr, a developed ice cover, is the initial state of both regimes
   (`data/warm_local_r{12,06,03}.nc`), which then run to 500 yr. Starting from the same state isolates the error
   of the ice-flow solver over 200 yr.
   - online (`params_warm_online.yaml`): the network of the sweep, with the same training (1 Adam iteration at
     every step); as it starts from random weights on an existing ice cover, it first makes 2000 Adam iterations
     (`nbit_init: 2000`);
   - offline (`params_warm_offline.yaml`): the pretrained network of the sweep, inference only.
3. At t = 500 yr, both are compared with the reference (`tools/eval_accuracy.py`): L2 error of the
   depth-averaged speed over all cells (`l2_velbar`) and ice-volume difference (`volume_rel_pct`).

| Δx | online L2 / volume | offline L2 / volume |
|---|---|---|
| 1200 m | 22.4 m/yr / +2.7 % | 98.8 m/yr / −29.6 % |
| 600 m | 9.4 m/yr / +1.3 % | 10.6 m/yr / +2.2 % |
| 300 m | 5.2 m/yr / +1.0 % | 2.3 m/yr / +0.8 % |

The L2 errors are the bottom row of Fig. 7b. The paper quotes the 300 m values ("5.2 and 2.3 m yr⁻¹, volume within
1.0 %"), the resolution at which the costs are compared, and the growth of the offline error from 2.3 to 99 m yr⁻¹
between 300 and 1200 m: outside its 50–250 m training range the pretrained network degrades, whereas the online
network, retrained at each resolution, keeps a moderate error. Only these accuracy numbers come from the
300 → 500 yr runs. (Measured against the Q1 reference, the online error includes the small difference between the
two quadratures: 5.2 instead of 4.5 m yr⁻¹ at 300 m with Q1 for the online network too.)

## Data (`data/`)

| file | content |
|---|---|
| `input_bench.nc` | full domain, 9020 × 6020 @ 100 m (UTM 32N): bed `topg` + zero `thk` |
| `input_crop.nc` | 160 × 134 km accuracy sub-domain (made by `tools/make_input.py`) |
| `warm_local_r{12,06,03}.nc` | reference state at t = 300 yr at 1200 / 600 / 300 m (made by `repro/20_accuracy.sh`) |

Source of the 100 m bed topography: SRTM DEM (Farr et al., 2007, doi:[10.1029/2005RG000183](https://doi.org/10.1029/2005RG000183)) with major lakes and present-day glaciers removed (ice thickness of Farinotti et al., 2019, doi:[10.1038/s41561-019-0300-3](https://doi.org/10.1038/s41561-019-0300-3)), as in Jouvet et al. (2023, doi:[10.1017/jog.2023.74](https://doi.org/10.1017/jog.2023.74)) and Leger et al. (2025, doi:[10.1038/s41467-025-56168-3](https://doi.org/10.1038/s41467-025-56168-3)).

## Software and hardware

- IGM v3.2.1, as a project-local clone `igm/` made by `repro/00_env.sh` and put first on `PYTHONPATH`. The recorded
  runs used v3.2.1, except the offline sweep (`rtx4090_offline`), run on commit `f379a0b6`, whose code is identical
  to v3.2.1 for this experiment.
- Conda envs: `env/conda_igm.yml` (runs: python 3.12.13, TensorFlow 2.17.0, Keras 3.14.1) and
  `env/conda_igm-fig.yml` (figure: matplotlib, netCDF4, cartopy, cmocean). If conda is not in `~/anaconda3`,
  set `CONDA_SH=/path/to/conda.sh`.
- User modules in `user/`: `bench_seed` (seed and op-determinism, online runs), `bench_sync_flow` and
  `bench_sync_thk` (wait for the GPU so that module times include their GPU work), `bench_timer` (per-step times
  and TensorFlow memory peak).
- GPU: one dedicated NVIDIA RTX 4090 (24 GB), driver 580.178.04. CPU comparison: 32-core Intel Xeon E5-2683 v4.
- The recorded online runs (`rtx4090_local`, `rtx4090_local_nz10`, `cpu55_local`, `isoL_b1`) were made with the
  cell-centred quadrature passed on the command line (`extra_overrides` in `meta.json`); it is now set in the
  configs, so the repro scripts need no override.

## Reproduce

Run from this folder. The second argument of the scaling scripts is the tag (results go to `results/<tag>/`).

| step | command | output | cost |
|---|---|---|---|
| 0 env | `repro/00_env.sh` | `igm/`, conda envs `igm` and `igm-fig` | once |
| 1 inputs | `repro/01_inputs.sh` | `data/input_bench.nc`, `data/input_crop.nc` (already shipped) | seconds |
| 2 online sweep | `repro/10_scaling_online.sh 0` | `results/rtx4090_local/` | ~30 min, GPU |
| 3 offline sweep | `repro/11_scaling_offline.sh 0` | `results/rtx4090_offline/` | ~1 h 30, GPU |
| 4 CPU sweep | `repro/12_cpu.sh` | `results/cpu55_local/` (20-yr runs) | ~10 min, CPU |
| 4b online sweep, Nz = 10 | `repro/13_scaling_online_nz10.sh 0` | `results/rtx4090_local_nz10/` | ~20 min, GPU |
| 5 accuracy | `repro/20_accuracy.sh 0` | `results/isoL_b1/`, `results/warm_local_offline/` | ~3 h, GPU |
| 6 figure and numbers | `repro/30_figures.sh` | `../figure/fig_scalability.{pdf,png,svg}`, numbers of the text | ~1 min, CPU |

`results/` (shipped) holds the per-run metrics; the raw runs (`outputs/`) are not shipped. Step 6 needs only
`results/` and `../figure/data/`.

## Metrics (`results/<tag>/scaling.*.csv`, `accuracy.csv`)

- ice-flow time per step = median over steps[1:] of the GPU-synchronised ice-flow time (`iceflow_mean_s`).
- run time (Fig. 7b, middle) = `n_iter` × `iceflow_mean_s`.
- memory (Fig. 7b, top) = TensorFlow allocator peak `tf_peak_mib` / 1024 (GiB).
- log–log slope = fit of per-step time vs cells over N ≥ 2 × 10⁶; throughput = cells / per-step time.
- CPU / GPU ratio = `step_mean_s` CPU / GPU at 400 m.
- L2 (Fig. 7b, bottom) = RMS over all sub-domain cells of the depth-averaged speed difference to the reference at
  t = 500 yr (`l2_velbar`); volume = `volume_rel_pct` (`tools/eval_accuracy.py`).

## What produces what

| paper element | from |
|---|---|
| Fig. 7a (200 m offline speed map) | `../figure/data/map_offline_200m_t500.nc` (t = 500 yr of the 200 m offline run) |
| Fig. 7b online / offline / online Nz = 10 (memory, run time) | `results/rtx4090_local/`, `results/rtx4090_offline/`, `results/rtx4090_local_nz10/` |
| Fig. 7b L2 error (online / offline) | `results/isoL_b1/`, `results/warm_local_offline/` |
| numbers of the text | `tools/paper_numbers.py` (run by `repro/30_figures.sh`), from `results/` |

Tags: `rtx4090_local` = online sweep, `rtx4090_local_nz10` = online sweep with Nz = 10, `rtx4090_offline` = offline
sweep, `cpu55_local` = CPU sweep, `isoL_b1` = online accuracy run, `warm_local_offline` = offline accuracy run,
`ref_local` = reference; `bench_rNN` = coarsening ratio NN (Δx = NN × 100 m).
