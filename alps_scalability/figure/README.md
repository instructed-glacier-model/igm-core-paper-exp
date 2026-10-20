# Alps scalability figure (Fig. 7, `fig:scalability`)

```bash
python fig_scalability.py      # from any directory; writes fig_scalability.{pdf,png,svg}
```

Needs `numpy`, `matplotlib`, `netCDF4`, `cartopy`, `cmocean`, `shapely` (conda env `../runs/env/conda_igm-fig.yml`).
The font Inter is shipped in `fonts/` (SIL Open Font License). cartopy downloads the Natural Earth borders of
the inset on first use. `fig_scalability.pdf` is Fig. 7 of the paper.

## Data (`data/`)

Written by `../runs/tools/make_figure_data.py` (run by `../runs/repro/30_figures.sh`):

| file | content |
|---|---|
| `rtx4090_local/scaling.*.csv` | online sweep, one row per grid (`../runs/results/rtx4090_local/`) |
| `rtx4090_offline/scaling.*.csv` | offline sweep, one row per grid (`../runs/results/rtx4090_offline/`) |
| `rtx4090_local_nz10/scaling.*.csv` | online sweep with Nz = 10 (Lagrange), one row per grid (`../runs/results/rtx4090_local_nz10/`); cost and memory only |
| `isoL_b1/accuracy.csv`, `warm_local_offline/accuracy.csv` | accuracy runs on the sub-domain, online / offline (`../runs/results/`); L2 error row |
| `map_offline_200m_t500.nc` | t = 500 yr of the 200 m offline run (`../runs/outputs/rtx4090_offline/bench_r02/`): `x`, `y`, `time`, `thk`, `usurf`, `velsurf_mag`, values unchanged |

CSV columns used: `status`, `dx_m` (grid spacing, m), `n_iter` (time steps), `iceflow_mean_s` (ice-flow time
per step, s), `tf_peak_mib` (GPU memory peak, MiB). The script plots run time = `n_iter × iceflow_mean_s` and
memory = `tf_peak_mib / 1024` (GiB). Accuracy columns: `dx_m`, `time_yr` (500), `l2_velbar` (m/yr).
