# Aletsch figure (Fig. 6)

```bash
python plot_aletsch_pareto_figure.py      # ~5 s; needs numpy, matplotlib, netCDF4, optuna, cmocean
```

Writes `aletsch_pareto.{pdf,svg,png}` (`aletsch_pareto.pdf` is Fig. 6 of the paper).
The font Inter is shipped in `fonts/` (SIL Open Font License, `fonts/LICENSE.txt`).

`data/` holds copies of the run results and inputs, refreshed by `../runs/workflow/05_figure.sh`:

| file | content | panel |
|---|---|---|
| `optuna.db` | Optuna study `aletsch` (`../runs/analysis/optuna.db`) | (a) |
| `knee_run/output.nc` | knee trial #259 at the DEM dates and in 2020: `thk`, `usurf`, `uvelsurf`, `vvelsurf` | (b), (c) |
| `past_surf.nc` | observed DEMs `surf_1880` … `surf_2017` | (b) |
| `input.nc` | initial state and observed 2020 velocities | (c) |
