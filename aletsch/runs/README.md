# Aletsch Glacier — multi-objective calibration (paper Sect. 6.1, Fig. 6)

Great Aletsch Glacier from 1880 to 2020 at 100 m (225 × 132 cells), with IGM's offline pretrained
ice-flow emulator (`dahunet_mini`, frozen) and a temperature-index SMB driven by daily climate.
An Optuna NSGA-II sweep of ~600 runs calibrates six parameters:
- accumulation and ablation factors w_acc, w_abl ∈ [0.5, 3];
- three firn-basin weights w_b ∈ [0.4, 1.6] (Aletschfirn, Jungfraufirn, Ewigschneefeld), acting as (2 + w_b)/3;
- the Weertman basal shear stress τ_ref ∈ [0.08, 0.5] MPa at u_ref = 100 m/yr.

Two misfits are minimised: the DEM RMSE averaged over 1926, 1957, 1980, 1999, 2009 and 2017 on observed ∪
modelled ice (1880 = initial state), and the 2020 surface-speed RMSE over all valid observed pixels.

## Data (`data/`)

Bed, initial state, observed DEMs 1880–2017 (`past_surf.nc`), daily temperature and precipitation
(`temp_prec.dat`), mass-balance maps and parameters (`massbalance.nc`, `mbparameter.dat`) and firn-basin
masks (`bassin.nc`) are from Jouvet et al. (2011), J. Glaciol. 57(206), 1033–1045,
doi:[10.3189/002214311798843359](https://doi.org/10.3189/002214311798843359). The observed surface
velocities in `input.nc` are from Millan et al. (2019), Remote Sensing 11, 2498.

## Environment

IGM 3.2.1 (`pip install -r workflow/requirements.txt`), one NVIDIA RTX 4090.

## Reproduce

Run from this folder (`runs/`).

| step | command | inputs → outputs | cost |
|---|---|---|---|
| 1 sweep | `workflow/01_sweep.sh` | `experiment/params.yaml`, `optuna/optuna_params.yaml` → `analysis/optuna_rerun.db` | ~100 min, 1 GPU |
| 2 knee run | `workflow/02_knee_run.sh` | `experiment/params.yaml` (= knee trial #259) → `outputs/knee_run/` | ~35 s |
| 3 sliding runs | `workflow/03_sliding_runs.sh` | trials #224, #259, #279 → `outputs/sliding_*/` | 3 × ~35 s |
| 4 numbers of the text | `python workflow/04_paper_numbers.py` | DB + runs of steps 2–3 → printed table | seconds |
| 5 figure | `workflow/05_figure.sh` | DB + knee run + observations → `../figure/aletsch_pareto.pdf` | ~5 s |
| trials table | `python workflow/export_sweep_trials.py` | DB + per-trial outputs → `analysis/sweep_trials.csv` (shipped; made from the paper sweep, whose outputs are not shipped) | seconds |

- The paper uses the shipped sweep `analysis/optuna.db` (study `aletsch`, 597 complete trials, 99 min);
  `analysis/sweep_trials.csv` lists its trials (parameters and scores). Step 1 writes a separate database;
  without a fixed seed a rerun gives an equivalent, not identical, front.
- Steps 4–5 need the runs of steps 2–3 (`outputs/` is not shipped).
- Keep `time.save = 1` in `params.yaml`: IGM cuts the time step at save times, so the scores depend on it.

## Layout

```
experiment/        configuration (params.yaml = knee trial #259) and ice-flow emulator settings
optuna/            sweep definition
user/              user modules: clim_aletsch, smb_accmelt, track_dem_rmse, track_velsurf_union
data/              inputs
analysis/          sweep of the paper: optuna.db, sweep_trials.csv
workflow/          steps 1-5, requirements.txt
```
