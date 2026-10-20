# Drønbreen — calibration of the thermal structure against radar (IGM GMD paper, Fig. 8)

Sect. 6.3 and Fig. 8 of the paper. Drønbreen
(Nordenskiöld Land, Svalbard) is a polythermal glacier. The model runs from 1900 to 2022 on a fixed
geometry (81 × 91 cells, 100 m), coupling ice flow (MOLHO, `experiment/common/iceflow_molho.yaml`),
enthalpy, Arrhenius factor and till-storage hydrology, with the user climate module `clim_carra_ela`.
Above an altitude `ela`, the air temperature is set to `T_ela`, a proxy for refreezing in the firn. As the
enthalpy module caps the surface temperature at 0 °C, any `T_ela` ≥ 0 gives the same run.

- **Calibration** (`experiment/params_step2_optuna.yaml`, `optuna/optuna_cts.yaml`): Optuna TPE, 64 trials,
  over `ela` ∈ [650, 720] m, `T_ela` ∈ [−0.5, 2] °C and the geothermal flux ∈ [0.03, 0.08] W m⁻². The
  objective `cost_cts` is the mean over the ten radar lines of the CTS elevation RMSE (user module
  `track_cts_err`, using `tools/cts.py`).
- **Best run** (`experiment/params_step1_spinup.yaml`): one run with the calibrated parameters.

## Data

| File | Contents |
|---|---|
| `data/input_dronbreen.nc` | 100 m grid, EPSG:32633: surface DEM (Norwegian Polar Institute, 2024), bed kriged from the GPR, ice mask, CARRA 1990–2024 mean annual and mean summer air temperature (Køltzow et al., 2022) |
| `data/cts_gpr/*.csv` | Ten GPR lines of 28–29 March 2022 (bed and CTS elevation), from Mannerfelt et al. (2025), doi:[10.5281/zenodo.17882300](https://doi.org/10.5281/zenodo.17882300), reduced to the columns used |

## Environment

IGM 3.2.1, plus
`optuna`, `pandas`, `scipy`, `netCDF4`, `xarray`, `omegaconf` and `matplotlib` ≥ 3.8. One NVIDIA RTX 4090.

## Reproduce

Run from this folder (`runs/`).

| step | command | inputs → outputs | cost |
|---|---|---|---|
| 1 sweep | `igm_run -m +experiment=params_step2_optuna hydra/sweeper=igm_optuna hydra.sweeper.optuna_config=optuna/optuna_cts.yaml` | → `optuna_cts.db`, `multirun/optuna_cts/` | ~45 min (3 trials at a time) |
| 2 trials table | `python export_optuna_trials.py` | `optuna_cts.db` → `optuna_trials.csv` (shipped) | seconds |
| 3 best run | `igm_run +experiment=params_step1_spinup hydra.run.dir=outputs/step1` | → `outputs/step1/` | ~2 min |
| 4 figure data | `python make_figure_data.py --run outputs/step1` | step 3 + `data/cts_gpr/` → `outputs/figure_data/` (`section.nc`, `transects.csv`, `map.nc`, `tracks.csv`) | seconds |
| 5 figure | `cp outputs/figure_data/* ../figure/ && cd ../figure && python make_figure.py` | → `cts_figure_minimal.{pdf,png}` | seconds |

The figure data of the paper ship in `../figure/`, so step 5 runs alone. Optuna's TPE sampler is not
seeded: a new sweep gives an equivalent, not identical, set of trials.

## Numbers of the text

- **Calibrated parameters and misfit 18.8 m:** trial #61 of `optuna_trials.csv` (`ela` 666 m, `T_ela`
  0.42 °C, flux 0.044 W m⁻², 18.81 m), the values of `params_step1_spinup.yaml`. Trial #62 ties at 18.81 m.
  Steps 4 and 5 print the misfit of the best run.
- **89 % of the observed temperate cross-section:** printed by `make_figure.py` (step 5).
- **30 trials within 1 m of the optimum, spanning 20 % of the `ela` range and over 70 % of the flux range,
  all with `T_ela` > 0 (the same model input, see above), whereas the trials with `T_ela` ≤ 0 fit much
  worse (25–35 m):** from `optuna_trials.csv`.
