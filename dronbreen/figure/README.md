# Drønbreen CTS figure (Fig. 8)

```bash
python make_figure.py      # writes cts_figure_minimal.{pdf,png} (Fig. 8 of the paper)
```

Options: `--data` (folder of the data files), `--out` (output folder), `--name` (output file stem).
Needs numpy, pandas, scipy, xarray and matplotlib ≥ 3.8. Font: Inter, shipped in `fonts/`
(SIL Open Font License).

The figure has three panels: (a) map of the glacier with the ten radar tracks and the featured transect (T3);
(b) the modelled thermal section along T3 with the observed CTS; (c) observed and modelled temperate ice per
transect, with the CTS RMSE.

## Data files

Written by `../runs/make_figure_data.py` from a Step 1 run (see `../runs/README.md`):

- `map.nc` — ice thickness `thk` and bed `topg` on the 100 m model grid (UTM 33N). The run keeps its geometry
  fixed, so this is the input geometry: the 2024 NPI surface DEM minus a bed kriged from the radar.
- `tracks.csv` — the ten radar tracks, one row per sample (`transect`, `easting`, `northing`).
- `section.nc` — the modelled thermal section along T3 on the enthalpy solver's 30-layer grid: `ice_type`
  (−1 no ice, 0 cold, 1 temperate), `elevation`, `surface`, `bed`, `cts_observed`, coordinate `distance_km`.
- `transects.csv` — one row per transect: observed and modelled temperate cross-section (m²), total ice
  cross-section (m²), CTS RMSE (m). Ranked by observed temperate area, so T1 is the most temperate line.

and, for the locator inset of panel (a):

- `svalbard.csv` — Svalbard coastline (`ring`, `lon`, `lat`; one polygon per `ring`), from Natural Earth 1:10m
  land (public domain, <https://www.naturalearthdata.com>), clipped to 9–35° E, 76–81.2° N.
