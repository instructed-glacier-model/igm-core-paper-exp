"""Step 4 — the numbers quoted in the Aletsch paragraph of the paper.

    in : analysis/optuna.db (study `aletsch`), outputs/knee_run, outputs/sliding_{224,259,279},
         data/input.nc, data/past_surf.nc
    out: printed table

Usage (from anywhere): python workflow/04_paper_numbers.py
"""
import json
from pathlib import Path

import numpy as np
import optuna
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parent.parent
P = "processes."
K_TAU, K_ACC, K_ABL = (P + "iceflow.physics.sliding.tau_ref", P + "smb_accmelt.weight_accumulation",
                       P + "smb_accmelt.weight_ablation")
K_BASINS = [P + f"smb_accmelt.weight_{b}" for b in ("Aletschfirn", "Jungfraufirn", "Ewigschneefeld")]
optuna.logging.set_verbosity(optuna.logging.WARNING)

# sweep, Pareto front, witness trials
study = optuna.load_study(study_name="aletsch", storage=f"sqlite:///{ROOT / 'analysis/optuna.db'}")
trials = [t for t in study.trials if t.state.name == "COMPLETE"]
pts = np.array([t.values for t in trials])
is_par = np.array([not np.any(np.all(pts <= p, axis=1) & np.any(pts < p, axis=1)) for p in pts])
front = [t for t, p in zip(trials, is_par) if p]
v = np.array([t.values for t in front])
norm = (v - v.min(0)) / np.maximum(1e-9, v.max(0) - v.min(0))
win = {"knee": front[int(np.argmin(norm.sum(1)))],             # same rule as the figure script
       "low": min(front, key=lambda t: t.params[K_TAU]),
       "high": max(front, key=lambda t: t.params[K_TAU])}
minutes = (max(t.datetime_complete for t in trials) - min(t.datetime_start for t in trials)).total_seconds() / 60
rng = lambda key: (min(t.params[key] for t in front), max(t.params[key] for t in front))
eff = lambda w: (2 + w) / 3   # smb_accmelt: accumulation in basin b is multiplied by (2 + w_b)/3

print(f"study 'aletsch': {len(trials)} complete trials, {len(front)} Pareto-optimal")
for name, t in win.items():
    print(f"  {name:4s} #{t.number}: DEM {t.values[0]:.3f} m, speed {t.values[1]:.3f} m/yr | "
          + ", ".join(f"{k.split('.')[-1]} {t.params[k]:.4f}" for k in [K_TAU, K_ACC, K_ABL] + K_BASINS))

# glacier-wide sliding fraction in 2020, on the ice of the knee run (thk > 10 m, u_s > 5 m/yr)
frac = {}
for name, t in win.items():
    with Dataset(ROOT / f"outputs/sliding_{t.number}/output.nc") as nc:
        i = int(np.argmin(abs(nc["time"][:] - 2020)))
        frac[name] = {k: np.asarray(nc[k][i]) for k in ("thk", "velsurf_mag", "velbase_mag")}
K = frac["knee"]
ice = (K["thk"] > 10) & (K["velsurf_mag"] > 5)
sf = {n: f["velbase_mag"][ice].sum() / f["velsurf_mag"][ice].sum() for n, f in frac.items()}

knee = win["knee"]
score = json.load(open(ROOT / "outputs/knee_run/_igm_score.json"))
ny, nx = Dataset(ROOT / "data/input.nc")["thk"].shape
years = [int(k[5:]) for k in Dataset(ROOT / "data/past_surf.nc").variables if k.startswith("surf_")]
tau, acc, abl = rng(K_TAU), rng(K_ACC), rng(K_ABL)
bas = [eff(w) for k in K_BASINS for w in rng(k)]
dem, spd = (v[:, 0].min(), v[:, 0].max()), (v[:, 1].min(), v[:, 1].max())
rows = [
    ("number of trials", f"{len(trials)}"),
    ("grid (ny x nx)", f"{ny} x {nx}"),
    ("DEM years (1880 = initial state)", ", ".join(map(str, years))),
    ("wall clock of the sweep", f"{minutes:.0f} min"),
    ("trials per minute", f"{len(trials) / minutes:.1f}"),
    ("τ_ref on the front (MPa)", f"{tau[0]:.3f}-{tau[1]:.3f}"),
    ("w_acc on the front", f"{acc[0]:.3f}-{acc[1]:.3f}"),
    ("w_abl on the front", f"{abl[0]:.3f}-{abl[1]:.3f}"),
    ("basin corrections (2+w)/3 on the front", f"{min(bas):.3f}-{max(bas):.3f}"),
    ("basin correction bounds (2+w)/3, w in [0.4, 1.6]", f"{eff(0.4):.1f}-{eff(1.6):.1f}"),
    ("sliding fraction u_b/u_s 2020, high-τ to low-τ", f"{sf['high']:.2f}-{sf['low']:.2f} (knee {sf['knee']:.2f})"),
    ("DEM misfit range on the front (m)", f"{dem[0]:.2f}-{dem[1]:.2f}"),
    ("speed misfit range on the front (m/yr)", f"{spd[0]:.2f}-{spd[1]:.2f}"),
    ("knee τ_ref (MPa)", f"{knee.params[K_TAU]:.3f}"),
    ("knee w_acc, w_abl", f"{knee.params[K_ACC]:.3f}, {knee.params[K_ABL]:.3f}"),
    ("knee DEM RMSE (m)", f"{knee.values[0]:.2f}"),
    ("knee mean absolute DEM bias (m)", f"{score['cost_usurf_bias']:.2f}"),
    ("knee speed RMSE (m/yr)", f"{knee.values[1]:.2f}"),
]
print()
for q, val in rows:
    print(f"  {q:50s} {val}")
