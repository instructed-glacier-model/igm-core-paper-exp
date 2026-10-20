"""Merge the per-trial outputs of the sweep into one CSV file.

    in : analysis/optuna.db (study `aletsch`), outputs/sweep/wK/<trial>/ (_igm_score.json, version.txt)
    out: analysis/sweep_trials.csv

One row per trial of the study: trial number, state, worker, the six calibrated parameters, the two
objectives and the three scores logged but not optimised (see user/code/processes/track_*), and the
IGM version of the run. The two objectives of the database are checked against the score files.
Trials without an output folder or a score file (interrupted at the end of the sweep) keep empty
worker or score columns.

Usage (from anywhere): python workflow/export_sweep_trials.py [--sweep outputs/sweep]
For a sweep rerun by step 1: --storage sqlite:///analysis/optuna_rerun.db --study aletsch_rerun --sweep outputs/sweep_rerun
"""
import argparse
import json
from pathlib import Path

import optuna
import pandas as pd

RUNS = Path(__file__).resolve().parent.parent

PARAMS = {
    "processes.smb_accmelt.weight_accumulation": "w_acc",
    "processes.smb_accmelt.weight_ablation": "w_abl",
    "processes.smb_accmelt.weight_Aletschfirn": "w_aletschfirn",
    "processes.smb_accmelt.weight_Jungfraufirn": "w_jungfraufirn",
    "processes.smb_accmelt.weight_Ewigschneefeld": "w_ewigschneefeld",
    "processes.iceflow.physics.sliding.tau_ref": "tau_ref",
}
OBJECTIVES = ["cost_usurf_rmse", "cost_velsurf_union"]
LOGGED = ["cost_usurf_bias", "cost_usurf", "cost_velsurf"]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--storage", default=f"sqlite:///{RUNS / 'analysis/optuna.db'}")
    p.add_argument("--study", default="aletsch")
    p.add_argument("--sweep", type=Path, default=RUNS / "outputs/sweep")
    p.add_argument("--out", type=Path, default=RUNS / "analysis/sweep_trials.csv")
    return p.parse_args()


def main():
    args = parse_args()
    study = optuna.load_study(study_name=args.study, storage=args.storage)
    folders = {int(d.name): d for d in args.sweep.glob("w*/*") if d.is_dir()}

    rows = []
    for trial in study.trials:
        row = {"trial": trial.number, "state": trial.state.name, "worker": None}
        for name, value in trial.params.items():
            row[PARAMS[name]] = value
        for name, value in zip(OBJECTIVES, trial.values or [None] * len(OBJECTIVES)):
            row[name] = value

        folder = folders.get(trial.number)
        if folder is not None:
            row["worker"] = int(folder.parent.name[1:])
            score = folder / "_igm_score.json"
            if score.exists():
                scores = json.loads(score.read_text())
                for name in OBJECTIVES:
                    if row[name] is not None and abs(scores[name] - row[name]) > 1e-9:
                        raise SystemExit(f"trial {trial.number}: {name} differs from {score}")
                    row[name] = scores[name]
                for name in LOGGED:
                    row[name] = scores[name]
            version = folder / "version.txt"
            if version.exists():
                info = dict(line.split(": ", 1) for line in version.read_text().splitlines() if ": " in line)
                row["igm_version"] = info.get("version")
        rows.append(row)

    columns = ["trial", "state", "worker", *PARAMS.values(), *OBJECTIVES, *LOGGED, "igm_version"]
    table = pd.DataFrame(rows).reindex(columns=columns).sort_values("trial")
    table["worker"] = table["worker"].astype("Int64")
    table.to_csv(args.out, index=False, float_format="%.6g")
    print(f"wrote {len(table)} trials of '{args.study}' "
          f"({(table.state == 'COMPLETE').sum()} complete) to {args.out}")


if __name__ == "__main__":
    main()
