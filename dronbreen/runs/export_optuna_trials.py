#!/usr/bin/env python3

"""Export the trials of the Step 2 Optuna study to a CSV file.

    python export_optuna_trials.py --storage sqlite:///optuna_cts.db

One row per trial: trial number, state, the three calibrated parameters (ela, T_ela,
basal_heat_flux_ref) and the objective cost_cts (mean CTS RMSE, m).

Requires optuna and pandas.
"""

import argparse
import os

import optuna
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

COLUMNS = {
    "processes.clim_carra_ela.ela": "ela",
    "processes.clim_carra_ela.T_ela": "T_ela",
    "processes.enthalpy.thermal.basal_heat_flux_ref": "basal_heat_flux_ref",
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--storage", default="sqlite:///optuna_cts.db")
    p.add_argument("--study", default="dronbreen_cts")
    p.add_argument("--out", default=os.path.join(HERE, "optuna_trials.csv"))

    return p.parse_args()


def main():
    args = parse_args()
    study = optuna.load_study(study_name=args.study, storage=args.storage)

    rows = []
    for trial in study.trials:
        row = {"trial": trial.number, "state": trial.state.name}
        for name, value in trial.params.items():
            row[COLUMNS[name]] = value
        row["cost_cts"] = trial.value
        rows.append(row)

    columns = ["trial", "state", "ela", "T_ela", "basal_heat_flux_ref", "cost_cts"]
    pd.DataFrame(rows)[columns].to_csv(args.out, index=False, float_format="%.6g")
    print(f"wrote {len(rows)} trials of '{args.study}' to {args.out}")


if __name__ == "__main__":
    main()
