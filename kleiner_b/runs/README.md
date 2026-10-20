# Kleiner et al. (2015) enthalpy benchmark, experiment B: runs

Steady state of a polythermal parallel-sided slab (thickness 200 m, slope 4°,
surface temperature −3 °C, accumulation 0.2 m yr⁻¹) with a prescribed velocity
field and strain heating, for which Kleiner et al. (2015, *The Cryosphere*, 9,
217–228, doi:10.5194/tc-9-217-2015, Appendix A2) give an analytical solution.

The benchmark prescribes the velocity field, so IGM is not run through
`igm_run`: `run_kleiner_b.py` loads the default IGM configuration and steps the
`enthalpy` and `subglacial_hydrology` (till storage) modules directly on a
2 × 2 grid, with dt = 1 yr and 500 equally spaced enthalpy levels, until the
enthalpy no longer changes.

## Contents

```
run_kleiner_b.py          runs IGM to steady state and writes the CSV files
analytical_solutions.py   analytical solution (Kleiner et al., 2015, App. A2)
```

## Running

From this folder, in an environment with IGM installed:

```bash
python run_kleiner_b.py
```

It prints the cold–temperate transition surface (CTS) height and the RMSD to
the analytical solution, and writes to `outputs/`:

- `kleiner_exp_b_results.csv`: IGM enthalpy, temperature and water content;
- `kleiner_exp_b_analytical.csv`: the analytical solution on the same grid.

To rebuild the figure from new results:

```bash
cp outputs/kleiner_exp_b_*.csv ../figure/ && cd ../figure && python make_figure.py
```

## IGM version

IGM 3.2.1 (`pip install igm-model==3.2.1`).
