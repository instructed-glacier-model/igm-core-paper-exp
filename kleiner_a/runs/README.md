# Kleiner et al. (2015) enthalpy benchmark, experiment A: runs

Transient thermal evolution of a 1000 m thick, motionless ice slab with a
geothermal heat flux of 0.042 W m⁻² and a basal till water storage, forced by
a surface temperature of −30 °C (0–100 kyr), −5 °C (100–150 kyr) and −30 °C
(150–300 kyr) (Kleiner et al., 2015, *The Cryosphere*, 9, 217–228,
doi:10.5194/tc-9-217-2015).

The benchmark prescribes the ice column and no flow, so IGM is not run through
`igm_run`: `run_kleiner_a.py` loads the default IGM configuration and steps the
`enthalpy` and `subglacial_hydrology` (till storage) modules directly on a
2 × 2 grid, with dt = 200 yr and 50 equally spaced enthalpy levels.

## Running

From this folder, in an environment with IGM installed:

```bash
python run_kleiner_a.py
```

It writes `outputs/kleiner_exp_a_results.csv` (time, basal temperature, basal
melt rate, till water thickness) in about two minutes. To rebuild the figure
from new results:

```bash
cp outputs/kleiner_exp_a_results.csv ../figure/ && cd ../figure && python make_figure.py
```

## IGM version

IGM 3.2.1 (`pip install igm-model==3.2.1`).

## Reference data

The reference curves of the figure (`../figure/kleiner_exp_a_reference_{tb,aw,hw}.csv`:
basal temperature, basal melt rate, basal water thickness) were digitized from
Kleiner et al. (2015).
