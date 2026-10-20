# ISMIP-HOM experiment B: runs

Diagnostic ice-flow benchmark B of ISMIP-HOM (Pattyn et al., 2008): a
flowline of length L = 20, 40, 80, 160 km, frozen to its sinusoidal bed and
periodic in x. IGM solves the Blatter–Pattyn problem directly on the velocity
degrees of freedom (`unified` method, `identity` mapping, L-BFGS, MOLHO
vertical basis with Nz = 2, 100 grid nodes along x), and its surface speed is
compared with the full-Stokes Elmer/Ice solution (`oga`).

## Contents

```
experiment/params.yaml              IGM configuration (Hydra experiment)
user/code/inputs/ismip_hom_b.py     input module: ISMIP-HOM B geometry
user/conf/inputs/ismip_hom_b.yaml   its parameters (L, number of nodes n)
data/oga/oga1b{020,040,080,160}.txt Elmer/Ice full-Stokes reference
run.sh                              runs the four lengths, then make_csv.py
make_csv.py                         results.nc + oga -> CSV files of the figure
```

## Running

From this folder, in an environment with IGM installed:

```bash
./run.sh
```

It runs one `igm_run` per length (`outputs/<L>km/`), then writes
`outputs/exp_b_unified_<L>km_identity_lbfgs_{igm,ref}.csv` and prints the
relative L2 error of IGM against the reference. Each run takes well under a
minute on an RTX 4090. To rebuild the figure from new results:

```bash
cp outputs/*.csv ../figure/ && cd ../figure && python make_figure.py
```

## IGM version

IGM 3.2.1 (`pip install igm-model==3.2.1`).

## Reference data

`data/oga/` holds the Elmer/Ice full-Stokes solution (`oga`, Gagliardini and
Zwinger, 2008) from the supplement of Pattyn et al. (2008), *The Cryosphere*,
2, 95–108, doi:10.5194/tc-2-95-2008 (archive `tc-2-95-2008-supplement.zip`):
columns are the normalized position x/L and the surface speed (m/yr).
