# igm-core-paper-exp

Material to reproduce the experiments and figures of the IGM v3.2.1 model description paper
submitted to *Geoscientific Model Development* (GMD).

IGM: <https://github.com/instructed-glacier-model/igm> — documentation: <https://igm-model.org/>

## Contents

One folder per experiment, each with `runs/` (to re-run it) and `figure/` (to redraw the figure).

| Experiment | Paper | Instructions |
|---|---|---|
| Calibration of Aletsch Glacier, 1880–2020 | Sect. 6.1, Fig. 6 | [`aletsch/runs/README.md`](aletsch/runs/README.md) |
| Scalability test, glaciation onset of the Alps | Sect. 6.2, Fig. 7 | [`alps_scalability/runs/README.md`](alps_scalability/runs/README.md) |
| Thermal calibration of Drønbreen, Svalbard | Sect. 6.3, Fig. 8 | [`dronbreen/runs/README.md`](dronbreen/runs/README.md) |
| ISMIP-HOM experiment B (Pattyn et al., 2008) | App. D, Fig. D1 | [`ismip_hom_b/runs/README.md`](ismip_hom_b/runs/README.md) |
| Enthalpy benchmark A (Kleiner et al., 2015) | App. D, Fig. D2 | [`kleiner_a/runs/README.md`](kleiner_a/runs/README.md) |
| Enthalpy benchmark B (Kleiner et al., 2015) | App. D, Fig. D3 | [`kleiner_b/runs/README.md`](kleiner_b/runs/README.md) |

## IGM version

```bash
pip install igm-model==3.2.1
```

All paper results were obtained on a single NVIDIA RTX 4090 (24 GB).

## License

GNU General Public License v3 (see [`LICENSE`](LICENSE)), as IGM. Third-party data keep the terms of their
sources, cited in the README of each experiment; the Inter font in the `fonts/` folders is under the
SIL Open Font License.
