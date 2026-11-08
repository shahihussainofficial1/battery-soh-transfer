# Does Partial-Charge Battery SOH Estimation Transfer?

Benchmark of battery state-of-health (SOH) estimation from a partial charging window (3.92–4.19 V) on three public lithium-ion aging datasets: NASA PCoE, Oxford Battery Degradation Dataset 1 and CALCE CS2.

The preprint is available in [`paper/main.pdf`](paper/main.pdf) and on Zenodo: [doi:10.5281/zenodo.24053885](https://doi.org/10.5281/zenodo.24053885).

## Key findings

Mean RMSE of the best configuration over the three datasets, in percentage points (pp) of SOH:

| Evaluation protocol | Best model (ref. features) | Cycle count only | Capacity ratio (no training) |
|---|---|---|---|
| Random 80/20 split of cycles | **0.8** | 2.6 | 8.3 |
| Leave-one-cell-out | **2.0** | 3.1 | 8.3 |
| Leave-one-dataset-out | 7.9 | 10.3 | 8.3 |

- Errors grow by an order of magnitude from random splits to unseen datasets, where most configurations have negative R².
- Within the Oxford and CALCE datasets, counting cycles alone is a strong baseline; on held-out CALCE cells it beats every charge-curve model (2.3 vs 2.45 pp).
- Normalizing features to each cell's own first charge lowers the cross-dataset error of tree ensembles on NASA and CALCE (e.g. 13.5 → 5.9 pp on CALCE) but not on Oxford.
- Transfer fails mainly because the relation between window charge and capacity differs between datasets (fitted slopes 0.66, 0.77 and 1.56), and the main aging peak of the Oxford cells lies below the common window.

## Data

| Dataset | Cells used | Source |
|---|---|---|
| NASA PCoE | B0005, B0006, B0007, B0018 | NASA Prognostics Data Repository |
| Oxford Battery Degradation Dataset 1 | Cells 1–8 | University of Oxford, doi:10.5287/bodleian:KO2kdmYGg (ODbL) |
| CALCE | CS2_33 to CS2_38 | CALCE Battery Data Archive, University of Maryland |

The data are not redistributed here; `scripts/download_data.py` downloads them from the providers (about 1 GB). Please cite the original datasets as listed in the paper.

## Repository structure

```
scripts/download_data.py   download and extract the three datasets
src/data.py                per-cycle capacity, SOH and CC charging curves Q(V) on a 5 mV grid
src/features.py            window features, reference normalization, capacity ratio
src/run_experiments.py     random split, leave-one-cell-out, leave-one-dataset-out, window study, footprint
src/run_anchoring.py       reference anchoring experiment
src/make_figures.py        figures and LaTeX tables for the paper
results/                   numeric results used in the paper
paper/                     preprint in IEEE conference format (main.pdf, main.tex, references.bib)
```

## Reproducing the results

Requires Python 3.10+ and runs on a CPU (about 30 minutes on a laptop, most of it reading the CALCE spreadsheets and the window study).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py
cd src
python data.py
python run_experiments.py
python run_anchoring.py
python make_figures.py
```

## Citation

```bibtex
@misc{shahi2026battery,
  title  = {Does Partial-Charge Battery State-of-Health Estimation Transfer Across Cells and Datasets? A Benchmark on NASA, Oxford and CALCE Data},
  author = {Haider Md. Faiyaz Hussain Shahi},
  year   = {2026},
  note   = {Preprint},
  doi    = {10.5281/zenodo.24053885},
  url    = {https://github.com/shahihussainofficial1/battery-soh-transfer}
}
```

## License

Code: MIT License (see `LICENSE`). The datasets keep their original licenses.
