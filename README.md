# A Detection-Bias-Corrected Temporal Event-Stream Dataset for Dynamic Plant–Pollinator Link Prediction

[![CI](https://github.com/m0u/pp-tgnn-dataset/actions/workflows/ci.yml/badge.svg)](https://github.com/m0u/pp-tgnn-dataset/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Reference implementation accompanying the data-in-brief paper:

> M. Ghosh & A. Mukhopadhyay, *"A Detection-Bias-Corrected Temporal Event-Stream Dataset for Dynamic Plant–Pollinator Link Prediction."* Dept. of CSE, Kalyani University.

This repository reprocesses the public **H.J. Andrews LTER plant–pollinator visitation dataset** (2011–2018, 2021; 180 plots, 18 meadows, 147 plant species, 800 pollinator taxa) into a **continuous-time, detection-bias-corrected, weighted event stream** ready for **Temporal Graph Neural Network (TGNN)** training — and includes a working TGN baseline.

---

## What's here

| | |
|---|---|
| **`src/pp_tgnn/`** | Installable package implementing Algorithms 1–8 of the paper, plus a memory-based TGNN baseline model |
| **`scripts/`** | CLI entry points: run the pipeline, generate Figure 1, train/evaluate the TGNN |
| **`notebooks/`** | Two notebooks mirroring the CLI scripts, with inline explanation of every algorithm |
| **`tests/`** | `pytest` suite validating each pipeline stage against small synthetic data |
| **`data/raw/`** | *(empty — see below)* place the two raw LTER CSVs here |
| **`data/processed/`** | *(generated)* the five processed output files land here |
| **`docs/data_dictionary.md`** | Full column-level documentation of the processed output |

---

## Why this dataset exists

Raw pollinator-watch visit counts underestimate true interaction frequency because pollinators are **imperfectly detected** during a timed observation window, and raw survey files have structural quirks (mismatched date formats between files, a real 2019–2020 multi-year sampling gap) that silently break naive processing. This pipeline:

1. **Separates** genuine interactions from structural non-detections (rather than treating all "no interaction recorded" rows as missing data).
2. **Corrects for detection bias** via a Royle-type Poisson **N-mixture model**, fitted by vectorized numerical marginal-likelihood maximization (fast: all ~17K replicate groups fit in seconds).
3. Computes a **phenology-based confidence weight** from flowering intensity.
4. Assembles a **continuous-time weighted event stream** `(τ, plant_id, pollinator_id, c)` with a **leakage-safe chronological split** that correctly absorbs the multi-year survey gap.

The result: **20,025 weighted interaction events** and **13,917 empirically grounded structural negatives** (1,389 at deduplicated watch-session granularity — see `docs/data_dictionary.md`), directly usable with TGN, TGAT, JODIE, and related dynamic-link-prediction architectures — see `docs/data_dictionary.md` for the full schema.

> **Audit note:** as of this version, `config.py`'s N-mixture `lam_grid` has been
> widened from `geomspace(0.01, 80, 150)` to `geomspace(0.01, 5000, 300)` (and
> `n_trunc_extra` from 30 to 100). The original narrower grid silently capped
> the posterior-mean λ̂ for 382/17,472 groups (2.2%) at the grid ceiling,
> understating the true detection-corrected strength of the highest-abundance
> interactions. See `docs/data_dictionary.md#known-structural-pitfalls` for
> details and the numbers this changes.

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/m0u/pp-tgnn-dataset.git
cd pp-tgnn-dataset
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -e .          # installs the pp_tgnn package (see pyproject.toml)
# or: pip install -r requirements.txt
```

### 2. Get the raw data

The raw survey files are **not included** in this repository (large, and licensed CC BY 4.0 by the original data providers — see `data/raw/README.md`). Download them from the [EDI data package `knb-lter-and.5216`](https://andlter.forestry.oregonstate.edu/data/abstract.aspx?dbcode=SA026) and place them at:

```
data/raw/SA02601_v6.csv   # raw visitation events
data/raw/SA02602_v5.csv   # raw flowering-phenology records
```

### 3. Run the preprocessing pipeline

```bash
python scripts/run_pipeline.py \
    --vis data/raw/SA02601_v6.csv \
    --flw data/raw/SA02602_v5.csv \
    --outdir data/processed
```

This writes the five processed files described in `docs/data_dictionary.md` into `data/processed/`, and logs summary statistics matching Section 1.1 of the paper.

### 4. Reproduce Figure 1

```bash
python scripts/make_figure1.py \
    --processed_dir data/processed \
    --out figures/figure1.png
```

### 5. Train the TGNN baseline

```bash
python scripts/train_tgn.py \
    --processed_dir data/processed \
    --epochs 5
```

Reports AP / AUC / MRR on the chronological val/test splits and saves model weights to `data/processed/tgnn_lite_state_dict.pt`.

### Or: run the notebooks

```bash
jupyter lab notebooks/
```

- `01_preprocessing_pipeline.ipynb` — runs the pipeline step-by-step with inline explanation of each algorithm, sanity-checks against the paper's reported summary statistics, and reproduces Figure 1.
- `02_tgn_link_prediction.ipynb` — trains and evaluates the TGNN baseline, with a discussion of extending to full TGN/TGAT architectures.

---

## Pipeline algorithms → code map

| Paper | Function | Module |
|---|---|---|
| Algorithm 1 — `HarmonizeRawData` | `harmonize_raw_data()` | `pp_tgnn/pipeline.py` |
| Algorithm 2 — `PartitionInteractions` | `partition_interactions()` | `pp_tgnn/pipeline.py` |
| Algorithm 3 — `BuildReplicateGroups` | `build_replicate_groups()` | `pp_tgnn/pipeline.py` |
| Listing 1 + Algorithm 4 — `FitDetectionProbability` | `log_marginal_over_lamgrid()`, `fit_detection_probability()` | `pp_tgnn/pipeline.py` |
| Algorithm 5 — `FloweringConfidence` | `flowering_confidence()` | `pp_tgnn/pipeline.py` |
| Algorithm 6 — `JoinWithFallback` | `join_with_fallback()` | `pp_tgnn/pipeline.py` |
| Algorithm 7 — `BuildEventStream` | `build_event_stream()` | `pp_tgnn/pipeline.py` |
| Algorithm 8 — `ChronologicalSplit` | `chronological_split()` | `pp_tgnn/pipeline.py` |
| Figure 1 | `make_figure1()` | `pp_tgnn/figures.py` |
| TGNN baseline | `TGNNLite`, `TimeEncode` | `pp_tgnn/model.py` |
| Training / evaluation loop | `run_epoch()`, `mrr_eval()` | `pp_tgnn/train.py` |

All eight algorithms are validated end-to-end against small synthetic data in `tests/` (see below) and have been run against the real raw files with results matching the paper's reported summary statistics (p̂ ≈ 0.15, median P(N>0) ≈ 0.985, pair density ≈ 3.85%, median edge weight c ≈ 0.053 — see the audit note above for why this differs from the earlier-reported 0.096).

---

## Two documented pitfalls this pipeline guards against

1. **Silent date-format mismatch**: `SA02601` uses `DD-MM-YYYY`, `SA02602` uses `YYYY-MM-DD`. Parsing both with one format string corrupts one file's dates without raising an error. `harmonize_raw_data()` parses each file independently and **asserts** the parsed year matches the recorded `YEAR` column, converting a silent corruption into a loud failure.
2. **Real multi-year survey gap**: no surveys were conducted in 2019–2020. A naive fixed calendar-year train/val/test boundary silently produces an empty validation set. `chronological_split()` instead splits by **cumulative event volume** over the continuous `τ` axis, which is robust to arbitrary sampling gaps.

---

## Adapting this pipeline to other datasets

Every function in `pp_tgnn/pipeline.py` takes a `PipelineConfig` object (`pp_tgnn/config.py`) specifying column names, date formats, and model hyperparameters — so the pipeline can be re-run against future LTER releases, or adapted to structurally similar plant–pollinator, host–parasite, or other bipartite ecological interaction datasets, without editing the pipeline code itself.

---

## Running the tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

The test suite uses small synthetic data (generated on the fly by `tests/conftest.py`, mirroring the real raw schema) so it runs in seconds without needing the actual LTER files.

---

## Citation

If you use this dataset or pipeline, please cite the data paper (see `CITATION.cff`) and the underlying raw data package:

> H.J. Andrews Experimental Forest LTER, "Plant-pollinator interactions in Willamette National Forest meadows," Environmental Data Initiative, data package `knb-lter-and.5216`.

## License

Code: [MIT](LICENSE). Underlying raw/processed data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (see `data/raw/README.md`).
