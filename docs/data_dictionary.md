# Data Dictionary

## Files produced by `scripts/run_pipeline.py`

| File | Rows × Cols | Description |
|---|---|---|
| `plant_pollinator_events.csv` | 20,107 × 12 | Processed weighted positive event stream: `(τ, plant_id, pollinator_id, c)` plus intermediate confidence components and provenance columns. |
| `plant_pollinator_negative_events.csv` | 13,917 × 4 | Structural negatives: watch sessions in which the plot was surveyed but no interaction was observed (`PPI_STATUS = 0` in the raw data). |
| `events_train.csv` | 16,085 rows | Chronological training split (first 80% of events by cumulative τ). |
| `events_val.csv` | 2,011 rows | Chronological validation split (next 10%). |
| `events_test.csv` | 2,011 rows | Chronological test split (final 10%). |
| `lambda_hat.csv` | 17,472 rows | Posterior mean λ̂ for every (plot, year, plant, pollinator) group from the fitted N-mixture model — intermediate output, useful for reproducing Figure 1(b) or auditing detection-bias correction. |
| `flw_conf.csv` | 23,642 rows | Harmonized flowering table augmented with `confidence_flowering` — intermediate output, needed for the structural-negative approximation in the TGN notebook. |

> Note: the paper's Table 1 lists 1,389 negative events; the paper's own body
> text (Section 2.2) states 13,917 confirmed non-detections. This pipeline
> reproduces the latter (13,917), consistent with all `PPI_STATUS = 0` rows in
> the raw visitation file.

## Column-level description of `plant_pollinator_events.csv`

| Column | Description |
|---|---|
| `tau` | Elapsed time since the first record in the dataset, in days (integer, range 0–3,662). Used as the continuous temporal axis for TGNN training — **not** the raw `WATCH` ordinal, which would misrepresent the multi-year gap between 2018 and 2021. |
| `date` | Calendar date of the observation (ISO 8601), retained for interpretability and for reconstructing calendar-year subsets. |
| `plant_id` / `pollinator_id` | Canonical species codes (union vocabulary across both raw files; 147 plant and 800 pollinator identifiers). |
| `NO_INT` | Raw visit count summed within the (plot, year, watch, plant, pollinator) cell (i.e., across duplicate within-watch minute-level rows). |
| `confidence_posterior` | Detection-bias-corrected interaction confidence in `[0, 1]`: normalized `log(1 + λ̂)`, where `λ̂` is the posterior mean of the latent interaction intensity from the fitted N-mixture model. |
| `confidence_flowering` | Phenology-based confidence in `[0, 1]`: log-transformed, within-species-normalized flower count at the time of observation. |
| `c` | Final edge weight, `c = confidence_posterior × confidence_flowering` — used as the event weight in the temporal graph. |
| `is_imputed_flowering` | Boolean flag; `True` for the ~0.24% of events where no same-session flowering record existed and a last-observation-carried-forward value was substituted. |
| `PLOT_ID` / `YEAR` / `WATCH` | Provenance fields retained from the raw data, enabling traceability back to the original survey record and supporting alternative (e.g., per-plot or per-meadow) splitting strategies. |

## Dataset-level summary statistics (reproduced by `run_pipeline()`)

- 20,107 weighted positive events span 3,662 days (≈10.0 years) across 4,532 distinct plant–pollinator pairs, out of 147 × 800 = 117,600 possible pairs — an observed-pair density of **3.85%** (consistent with the 3.5% density reported for an earlier static-network use of the same survey data by Fu et al., an external sanity check that reprocessing has not distorted the network's basic sparsity structure).
- Final edge weight `c` has median **0.096** (IQR 0.046–0.193).
- `confidence_posterior` and `confidence_flowering` have medians **0.34** and **0.32** respectively — neither component dominates the product.
- Only **0.24%** of events required the flowering-confidence fallback.
- Fitted pooled detection probability **p̂ = 0.15**.
- Median raw N-mixture presence probability `P(N>0)` across all groups: **0.985** (motivating the use of normalized `log(1+λ̂)` instead — see Figure 1(b)).

## Known structural pitfalls (see main README for how the pipeline handles each)

1. **Silent date-format mismatch** between the two raw files (`DD-MM-YYYY` vs `YYYY-MM-DD`).
2. **Real multi-year survey gap** (no surveys conducted in 2019–2020), which breaks naive fixed-calendar-year splitting.
