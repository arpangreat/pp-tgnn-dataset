# Data Dictionary

## Files produced by `scripts/run_pipeline.py`

| File | Rows × Cols | Description |
|---|---|---|
| `plant_pollinator_events.csv` | 20,025 × 12 | Processed weighted positive event stream: `(τ, plant_id, pollinator_id, c)` plus intermediate confidence components and provenance columns. |
| `plant_pollinator_negative_events.csv` | 13,917 × 4 | Structural negatives at raw minute-level row granularity (`PPI_STATUS = 0`). A deduplicated watch-session-level version (1,389 rows) is also available — see the note below. |
| `events_train.csv` | 16,020 rows | Chronological training split (first 80% of events by cumulative τ). |
| `events_val.csv` | 2,002 rows | Chronological validation split (next 10%). |
| `events_test.csv` | 2,003 rows | Chronological test split (final 10%). |
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

- 20,025 weighted positive events span 3,662 days (≈10.0 years) across 4,532 distinct plant–pollinator pairs, out of 147 × 800 = 117,600 possible pairs — an observed-pair density of **3.85%** (consistent with the 3.5% density reported for an earlier static-network use of the same survey data by Fu et al., an external sanity check that reprocessing has not distorted the network's basic sparsity structure). Note: an earlier manuscript draft reported 20,107 events; 20,025 is what the pipeline, run against the real raw files, actually produces, and is the number that should be reported going forward.
- Final edge weight `c` has median **0.053** (IQR 0.025–0.109) as of the `lam_grid` widening described below. An earlier version of this pipeline (grid capped at λ=80) reported a median of 0.096 — that number was an artifact of 382 high-abundance groups (2.2% of all groups) having their λ̂ silently capped at the grid ceiling, which compressed the *entire* global confidence-normalization scale, not just those 382 groups. See "Known structural pitfalls" below.
- `confidence_posterior` and `confidence_flowering` have medians **0.19** and **0.32** respectively (see note below on why `confidence_posterior`'s median moved from an earlier-reported 0.34).
- Only **0.24%** of events required the flowering-confidence fallback.
- Fitted pooled detection probability **p̂ = 0.15** (95% bootstrap CI, fit on all 17,342 informative groups: 0.15–0.20). Detection probability is not constant across meadow complexes: Cone comes in at p̂=0.10 vs. Bunchgrass/Carpenter/Frissell at p̂=0.20 — see `p_hat_by_meadow_complex.csv` from `scripts/run_pipeline_corrected.py`.
- Median raw N-mixture presence probability `P(N>0)` across all groups: **0.985** (motivating the use of normalized `log(1+λ̂)` instead — see Figure 1(b)).

## Known structural pitfalls (see main README for how the pipeline handles each)

1. **Silent date-format mismatch** between the two raw files (`DD-MM-YYYY` vs `YYYY-MM-DD`).
2. **Real multi-year survey gap** (no surveys conducted in 2019–2020), which breaks naive fixed-calendar-year splitting.
3. **N-mixture `lam_grid` ceiling too narrow for high-abundance groups.** The candidate range of λ values tested by the model (`config.PipelineConfig.lam_grid`) must extend beyond `max(observed single-watch count) / p̂`, or the posterior mean λ̂ for the busiest interactions gets silently pinned at the grid's upper limit rather than reflecting their true (much higher) strength. With the original default ceiling of 80, this affected 382/17,472 groups (2.2%) — all among the highest-abundance interactions, e.g. honeybee visits to *Gilia capitata* with single-watch counts up to 1,770. Because `confidence_posterior` is a *global* min–max normalization across all groups, this is not a local error: fixing the 382 capped groups stretches the normalization scale and lowers `confidence_posterior` for every other group too (median final edge weight `c` moves from 0.096 to 0.053; the share of events with `confidence_posterior` > 0.9 falls from ~9.5% to under 0.1%). Fixed as of this version by widening `lam_grid` to `geomspace(0.01, 5000, 300)` and `n_trunc_extra` to 100. When adapting this pipeline to other survey data, set these based on your own data's maximum observed count and fitted p̂, not the defaults.
