# Dataset Discrepancy & Integrity Audit Report

## 1. Executive Summary

- **Total Checks Evaluated:** 18
- **Errors (Breaking / Data-Corruption Risks):** 3
- **Warnings (Semantic Nuances & Fallback Triggers):** 7
- **Informational (Normal Data Reductions & Milestones):** 8

---

## 2. Findings by Severity

| Severity | Check Name | Category | Issues Count | Issues % | Impact |
|---|---|---|---|---|---|
| **ERROR** | `SA02601_visitation_date_year_consistency` | Date Formatting & Consistency | 74,942 | 100.00% | Pipeline execution with default config '%d-%m-%Y' will immediately crash. Must set PipelineConfig.vis_date_fmt = '%Y-%m-%d'. |
| **ERROR** | `lambda_grid_capping_risk` | Distributional Boundary | 4,399 | 25.18% | Silently pins posterior mean lambda_hat at grid ceiling, artificially squashing strongest edges. |
| **ERROR** | `plot_to_meadow_complex_inconsistency` | Spatial Hierarchy | 5 | 2.78% | Corrupts spatial partitioning and meadow-stratified detection probability analysis. |
| **WARNING** | `calendar_year_gaps` | Temporal Continuity | 2 | 18.18% | Naive calendar-based train/val/test splits produce empty partitions or invalid temporal decay. |
| **WARNING** | `degenerate_flowering_species` | Distributional Boundary | 38 | 18.81% | Min-max normalization yields 0/0; triggers neutral 0.5 confidence fallback in Algorithm 5. |
| **WARNING** | `flw_negative_or_null_flower_count` | Semantic Integrity | 2,154 | 9.11% | Negative counts are physical impossibilities; nulls require fillna(0) for log1p. |
| **WARNING** | `interactions_lacking_same_session_flowering` | Referential Integrity | 107 | 0.18% | Requires last-observation-carried-forward (LOCF) imputation (Algorithm 6). |
| **WARNING** | `structural_negative_granularity_discrepancy` | Granularity & Aggregation | 12,528 | 90.02% | Users expecting 1,389 negative rows will encounter schema confusion when seeing 13,917 rows. |
| **WARNING** | `vis_pos_missing_or_zero_visit_count` | Semantic Integrity | 69 | 0.11% | Algorithm 2 substitutes missing counts with 1. Causes slight variation in raw totals. |
| **WARNING** | `visitation_watches_missing_flowering_survey` | Referential Integrity | 4 | 0.09% | Indicates asynchronous sampling sessions or unsynchronized data entry. |
| INFO | `SA02601_visitation_missing_columns` | Schema Integrity | 0 | 0.00% | Downstream pipeline failure if required survey attributes are absent. |
| INFO | `SA02602_flowering_date_year_consistency` | Date Formatting & Consistency | 0 | 0.00% | Silent date corruption; links events to incorrect field seasons or chronologies. |
| INFO | `SA02602_flowering_missing_columns` | Schema Integrity | 0 | 0.00% | Downstream pipeline failure if required survey attributes are absent. |
| INFO | `unmatched_visited_plant_species` | Referential Integrity | 0 | 0.00% | Interactions on these plants cannot compute phenological confidence and rely purely on default fallback (0.5). |
| INFO | `vis_neg_interaction_has_species` | Semantic Integrity | 0 | 0.00% | Contradiction: record flagged as empty watch but lists a species. |
| INFO | `vis_pos_interaction_missing_species` | Semantic Integrity | 0 | 0.00% | Interactions cannot be mapped to bipartite graph nodes without species identifiers. |
| INFO | `vis_ppi_status_valid_domain` | Semantic Integrity | 0 | 0.00% | Unknown record type; invalid status breaks E+ / E0 partitioning. |
| INFO | `within_watch_minute_level_duplicates` | Granularity & Aggregation | 7,591 | 37.92% | Explains the reduction from 61,025 raw positive rows to ~20,025 watch-level positive events. |

---

## 3. Detailed Descriptions & Remediation Guidance

### `SA02601_visitation_date_year_consistency` (ERROR)
- **Category:** Date Formatting & Consistency
- **Total Rows / Entities Evaluated:** 74,942
- **Discrepancies Identified:** 74,942 (100.00%)
- **Description:** Format mismatch in SA02601_visitation: Expected format '%d-%m-%Y' failed on 74942 rows. Auto-detection identified actual raw format as '%Y-%m-%d' (100% match with 'YEAR').
- **Downstream Impact:** Pipeline execution with default config '%d-%m-%Y' will immediately crash. Must set PipelineConfig.vis_date_fmt = '%Y-%m-%d'.
- **Recommended Remediation:** Update PipelineConfig(vis_date_fmt='%Y-%m-%d') when running pipeline on this dataset version.
- **Sample Offending Data Export:** [`offending_records/sa02601_visitation_date_year_consistency.csv`](offending_records/sa02601_visitation_date_year_consistency.csv) (74942 rows)

### `lambda_grid_capping_risk` (ERROR)
- **Category:** Distributional Boundary
- **Total Rows / Entities Evaluated:** 17,472
- **Discrepancies Identified:** 4,399 (25.18%)
- **Description:** Evaluated lambda-grid ceiling of 80.0. Maximum single-watch count observed = 1770. Groups at risk of boundary pinning: 4399 (25.18%).
- **Downstream Impact:** Silently pins posterior mean lambda_hat at grid ceiling, artificially squashing strongest edges.
- **Recommended Remediation:** Widen PipelineConfig.lam_grid upper ceiling to at least 5000 (currently evaluated: 80.0).
- **Sample Offending Data Export:** [`offending_records/lambda_grid_capping_risk.csv`](offending_records/lambda_grid_capping_risk.csv) (4399 rows)

### `plot_to_meadow_complex_inconsistency` (ERROR)
- **Category:** Spatial Hierarchy
- **Total Rows / Entities Evaluated:** 180
- **Discrepancies Identified:** 5 (2.78%)
- **Description:** Plots assigned to contradictory meadows or complexes across files: 5.
- **Downstream Impact:** Corrupts spatial partitioning and meadow-stratified detection probability analysis.
- **Recommended Remediation:** Standardize spatial lookup metadata table and enforce foreign keys.
- **Sample Offending Data Export:** [`offending_records/plot_to_meadow_complex_inconsistency.csv`](offending_records/plot_to_meadow_complex_inconsistency.csv) (10 rows)

### `calendar_year_gaps` (WARNING)
- **Category:** Temporal Continuity
- **Total Rows / Entities Evaluated:** 11
- **Discrepancies Identified:** 2 (18.18%)
- **Description:** Multi-year hiatus detected over survey horizon [2011-2021]. Unsampled years: [2019, 2020].
- **Downstream Impact:** Naive calendar-based train/val/test splits produce empty partitions or invalid temporal decay.
- **Recommended Remediation:** Use cumulative event volume splitting (Algorithm 8) over continuous elapsed time tau.

### `degenerate_flowering_species` (WARNING)
- **Category:** Distributional Boundary
- **Total Rows / Entities Evaluated:** 202
- **Discrepancies Identified:** 38 (18.81%)
- **Description:** Species with zero variance in flower counts: 38 (18.81%).
- **Downstream Impact:** Min-max normalization yields 0/0; triggers neutral 0.5 confidence fallback in Algorithm 5.
- **Recommended Remediation:** Confirm Algorithm 5 handles degenerate species via the 0.5 fallback condition.

### `flw_negative_or_null_flower_count` (WARNING)
- **Category:** Semantic Integrity
- **Total Rows / Entities Evaluated:** 23,642
- **Discrepancies Identified:** 2,154 (9.11%)
- **Description:** Audited NO_FLWS in flowering records. Negative: 0, Null: 2154.
- **Downstream Impact:** Negative counts are physical impossibilities; nulls require fillna(0) for log1p.
- **Recommended Remediation:** Impute null flower counts to 0; investigate origin of any negative values.
- **Sample Offending Data Export:** [`offending_records/flw_negative_or_null_flower_count.csv`](offending_records/flw_negative_or_null_flower_count.csv) (2154 rows)

### `interactions_lacking_same_session_flowering` (WARNING)
- **Category:** Referential Integrity
- **Total Rows / Entities Evaluated:** 61,025
- **Discrepancies Identified:** 107 (0.18%)
- **Description:** Plant interaction records lacking same-session flowering record: 107 (0.175%). Paper observed ~0.24% requiring LOCF fallback.
- **Downstream Impact:** Requires last-observation-carried-forward (LOCF) imputation (Algorithm 6).
- **Recommended Remediation:** Confirm Algorithm 6 join_with_fallback resolves these without unintended defaults.
- **Sample Offending Data Export:** [`offending_records/interactions_lacking_same_session_flowering.csv`](offending_records/interactions_lacking_same_session_flowering.csv) (107 rows)

### `structural_negative_granularity_discrepancy` (WARNING)
- **Category:** Granularity & Aggregation
- **Total Rows / Entities Evaluated:** 13,917
- **Discrepancies Identified:** 12,528 (90.02%)
- **Description:** Negative events exist at two different granularities: 13917 raw minute-level rows vs 1389 deduplicated watch sessions. (Paper Table 1 cited 1,389; Section 2.2 cited 13,917).
- **Downstream Impact:** Users expecting 1,389 negative rows will encounter schema confusion when seeing 13,917 rows.
- **Recommended Remediation:** Export both granularities explicitly or document that 1,389 is deduplicated by (PLOT_ID, YEAR, WATCH).

### `vis_pos_missing_or_zero_visit_count` (WARNING)
- **Category:** Semantic Integrity
- **Total Rows / Entities Evaluated:** 61,025
- **Discrepancies Identified:** 69 (0.11%)
- **Description:** Audited NO_INT in positive interactions for missing or non-positive counts.
- **Downstream Impact:** Algorithm 2 substitutes missing counts with 1. Causes slight variation in raw totals.
- **Recommended Remediation:** Apply fillna(1) as prescribed by Algorithm 2 in paper.
- **Sample Offending Data Export:** [`offending_records/vis_pos_missing_or_zero_visit_count.csv`](offending_records/vis_pos_missing_or_zero_visit_count.csv) (69 rows)

### `visitation_watches_missing_flowering_survey` (WARNING)
- **Category:** Referential Integrity
- **Total Rows / Entities Evaluated:** 4,579
- **Discrepancies Identified:** 4 (0.09%)
- **Description:** Watch sessions surveyed for pollinator visits that have NO corresponding record in flowering survey.
- **Downstream Impact:** Indicates asynchronous sampling sessions or unsynchronized data entry.
- **Recommended Remediation:** Fall back to prior flowering observation (LOCF) for that plot-season.
- **Sample Offending Data Export:** [`offending_records/visitation_watches_missing_flowering_survey.csv`](offending_records/visitation_watches_missing_flowering_survey.csv) (4 rows)

### `SA02601_visitation_missing_columns` (INFO)
- **Category:** Schema Integrity
- **Total Rows / Entities Evaluated:** 14
- **Discrepancies Identified:** 0 (0.00%)
- **Description:** Evaluated expected canonical columns in SA02601_visitation. Missing: []
- **Downstream Impact:** Downstream pipeline failure if required survey attributes are absent.
- **Recommended Remediation:** Ensure source CSV contains required columns or remap via PipelineConfig. Missing: []

### `SA02602_flowering_date_year_consistency` (INFO)
- **Category:** Date Formatting & Consistency
- **Total Rows / Entities Evaluated:** 23,642
- **Discrepancies Identified:** 0 (0.00%)
- **Description:** Evaluated SA02602_flowering 'SAMPLEDATE' with format '%Y-%m-%d'. Failed parsing: 0. Contradicted 'YEAR': 0.
- **Downstream Impact:** Silent date corruption; links events to incorrect field seasons or chronologies.
- **Recommended Remediation:** Verify expected date format string ('%Y-%m-%d'). Remember SA02601 and SA02602 may have varying formats across EDI release versions.

### `SA02602_flowering_missing_columns` (INFO)
- **Category:** Schema Integrity
- **Total Rows / Entities Evaluated:** 13
- **Discrepancies Identified:** 0 (0.00%)
- **Description:** Evaluated expected canonical columns in SA02602_flowering. Missing: []
- **Downstream Impact:** Downstream pipeline failure if required survey attributes are absent.
- **Recommended Remediation:** Ensure source CSV contains required columns or remap via PipelineConfig. Missing: []

### `unmatched_visited_plant_species` (INFO)
- **Category:** Referential Integrity
- **Total Rows / Entities Evaluated:** 147
- **Discrepancies Identified:** 0 (0.00%)
- **Description:** Visited plant species missing from flowering survey catalog: []
- **Downstream Impact:** Interactions on these plants cannot compute phenological confidence and rely purely on default fallback (0.5).
- **Recommended Remediation:** Harmonize species taxonomic codes across both datasets (check synonyms or typo suffixes).

### `vis_neg_interaction_has_species` (INFO)
- **Category:** Semantic Integrity
- **Total Rows / Entities Evaluated:** 13,917
- **Discrepancies Identified:** 0 (0.00%)
- **Description:** Verified that structural negatives (PPI_STATUS=0) have null species codes.
- **Downstream Impact:** Contradiction: record flagged as empty watch but lists a species.
- **Recommended Remediation:** Review if record is an interaction with wrong status flag or stray plant entry.

### `vis_pos_interaction_missing_species` (INFO)
- **Category:** Semantic Integrity
- **Total Rows / Entities Evaluated:** 61,025
- **Discrepancies Identified:** 0 (0.00%)
- **Description:** Verified that PPI_STATUS=1 records contain non-null plant and pollinator species.
- **Downstream Impact:** Interactions cannot be mapped to bipartite graph nodes without species identifiers.
- **Recommended Remediation:** Impute or drop unidentifiable visitor/plant interaction rows.

### `vis_ppi_status_valid_domain` (INFO)
- **Category:** Semantic Integrity
- **Total Rows / Entities Evaluated:** 74,942
- **Discrepancies Identified:** 0 (0.00%)
- **Description:** Checked if PPI_STATUS is strictly binary {0, 1}.
- **Downstream Impact:** Unknown record type; invalid status breaks E+ / E0 partitioning.
- **Recommended Remediation:** Sanitize raw codes to {0: absence, 1: confirmed interaction}.

### `within_watch_minute_level_duplicates` (INFO)
- **Category:** Granularity & Aggregation
- **Total Rows / Entities Evaluated:** 20,018
- **Discrepancies Identified:** 7,591 (37.92%)
- **Description:** Found 7591 plant-pollinator pairs with multiple minute-level visits in the same watch (41007 extra rows collapsed into 20018 watch-level events).
- **Downstream Impact:** Explains the reduction from 61,025 raw positive rows to ~20,025 watch-level positive events.
- **Recommended Remediation:** Ensure GroupSum(..., value=NO_INT) is performed as prescribed in Algorithm 7.
