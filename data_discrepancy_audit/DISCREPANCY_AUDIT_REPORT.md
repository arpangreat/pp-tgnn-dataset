# Dataset Discrepancy & Integrity Audit Report

## 1. Executive Summary

- **Total Checks Evaluated:** 27
- **Errors (Breaking / Data-Corruption Risks):** 2
- **Warnings (Semantic Nuances & Fallback Triggers):** 14
- **Informational (Normal Data Reductions & Milestones):** 11

---

## 2. Findings by Severity

| Severity | Target Dataset | Check Name | Category | Issues Count | Issues % | Impact |
|---|---|---|---|---|---|---|
| **ERROR** | `SA02601_v6.csv & Model Hyperparameters` | `lambda_grid_capping_risk` | Distributional Boundary | 4,399 | 25.18% | Silently pins posterior mean lambda_hat at grid ceiling, artificially squashing strongest edges. |
| **ERROR** | `Both (SA02601 vs SA02602)` | `plot_to_meadow_complex_inconsistency` | Spatial Hierarchy | 5 | 2.78% | Corrupts spatial partitioning and meadow-stratified detection probability analysis. |
| **WARNING** | `SA02601_v6.csv (Visitation)` | `SA02601_visitation_watch_multi_date_splits` | Date Formatting & Consistency | 37 | 0.81% | Watch sessions split across days (e.g. due to weather delays) create ambiguous event timestamps. |
| **WARNING** | `SA02602_v5.csv (Flowering)` | `SA02602_flowering_watch_multi_date_splits` | Date Formatting & Consistency | 229 | 4.09% | Watch sessions split across days (e.g. due to weather delays) create ambiguous event timestamps. |
| **WARNING** | `Both (Survey Sampling Horizon)` | `calendar_year_gaps` | Temporal Continuity | 2 | 18.18% | Naive calendar-based train/val/test splits produce empty partitions or invalid temporal decay. |
| **WARNING** | `Both (SA02601 vs SA02602)` | `cross_dataset_watch_date_mismatch` | Date Formatting & Consistency | 2,018 | 41.73% | A naive inner join requiring SAMPLEDATE equality drops 41.7% of valid watch sessions. |
| **WARNING** | `SA02602_v5.csv (Flowering)` | `degenerate_flowering_species` | Distributional Boundary | 38 | 18.81% | Min-max normalization yields 0/0; triggers neutral 0.5 confidence fallback in Algorithm 5. |
| **WARNING** | `SA02602_v5.csv (Flowering)` | `flw_negative_or_null_flower_count` | Semantic Integrity | 2,154 | 9.11% | Negative counts are physical impossibilities; nulls require fillna(0) for log1p. |
| **WARNING** | `SA02601_v6.csv (Visitation)` | `functional_guild_synonyms_and_typos` | Naming Inconsistency | 13 | 16.67% | Categorical encodings create redundant, fragmented one-hot dimensions for the same biological guild. |
| **WARNING** | `Both (SA02601 vs SA02602)` | `interactions_lacking_same_session_flowering` | Referential Integrity | 107 | 0.18% | Requires last-observation-carried-forward (LOCF) imputation (Algorithm 6). |
| **WARNING** | `Both (SA02601 vs SA02602)` | `observer_naming_style_fragmentation` | Naming Inconsistency | 32 | 29.91% | Observer bias cannot be controlled as a fixed effect in statistical models without disambiguating names. |
| **WARNING** | `SA02601_v6.csv (Visitation)` | `pollinator_type_multi_assignment` | Naming Inconsistency | 143 | 20.69% | Splits functional insect guilds into inconsistent categories (e.g., Bombus bifarius as social vs solitary). |
| **WARNING** | `SA02601_v6.csv (Visitation)` | `structural_negative_granularity_discrepancy` | Granularity & Aggregation | 12,528 | 90.02% | Users expecting 1,389 negative rows will encounter schema confusion when seeing 13,917 rows. |
| **WARNING** | `Both (SA02601 vs SA02602)` | `taxonomic_synonym_botanical_drift` | Naming Inconsistency | 1 | 0.50% | Label inconsistency if grouping by scientific binomial name instead of PLTSP_CODE. |
| **WARNING** | `SA02601_v6.csv (Visitation)` | `vis_pos_missing_or_zero_visit_count` | Semantic Integrity | 69 | 0.11% | Algorithm 2 substitutes missing counts with 1. Causes slight variation in raw totals. |
| **WARNING** | `Both (SA02601 vs SA02602)` | `visitation_watches_missing_flowering_survey` | Referential Integrity | 4 | 0.09% | Indicates asynchronous sampling sessions or unsynchronized data entry. |
| INFO | `SA02601_v6.csv (Visitation)` | `SA02601_visitation_date_consistency` | Date Formatting & Consistency | 0 | 0.00% | Clean; dates parse unambiguously. |
| INFO | `SA02601_v6.csv (Visitation)` | `SA02601_visitation_highly_sparse_fields` | Field Sparsity | 4 | 15.38% | Downstream models cannot rely on these features without heavy imputation (e.g., TEMP 38% missing). |
| INFO | `SA02601_v6.csv (Visitation)` | `SA02601_visitation_missing_columns` | Schema Integrity | 0 | 0.00% | Downstream pipeline failure if required survey attributes are absent. |
| INFO | `SA02602_v5.csv (Flowering)` | `SA02602_flowering_date_consistency` | Date Formatting & Consistency | 0 | 0.00% | Clean; dates parse unambiguously. |
| INFO | `SA02602_v5.csv (Flowering)` | `SA02602_flowering_highly_sparse_fields` | Field Sparsity | 1 | 6.25% | Downstream models cannot rely on these features without heavy imputation (e.g., TEMP 38% missing). |
| INFO | `SA02602_v5.csv (Flowering)` | `SA02602_flowering_missing_columns` | Schema Integrity | 0 | 0.00% | Downstream pipeline failure if required survey attributes are absent. |
| INFO | `Both (SA02601 vs SA02602)` | `unmatched_visited_plant_species` | Referential Integrity | 0 | 0.00% | Interactions on these plants cannot compute phenological confidence and rely purely on default fallback (0.5). |
| INFO | `SA02601_v6.csv (Visitation)` | `vis_neg_interaction_has_species` | Semantic Integrity | 0 | 0.00% | Contradiction: record flagged as empty watch but lists a species. |
| INFO | `SA02601_v6.csv (Visitation)` | `vis_pos_interaction_missing_species` | Semantic Integrity | 0 | 0.00% | Interactions cannot be mapped to bipartite graph nodes without species identifiers. |
| INFO | `SA02601_v6.csv (Visitation)` | `vis_ppi_status_valid_domain` | Semantic Integrity | 0 | 0.00% | Unknown record type; invalid status breaks E+ / E0 partitioning. |
| INFO | `SA02601_v6.csv (Visitation)` | `within_watch_minute_level_duplicates` | Granularity & Aggregation | 7,591 | 37.92% | Explains the reduction from 61,025 raw positive rows to ~20,025 watch-level positive events. |

---

## 3. Detailed Descriptions & Remediation Guidance

### `lambda_grid_capping_risk` (ERROR)
- **Target Dataset:** SA02601_v6.csv & Model Hyperparameters
- **Category:** Distributional Boundary
- **Total Rows / Entities Evaluated:** 17,472
- **Discrepancies Identified:** 4,399 (25.18%)
- **Description:** Evaluated lambda-grid ceiling of 80.0. Maximum single-watch count observed = 1770. Groups at risk of boundary pinning: 4399 (25.18%).
- **Downstream Impact:** Silently pins posterior mean lambda_hat at grid ceiling, artificially squashing strongest edges.
- **Recommended Remediation:** Widen PipelineConfig.lam_grid upper ceiling to at least 5000 (currently evaluated: 80.0).
- **Sample Offending Data Export:** [`offending_records/lambda_grid_capping_risk.csv`](offending_records/lambda_grid_capping_risk.csv) (4399 rows)

### `plot_to_meadow_complex_inconsistency` (ERROR)
- **Target Dataset:** Both (SA02601 vs SA02602)
- **Category:** Spatial Hierarchy
- **Total Rows / Entities Evaluated:** 180
- **Discrepancies Identified:** 5 (2.78%)
- **Description:** Plots assigned to contradictory meadows or complexes across files: 5.
- **Downstream Impact:** Corrupts spatial partitioning and meadow-stratified detection probability analysis.
- **Recommended Remediation:** Standardize spatial lookup metadata table and enforce foreign keys.
- **Sample Offending Data Export:** [`offending_records/plot_to_meadow_complex_inconsistency.csv`](offending_records/plot_to_meadow_complex_inconsistency.csv) (10 rows)

### `SA02601_visitation_watch_multi_date_splits` (WARNING)
- **Target Dataset:** SA02601_v6.csv (Visitation)
- **Category:** Date Formatting & Consistency
- **Total Rows / Entities Evaluated:** 4,579
- **Discrepancies Identified:** 37 (0.81%)
- **Description:** Found 37 watch sessions in SA02601_visitation that span multiple distinct calendar dates (0.81% of sessions). Sample: {"('CNE03', 2012, 1.0)": ['2012-07-09', '2012-07-10'], "('CNE04', 2012, 1.0)": ['2012-07-09', '2012-07-10'], "('CNE05', 2012, 1.0)": ['2012-07-09', '2012-07-10'], "('CNE06', 2012, 1.0)": ['2012-07-09', '2012-07-10'], "('CNM06', 2012, 1.0)": ['2012-07-09', '2012-07-10']}
- **Downstream Impact:** Watch sessions split across days (e.g. due to weather delays) create ambiguous event timestamps.
- **Recommended Remediation:** Assign the session's start date or aggregate observations at watch session granularity.

### `SA02602_flowering_watch_multi_date_splits` (WARNING)
- **Target Dataset:** SA02602_v5.csv (Flowering)
- **Category:** Date Formatting & Consistency
- **Total Rows / Entities Evaluated:** 5,602
- **Discrepancies Identified:** 229 (4.09%)
- **Description:** Found 229 watch sessions in SA02602_flowering that span multiple distinct calendar dates (4.09% of sessions). Sample: {"('BGD03', 2011, 2.0)": ['2011-07-24', '2011-07-25'], "('BGD07', 2012, 2.0)": ['2012-07-05', '2012-07-12'], "('CNE01', 2011, 1.0)": ['2011-07-21', '2011-07-22'], "('CNE01', 2011, 4.0)": ['2011-08-11', '2011-08-12'], "('CNE02', 2011, 3.0)": ['2011-08-04', '2011-08-05']}
- **Downstream Impact:** Watch sessions split across days (e.g. due to weather delays) create ambiguous event timestamps.
- **Recommended Remediation:** Assign the session's start date or aggregate observations at watch session granularity.

### `calendar_year_gaps` (WARNING)
- **Target Dataset:** Both (Survey Sampling Horizon)
- **Category:** Temporal Continuity
- **Total Rows / Entities Evaluated:** 11
- **Discrepancies Identified:** 2 (18.18%)
- **Description:** Multi-year hiatus detected over survey horizon [2011-2021]. Unsampled years: [2019, 2020].
- **Downstream Impact:** Naive calendar-based train/val/test splits produce empty partitions or invalid temporal decay.
- **Recommended Remediation:** Use cumulative event volume splitting (Algorithm 8) over continuous elapsed time tau.

### `cross_dataset_watch_date_mismatch` (WARNING)
- **Target Dataset:** Both (SA02601 vs SA02602)
- **Category:** Date Formatting & Consistency
- **Total Rows / Entities Evaluated:** 4,836
- **Discrepancies Identified:** 2,018 (41.73%)
- **Description:** Found 2018 shared watch sessions (41.73%) where visitation and flowering surveys occurred on DIFFERENT calendar dates (offsets up to 20 days, median offset 1 day). Most common offset is +1 day (flowering surveyed the day after insect watches).
- **Downstream Impact:** A naive inner join requiring SAMPLEDATE equality drops 41.7% of valid watch sessions.
- **Recommended Remediation:** Join visitation and flowering on [PLOT_ID, YEAR, WATCH, PLTSP_CODE] without enforcing SAMPLEDATE equality.
- **Sample Offending Data Export:** [`offending_records/cross_dataset_watch_date_mismatch.csv`](offending_records/cross_dataset_watch_date_mismatch.csv) (2018 rows)

### `degenerate_flowering_species` (WARNING)
- **Target Dataset:** SA02602_v5.csv (Flowering)
- **Category:** Distributional Boundary
- **Total Rows / Entities Evaluated:** 202
- **Discrepancies Identified:** 38 (18.81%)
- **Description:** Species with zero variance in flower counts: 38 (18.81%).
- **Downstream Impact:** Min-max normalization yields 0/0; triggers neutral 0.5 confidence fallback in Algorithm 5.
- **Recommended Remediation:** Confirm Algorithm 5 handles degenerate species via the 0.5 fallback condition.

### `flw_negative_or_null_flower_count` (WARNING)
- **Target Dataset:** SA02602_v5.csv (Flowering)
- **Category:** Semantic Integrity
- **Total Rows / Entities Evaluated:** 23,642
- **Discrepancies Identified:** 2,154 (9.11%)
- **Description:** Audited NO_FLWS in flowering records. Negative: 0, Null: 2154.
- **Downstream Impact:** Negative counts are physical impossibilities; nulls require fillna(0) for log1p.
- **Recommended Remediation:** Impute null flower counts to 0; investigate origin of any negative values.
- **Sample Offending Data Export:** [`offending_records/flw_negative_or_null_flower_count.csv`](offending_records/flw_negative_or_null_flower_count.csv) (2154 rows)

### `functional_guild_synonyms_and_typos` (WARNING)
- **Target Dataset:** SA02601_v6.csv (Visitation)
- **Category:** Naming Inconsistency
- **Total Rows / Entities Evaluated:** 78
- **Discrepancies Identified:** 13 (16.67%)
- **Description:** Categorical naming variations detected in VISSP_TYPE: Typos: {'herbivorous muscoid fly': 1026, 'predaceous muscoid fly': 1073, 'herbivore-parasitic muscoid fl': 255, 'muscoid fly': 80, 'beeetle': 5, 'kloeptoparasitic bee': 2, 'decomposer muscoid fly': 5, 'predaceous onm bees': 1}. Synonym/Punctuation splits: ["'true-bug' vs 'true bug'", "'social-wasp' vs 'social wasp'", "'hover-fly' vs 'Hover-fly'", "'beetle' vs 'herbivorous beetle'", "'grasshopper' vs 'herbivorous grasshopper'"].
- **Downstream Impact:** Categorical encodings create redundant, fragmented one-hot dimensions for the same biological guild.
- **Recommended Remediation:** Standardize string representation (e.g. replace hyphens, fix 'beeetle' -> 'beetle').

### `interactions_lacking_same_session_flowering` (WARNING)
- **Target Dataset:** Both (SA02601 vs SA02602)
- **Category:** Referential Integrity
- **Total Rows / Entities Evaluated:** 61,025
- **Discrepancies Identified:** 107 (0.18%)
- **Description:** Plant interaction records lacking same-session flowering record: 107 (0.175%). Paper observed ~0.24% requiring LOCF fallback.
- **Downstream Impact:** Requires last-observation-carried-forward (LOCF) imputation (Algorithm 6).
- **Recommended Remediation:** Confirm Algorithm 6 join_with_fallback resolves these without unintended defaults.
- **Sample Offending Data Export:** [`offending_records/interactions_lacking_same_session_flowering.csv`](offending_records/interactions_lacking_same_session_flowering.csv) (107 rows)

### `observer_naming_style_fragmentation` (WARNING)
- **Target Dataset:** Both (SA02601 vs SA02602)
- **Category:** Naming Inconsistency
- **Total Rows / Entities Evaluated:** 107
- **Discrepancies Identified:** 32 (29.91%)
- **Description:** Observer column contains mixed naming styles: 44 uppercase initials (e.g. 'AS', 'ARM'), 20 single first names (e.g. 'Andy', 'Vera'), 24 concatenated observer teams (e.g. 'Andy and Tim', 'Melinda, Logan'), and 8 field log annotations (e.g. 'ARMchange watch', 'xARMx').
- **Downstream Impact:** Observer bias cannot be controlled as a fixed effect in statistical models without disambiguating names.
- **Recommended Remediation:** Create a standardized observer identity dictionary mapping names, teams, and initials to unique IDs.

### `pollinator_type_multi_assignment` (WARNING)
- **Target Dataset:** SA02601_v6.csv (Visitation)
- **Category:** Naming Inconsistency
- **Total Rows / Entities Evaluated:** 691
- **Discrepancies Identified:** 143 (20.69%)
- **Description:** Found 143 pollinator taxa assigned to multiple different functional guilds (VISSP_TYPE). Sample: {'ACRISPX1': ['herbivorous grasshopper', 'grasshopper'], 'ADELTRIS': ['herbivorous butterfly; diurnal', 'diurnal moth'], 'AMMOSPX1': ['predaceous wasp', 'halictine bee'], 'ANASRUFA': ['herbivorous beetle', 'beetle'], 'ANTHRSP1': ['predaceous bee-fly', 'herbivore-parasitic bee-fly'], 'APHISPX1': ['herbivorous true-bug', 'true bug'], 'ASEMPOLY': ['decomposer hover-fly', 'hover-fly'], 'ASILSPX1': ['pollinator-predaceous fly', 'predaceous onm bees']}
- **Downstream Impact:** Splits functional insect guilds into inconsistent categories (e.g., Bombus bifarius as social vs solitary).
- **Recommended Remediation:** Construct a canonical 1-to-1 species->guild lookup table to harmonize visitor types.

### `structural_negative_granularity_discrepancy` (WARNING)
- **Target Dataset:** SA02601_v6.csv (Visitation)
- **Category:** Granularity & Aggregation
- **Total Rows / Entities Evaluated:** 13,917
- **Discrepancies Identified:** 12,528 (90.02%)
- **Description:** Negative events exist at two different granularities: 13917 raw minute-level rows vs 1389 deduplicated watch sessions. (Paper Table 1 cited 1,389; Section 2.2 cited 13,917).
- **Downstream Impact:** Users expecting 1,389 negative rows will encounter schema confusion when seeing 13,917 rows.
- **Recommended Remediation:** Export both granularities explicitly or document that 1,389 is deduplicated by (PLOT_ID, YEAR, WATCH).

### `taxonomic_synonym_botanical_drift` (WARNING)
- **Target Dataset:** Both (SA02601 vs SA02602)
- **Category:** Naming Inconsistency
- **Total Rows / Entities Evaluated:** 202
- **Discrepancies Identified:** 1 (0.50%)
- **Description:** Plant species codes with shifting scientific names across survey years: {'ZIGAVENE': ['Zigadenus venenosus', 'Toxicoscordion venenosum']}. Example: ZIGAVENE shifts between 'Zigadenus venenosus' and 'Toxicoscordion venenosum'.
- **Downstream Impact:** Label inconsistency if grouping by scientific binomial name instead of PLTSP_CODE.
- **Recommended Remediation:** Always use canonical PLTSP_CODE for relational joins; maintain a taxonomy synonym crosswalk.

### `vis_pos_missing_or_zero_visit_count` (WARNING)
- **Target Dataset:** SA02601_v6.csv (Visitation)
- **Category:** Semantic Integrity
- **Total Rows / Entities Evaluated:** 61,025
- **Discrepancies Identified:** 69 (0.11%)
- **Description:** Audited NO_INT in positive interactions for missing or non-positive counts.
- **Downstream Impact:** Algorithm 2 substitutes missing counts with 1. Causes slight variation in raw totals.
- **Recommended Remediation:** Apply fillna(1) as prescribed by Algorithm 2 in paper.
- **Sample Offending Data Export:** [`offending_records/vis_pos_missing_or_zero_visit_count.csv`](offending_records/vis_pos_missing_or_zero_visit_count.csv) (69 rows)

### `visitation_watches_missing_flowering_survey` (WARNING)
- **Target Dataset:** Both (SA02601 vs SA02602)
- **Category:** Referential Integrity
- **Total Rows / Entities Evaluated:** 4,579
- **Discrepancies Identified:** 4 (0.09%)
- **Description:** Watch sessions surveyed for pollinator visits that have NO corresponding record in flowering survey.
- **Downstream Impact:** Indicates asynchronous sampling sessions or unsynchronized data entry.
- **Recommended Remediation:** Fall back to prior flowering observation (LOCF) for that plot-season.
- **Sample Offending Data Export:** [`offending_records/visitation_watches_missing_flowering_survey.csv`](offending_records/visitation_watches_missing_flowering_survey.csv) (4 rows)

### `SA02601_visitation_date_consistency` (INFO)
- **Target Dataset:** SA02601_v6.csv (Visitation)
- **Category:** Date Formatting & Consistency
- **Total Rows / Entities Evaluated:** 74,942
- **Discrepancies Identified:** 0 (0.00%)
- **Description:** Dates in SA02601_visitation are 100% internally consistent: All 74,942 rows follow uniform 'YYYY-MM-DD' and match the recorded 'YEAR' column.
- **Downstream Impact:** Clean; dates parse unambiguously.
- **Recommended Remediation:** None required.

### `SA02601_visitation_highly_sparse_fields` (INFO)
- **Target Dataset:** SA02601_v6.csv (Visitation)
- **Category:** Field Sparsity
- **Total Rows / Entities Evaluated:** 26
- **Discrepancies Identified:** 4 (15.38%)
- **Description:** Fields where over 30% of records are missing or unrecorded: {'END_TIME': np.float64(89.99), 'TEMP': np.float64(38.03), 'REF_NO': np.float64(79.75), 'QC_NOTES': np.float64(95.15)}.
- **Downstream Impact:** Downstream models cannot rely on these features without heavy imputation (e.g., TEMP 38% missing).
- **Recommended Remediation:** Exclude highly sparse attributes from graph node/edge features or apply targeted imputation.

### `SA02601_visitation_missing_columns` (INFO)
- **Target Dataset:** SA02601_v6.csv (Visitation)
- **Category:** Schema Integrity
- **Total Rows / Entities Evaluated:** 14
- **Discrepancies Identified:** 0 (0.00%)
- **Description:** Evaluated expected canonical columns in SA02601_visitation. Missing: []
- **Downstream Impact:** Downstream pipeline failure if required survey attributes are absent.
- **Recommended Remediation:** Ensure source CSV contains required columns or remap via PipelineConfig. Missing: []

### `SA02602_flowering_date_consistency` (INFO)
- **Target Dataset:** SA02602_v5.csv (Flowering)
- **Category:** Date Formatting & Consistency
- **Total Rows / Entities Evaluated:** 23,642
- **Discrepancies Identified:** 0 (0.00%)
- **Description:** Dates in SA02602_flowering are 100% internally consistent: All 23,642 rows follow uniform 'YYYY-MM-DD' and match the recorded 'YEAR' column.
- **Downstream Impact:** Clean; dates parse unambiguously.
- **Recommended Remediation:** None required.

### `SA02602_flowering_highly_sparse_fields` (INFO)
- **Target Dataset:** SA02602_v5.csv (Flowering)
- **Category:** Field Sparsity
- **Total Rows / Entities Evaluated:** 16
- **Discrepancies Identified:** 1 (6.25%)
- **Description:** Fields where over 30% of records are missing or unrecorded: {'QC_NOTES': np.float64(76.99)}.
- **Downstream Impact:** Downstream models cannot rely on these features without heavy imputation (e.g., TEMP 38% missing).
- **Recommended Remediation:** Exclude highly sparse attributes from graph node/edge features or apply targeted imputation.

### `SA02602_flowering_missing_columns` (INFO)
- **Target Dataset:** SA02602_v5.csv (Flowering)
- **Category:** Schema Integrity
- **Total Rows / Entities Evaluated:** 13
- **Discrepancies Identified:** 0 (0.00%)
- **Description:** Evaluated expected canonical columns in SA02602_flowering. Missing: []
- **Downstream Impact:** Downstream pipeline failure if required survey attributes are absent.
- **Recommended Remediation:** Ensure source CSV contains required columns or remap via PipelineConfig. Missing: []

### `unmatched_visited_plant_species` (INFO)
- **Target Dataset:** Both (SA02601 vs SA02602)
- **Category:** Referential Integrity
- **Total Rows / Entities Evaluated:** 147
- **Discrepancies Identified:** 0 (0.00%)
- **Description:** Visited plant species missing from flowering survey catalog: []
- **Downstream Impact:** Interactions on these plants cannot compute phenological confidence and rely purely on default fallback (0.5).
- **Recommended Remediation:** Harmonize species taxonomic codes across both datasets (check synonyms or typo suffixes).

### `vis_neg_interaction_has_species` (INFO)
- **Target Dataset:** SA02601_v6.csv (Visitation)
- **Category:** Semantic Integrity
- **Total Rows / Entities Evaluated:** 13,917
- **Discrepancies Identified:** 0 (0.00%)
- **Description:** Verified that structural negatives (PPI_STATUS=0) have null species codes.
- **Downstream Impact:** Contradiction: record flagged as empty watch but lists a species.
- **Recommended Remediation:** Review if record is an interaction with wrong status flag or stray plant entry.

### `vis_pos_interaction_missing_species` (INFO)
- **Target Dataset:** SA02601_v6.csv (Visitation)
- **Category:** Semantic Integrity
- **Total Rows / Entities Evaluated:** 61,025
- **Discrepancies Identified:** 0 (0.00%)
- **Description:** Verified that PPI_STATUS=1 records contain non-null plant and pollinator species.
- **Downstream Impact:** Interactions cannot be mapped to bipartite graph nodes without species identifiers.
- **Recommended Remediation:** Impute or drop unidentifiable visitor/plant interaction rows.

### `vis_ppi_status_valid_domain` (INFO)
- **Target Dataset:** SA02601_v6.csv (Visitation)
- **Category:** Semantic Integrity
- **Total Rows / Entities Evaluated:** 74,942
- **Discrepancies Identified:** 0 (0.00%)
- **Description:** Checked if PPI_STATUS is strictly binary {0, 1}.
- **Downstream Impact:** Unknown record type; invalid status breaks E+ / E0 partitioning.
- **Recommended Remediation:** Sanitize raw codes to {0: absence, 1: confirmed interaction}.

### `within_watch_minute_level_duplicates` (INFO)
- **Target Dataset:** SA02601_v6.csv (Visitation)
- **Category:** Granularity & Aggregation
- **Total Rows / Entities Evaluated:** 20,018
- **Discrepancies Identified:** 7,591 (37.92%)
- **Description:** Found 7591 plant-pollinator pairs with multiple minute-level visits in the same watch (41007 extra rows collapsed into 20018 watch-level events).
- **Downstream Impact:** Explains the reduction from 61,025 raw positive rows to ~20,025 watch-level positive events.
- **Recommended Remediation:** Ensure GroupSum(..., value=NO_INT) is performed as prescribed in Algorithm 7.
