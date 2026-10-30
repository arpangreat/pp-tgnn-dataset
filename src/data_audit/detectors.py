"""
Individual discrepancy detectors for SA02601 (visitation) and SA02602 (flowering).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DiscrepancyResult:
    category: str
    check_name: str
    severity: str  # "ERROR", "WARNING", "INFO"
    total_evaluated: int
    discrepancy_count: int
    discrepancy_pct: float
    description: str
    impact: str
    remediation: str
    offending_data: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "check_name": self.check_name,
            "severity": self.severity,
            "total_evaluated": self.total_evaluated,
            "discrepancy_count": self.discrepancy_count,
            "discrepancy_pct": round(self.discrepancy_pct, 4),
            "description": self.description,
            "impact": self.impact,
            "remediation": self.remediation,
            "metadata": self.metadata,
        }


# ==============================================================================
# 1. Schema & Date Formatting Detector
# ==============================================================================

class SchemaAndDateDetector:
    """Detects missing columns, unparseable dates, and parsed-date vs YEAR mismatches."""

    EXPECTED_VIS_COLS = [
        "DBCODE", "ENTITY", "COMPLEX", "MEADOW", "PLOT_ID", "YEAR",
        "SAMPLEDATE", "WATCH", "OBSERVER", "PLOT", "PPI_STATUS", "NO_INT",
        "PLTSP_CODE", "VISSP_CODE"
    ]
    EXPECTED_FLW_COLS = [
        "DBCODE", "ENTITY", "COMPLEX", "MEADOW", "PLOT_ID", "YEAR",
        "SAMPLEDATE", "WATCH", "OBSERVER", "PLOT", "FLW_STATUS",
        "PLTSP_CODE", "NO_FLWS"
    ]

    @staticmethod
    def audit_columns(df: pd.DataFrame, dataset_name: str, expected_cols: List[str]) -> DiscrepancyResult:
        missing = [col for col in expected_cols if col not in df.columns]
        extra = [col for col in df.columns if col not in expected_cols]
        count = len(missing)
        pct = (count / len(expected_cols)) * 100 if expected_cols else 0.0

        return DiscrepancyResult(
            category="Schema Integrity",
            check_name=f"{dataset_name}_missing_columns",
            severity="ERROR" if count > 0 else "INFO",
            total_evaluated=len(expected_cols),
            discrepancy_count=count,
            discrepancy_pct=pct,
            description=f"Evaluated expected canonical columns in {dataset_name}. Missing: {missing}",
            impact="Downstream pipeline failure if required survey attributes are absent.",
            remediation=f"Ensure source CSV contains required columns or remap via PipelineConfig. Missing: {missing}",
            metadata={"missing_columns": missing, "extra_columns": extra},
        )

    @staticmethod
    def audit_date_format(
        df: pd.DataFrame, dataset_name: str, date_col: str, expected_fmt: str, year_col: str = "YEAR"
    ) -> Tuple[DiscrepancyResult, pd.Series]:
        """Audits SAMPLEDATE parsing and asserts consistency with YEAR.
        If expected_fmt fails, probes alternative common date formats.
        """
        n_total = len(df)
        parsed_dates = pd.to_datetime(df[date_col], format=expected_fmt, errors="coerce")
        null_dates = parsed_dates.isna()
        n_parse_fails = int(null_dates.sum())

        detected_fmt = None
        alt_parsed = None
        if n_parse_fails > 0:
            candidate_formats = ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d", "%d/%m/%Y"]
            for cand in candidate_formats:
                if cand == expected_fmt:
                    continue
                test_parsed = pd.to_datetime(df[date_col], format=cand, errors="coerce")
                if test_parsed.notna().all() and (test_parsed.dt.year == df[year_col]).all():
                    detected_fmt = cand
                    alt_parsed = test_parsed
                    break

        # Check year consistency with current parsed_dates
        if n_parse_fails == 0:
            year_mismatch = (parsed_dates.dt.year != df[year_col]) & (~null_dates)
            n_mismatches = int(year_mismatch.sum())
        else:
            year_mismatch = pd.Series(False, index=df.index)
            n_mismatches = 0

        total_issues = n_parse_fails + n_mismatches
        pct_issues = (total_issues / n_total) * 100 if n_total > 0 else 0.0

        offending_mask = null_dates | year_mismatch
        offending_df = df[offending_mask].copy() if total_issues > 0 else None
        if offending_df is not None:
            offending_df["_parsed_date"] = parsed_dates[offending_mask]
            offending_df["_issue_type"] = np.where(null_dates[offending_mask], "parse_failed", "year_contradiction")

        if detected_fmt:
            description = (
                f"Format mismatch in {dataset_name}: Expected format '{expected_fmt}' failed on {n_parse_fails} rows. "
                f"Auto-detection identified actual raw format as '{detected_fmt}' (100% match with '{year_col}')."
            )
            impact = (
                f"Pipeline execution with default config '{expected_fmt}' will immediately crash. "
                f"Must set PipelineConfig.{'vis' if 'vis' in dataset_name else 'flw'}_date_fmt = '{detected_fmt}'."
            )
            remediation = f"Update PipelineConfig(vis_date_fmt='{detected_fmt}') when running pipeline on this dataset version."
            # Use successfully detected dates for downstream continuity checks
            valid_dates_for_downstream = alt_parsed
        else:
            description = (
                f"Evaluated {dataset_name} '{date_col}' with format '{expected_fmt}'. "
                f"Failed parsing: {n_parse_fails}. Contradicted '{year_col}': {n_mismatches}."
            )
            impact = "Silent date corruption; links events to incorrect field seasons or chronologies."
            remediation=(
                f"Verify expected date format string ('{expected_fmt}'). Remember SA02601 and SA02602 "
                "may have varying formats across EDI release versions."
            )
            valid_dates_for_downstream = parsed_dates

        result = DiscrepancyResult(
            category="Date Formatting & Consistency",
            check_name=f"{dataset_name}_date_year_consistency",
            severity="ERROR" if total_issues > 0 else "INFO",
            total_evaluated=n_total,
            discrepancy_count=total_issues,
            discrepancy_pct=pct_issues,
            description=description,
            impact=impact,
            remediation=remediation,
            offending_data=offending_df,
            metadata={
                "expected_format": expected_fmt,
                "detected_actual_format": detected_fmt,
                "parse_failures": n_parse_fails,
                "year_mismatches": n_mismatches,
            },
        )
        return result, valid_dates_for_downstream


# ==============================================================================
# 2. Semantic & Status Flag Detector
# ==============================================================================

class SemanticStatusDetector:
    """Audits PPI_STATUS and FLW_STATUS against their associated entity and count fields."""

    @staticmethod
    def audit_ppi_status_invariants(vis_df: pd.DataFrame) -> List[DiscrepancyResult]:
        results = []
        n_total = len(vis_df)

        # Check 1: PPI_STATUS domain
        invalid_status = ~vis_df["PPI_STATUS"].isin([0, 1])
        n_invalid_status = int(invalid_status.sum())
        results.append(DiscrepancyResult(
            category="Semantic Integrity",
            check_name="vis_ppi_status_valid_domain",
            severity="ERROR" if n_invalid_status > 0 else "INFO",
            total_evaluated=n_total,
            discrepancy_count=n_invalid_status,
            discrepancy_pct=(n_invalid_status / n_total) * 100 if n_total else 0.0,
            description="Checked if PPI_STATUS is strictly binary {0, 1}.",
            impact="Unknown record type; invalid status breaks E+ / E0 partitioning.",
            remediation="Sanitize raw codes to {0: absence, 1: confirmed interaction}.",
            offending_data=vis_df[invalid_status] if n_invalid_status else None,
        ))

        # Check 2: PPI_STATUS == 1 must have plant and pollinator species
        pos_mask = vis_df["PPI_STATUS"] == 1
        pos_df = vis_df[pos_mask]
        missing_species = pos_df["PLTSP_CODE"].isna() | pos_df.get("VISSP_CODE", pd.Series(dtype=object)).isna()
        n_missing_species = int(missing_species.sum())
        results.append(DiscrepancyResult(
            category="Semantic Integrity",
            check_name="vis_pos_interaction_missing_species",
            severity="ERROR" if n_missing_species > 0 else "INFO",
            total_evaluated=len(pos_df),
            discrepancy_count=n_missing_species,
            discrepancy_pct=(n_missing_species / len(pos_df)) * 100 if len(pos_df) else 0.0,
            description="Verified that PPI_STATUS=1 records contain non-null plant and pollinator species.",
            impact="Interactions cannot be mapped to bipartite graph nodes without species identifiers.",
            remediation="Impute or drop unidentifiable visitor/plant interaction rows.",
            offending_data=pos_df[missing_species] if n_missing_species else None,
        ))

        # Check 3: PPI_STATUS == 0 should NOT have plant species assigned
        neg_mask = vis_df["PPI_STATUS"] == 0
        neg_df = vis_df[neg_mask]
        unexpected_species = neg_df["PLTSP_CODE"].notna()
        n_unexpected_species = int(unexpected_species.sum())
        results.append(DiscrepancyResult(
            category="Semantic Integrity",
            check_name="vis_neg_interaction_has_species",
            severity="WARNING" if n_unexpected_species > 0 else "INFO",
            total_evaluated=len(neg_df),
            discrepancy_count=n_unexpected_species,
            discrepancy_pct=(n_unexpected_species / len(neg_df)) * 100 if len(neg_df) else 0.0,
            description="Verified that structural negatives (PPI_STATUS=0) have null species codes.",
            impact="Contradiction: record flagged as empty watch but lists a species.",
            remediation="Review if record is an interaction with wrong status flag or stray plant entry.",
            offending_data=neg_df[unexpected_species] if n_unexpected_species else None,
        ))

        # Check 4: Missing interaction counts in E+
        if "NO_INT" in pos_df.columns:
            missing_no_int = pos_df["NO_INT"].isna() | (pos_df["NO_INT"] <= 0)
            n_missing_no_int = int(missing_no_int.sum())
            results.append(DiscrepancyResult(
                category="Semantic Integrity",
                check_name="vis_pos_missing_or_zero_visit_count",
                severity="WARNING" if n_missing_no_int > 0 else "INFO",
                total_evaluated=len(pos_df),
                discrepancy_count=n_missing_no_int,
                discrepancy_pct=(n_missing_no_int / len(pos_df)) * 100 if len(pos_df) else 0.0,
                description="Audited NO_INT in positive interactions for missing or non-positive counts.",
                impact="Algorithm 2 substitutes missing counts with 1. Causes slight variation in raw totals.",
                remediation="Apply fillna(1) as prescribed by Algorithm 2 in paper.",
                offending_data=pos_df[missing_no_int] if n_missing_no_int else None,
            ))

        return results

    @staticmethod
    def audit_flw_status_invariants(flw_df: pd.DataFrame) -> List[DiscrepancyResult]:
        results = []
        n_total = len(flw_df)

        if "NO_FLWS" in flw_df.columns:
            neg_flws = flw_df["NO_FLWS"] < 0
            n_neg_flws = int(neg_flws.sum())
            null_flws = flw_df["NO_FLWS"].isna()
            n_null_flws = int(null_flws.sum())

            results.append(DiscrepancyResult(
                category="Semantic Integrity",
                check_name="flw_negative_or_null_flower_count",
                severity="ERROR" if n_neg_flws > 0 else ("WARNING" if n_null_flws > 0 else "INFO"),
                total_evaluated=n_total,
                discrepancy_count=n_neg_flws + n_null_flws,
                discrepancy_pct=((n_neg_flws + n_null_flws) / n_total) * 100 if n_total else 0.0,
                description=f"Audited NO_FLWS in flowering records. Negative: {n_neg_flws}, Null: {n_null_flws}.",
                impact="Negative counts are physical impossibilities; nulls require fillna(0) for log1p.",
                remediation="Impute null flower counts to 0; investigate origin of any negative values.",
                offending_data=flw_df[neg_flws | null_flws] if (n_neg_flws + n_null_flws) else None,
                metadata={"negative_counts": n_neg_flws, "null_counts": n_null_flws},
            ))

        return results


# ==============================================================================
# 3. Referential Integrity & Relational Alignment Detector
# ==============================================================================

class ReferentialIntegrityDetector:
    """Audits cross-dataset joins, foreign key relationships, and spatial hierarchies."""

    @staticmethod
    def audit_species_vocabulary_overlap(vis_df: pd.DataFrame, flw_df: pd.DataFrame) -> DiscrepancyResult:
        vis_plants = set(vis_df.loc[vis_df["PPI_STATUS"] == 1, "PLTSP_CODE"].dropna().unique())
        flw_plants = set(flw_df["PLTSP_CODE"].dropna().unique())

        unmatched_plants = vis_plants - flw_plants
        n_unmatched = len(unmatched_plants)
        pct = (n_unmatched / len(vis_plants)) * 100 if vis_plants else 0.0

        offending_rows = None
        if n_unmatched > 0:
            offending_mask = (vis_df["PPI_STATUS"] == 1) & (vis_df["PLTSP_CODE"].isin(unmatched_plants))
            offending_rows = vis_df[offending_mask]

        return DiscrepancyResult(
            category="Referential Integrity",
            check_name="unmatched_visited_plant_species",
            severity="ERROR" if n_unmatched > 0 else "INFO",
            total_evaluated=len(vis_plants),
            discrepancy_count=n_unmatched,
            discrepancy_pct=pct,
            description=f"Visited plant species missing from flowering survey catalog: {sorted(list(unmatched_plants))}",
            impact="Interactions on these plants cannot compute phenological confidence and rely purely on default fallback (0.5).",
            remediation="Harmonize species taxonomic codes across both datasets (check synonyms or typo suffixes).",
            offending_data=offending_rows,
            metadata={"unmatched_species": sorted(list(unmatched_plants))},
        )

    @staticmethod
    def audit_watch_level_join_coverage(
        vis_df: pd.DataFrame, flw_df: pd.DataFrame
    ) -> Tuple[DiscrepancyResult, DiscrepancyResult]:
        """Audits:
        1. Watch sessions conducted in vis but completely missing in flw.
        2. Positive interaction records lacking concurrent same-session flowering survey (needing LOCF fallback).
        """
        key_cols = ["PLOT_ID", "YEAR", "WATCH"]

        vis_watches = vis_df[key_cols].drop_duplicates()
        flw_watches = flw_df[key_cols].drop_duplicates()

        merged_watches = vis_watches.merge(flw_watches, on=key_cols, how="left", indicator=True)
        missing_watch_surveys = merged_watches[merged_watches["_merge"] == "left_only"]
        n_missing_watches = len(missing_watch_surveys)

        res_watches = DiscrepancyResult(
            category="Referential Integrity",
            check_name="visitation_watches_missing_flowering_survey",
            severity="WARNING" if n_missing_watches > 0 else "INFO",
            total_evaluated=len(vis_watches),
            discrepancy_count=n_missing_watches,
            discrepancy_pct=(n_missing_watches / len(vis_watches)) * 100 if len(vis_watches) else 0.0,
            description="Watch sessions surveyed for pollinator visits that have NO corresponding record in flowering survey.",
            impact="Indicates asynchronous sampling sessions or unsynchronized data entry.",
            remediation="Fall back to prior flowering observation (LOCF) for that plot-season.",
            offending_data=missing_watch_surveys[key_cols],
        )

        # Specific plant-interaction join coverage (Algorithm 6 fallback check)
        pos_df = vis_df[vis_df["PPI_STATUS"] == 1].copy()
        join_cols = ["PLOT_ID", "YEAR", "WATCH", "PLTSP_CODE"]
        flw_avail = flw_df[join_cols].drop_duplicates()

        merged_events = pos_df.merge(flw_avail, on=join_cols, how="left", indicator=True)
        unmatched_events = merged_events[merged_events["_merge"] == "left_only"]
        n_unmatched_events = len(unmatched_events)
        pct_unmatched = (n_unmatched_events / len(pos_df)) * 100 if len(pos_df) else 0.0

        res_events = DiscrepancyResult(
            category="Referential Integrity",
            check_name="interactions_lacking_same_session_flowering",
            severity="WARNING" if n_unmatched_events > 0 else "INFO",
            total_evaluated=len(pos_df),
            discrepancy_count=n_unmatched_events,
            discrepancy_pct=pct_unmatched,
            description=(
                f"Plant interaction records lacking same-session flowering record: {n_unmatched_events} "
                f"({pct_unmatched:.3f}%). Paper observed ~0.24% requiring LOCF fallback."
            ),
            impact="Requires last-observation-carried-forward (LOCF) imputation (Algorithm 6).",
            remediation="Confirm Algorithm 6 join_with_fallback resolves these without unintended defaults.",
            offending_data=unmatched_events[join_cols],
        )

        return res_watches, res_events

    @staticmethod
    def audit_spatial_hierarchy_consistency(vis_df: pd.DataFrame, flw_df: pd.DataFrame) -> DiscrepancyResult:
        """Audits consistency of PLOT_ID -> MEADOW -> COMPLEX mappings across both files."""
        hierarchy_cols = [c for c in ["PLOT_ID", "MEADOW", "COMPLEX"] if c in vis_df.columns and c in flw_df.columns]
        if len(hierarchy_cols) < 2:
            return DiscrepancyResult(
                category="Spatial Hierarchy",
                check_name="spatial_hierarchy_consistency",
                severity="INFO",
                total_evaluated=0,
                discrepancy_count=0,
                discrepancy_pct=0.0,
                description="Insufficient spatial hierarchy columns in raw files.",
                impact="None",
                remediation="Ensure PLOT_ID, MEADOW, and COMPLEX are present.",
            )

        vis_geo = vis_df[hierarchy_cols].drop_duplicates()
        flw_geo = flw_df[hierarchy_cols].drop_duplicates()

        merged = pd.concat([vis_geo, flw_geo]).drop_duplicates()
        # Check if any PLOT_ID maps to multiple MEADOWs or COMPLEXes
        plot_conflicts = merged.groupby("PLOT_ID").filter(lambda g: len(g) > 1)
        n_conflicts = plot_conflicts["PLOT_ID"].nunique()

        return DiscrepancyResult(
            category="Spatial Hierarchy",
            check_name="plot_to_meadow_complex_inconsistency",
            severity="ERROR" if n_conflicts > 0 else "INFO",
            total_evaluated=merged["PLOT_ID"].nunique(),
            discrepancy_count=n_conflicts,
            discrepancy_pct=(n_conflicts / merged["PLOT_ID"].nunique()) * 100 if merged["PLOT_ID"].nunique() else 0.0,
            description=f"Plots assigned to contradictory meadows or complexes across files: {n_conflicts}.",
            impact="Corrupts spatial partitioning and meadow-stratified detection probability analysis.",
            remediation="Standardize spatial lookup metadata table and enforce foreign keys.",
            offending_data=plot_conflicts if n_conflicts else None,
        )


# ==============================================================================
# 4. Granularity & Deduplication Detector
# ==============================================================================

class GranularityDetector:
    """Audits minute-level vs watch-session granularity discrepancies and duplicate records."""

    @staticmethod
    def audit_minute_level_duplicates(vis_df: pd.DataFrame) -> DiscrepancyResult:
        """Counts within-watch duplicate records that get collapsed during event stream assembly."""
        pos_df = vis_df[vis_df["PPI_STATUS"] == 1]
        dup_cols = ["PLOT_ID", "YEAR", "WATCH", "PLTSP_CODE", "VISSP_CODE"]
        if "MINUTE" in pos_df.columns:
            # Check how many interactions occur across multiple minutes within same watch
            watch_groups = pos_df.groupby(dup_cols).size()
            multi_minute = watch_groups[watch_groups > 1]
            n_multi = len(multi_minute)
            n_extra_rows = int((watch_groups - 1).clip(lower=0).sum())
            return DiscrepancyResult(
                category="Granularity & Aggregation",
                check_name="within_watch_minute_level_duplicates",
                severity="INFO",
                total_evaluated=len(watch_groups),
                discrepancy_count=n_multi,
                discrepancy_pct=(n_multi / len(watch_groups)) * 100 if len(watch_groups) else 0.0,
                description=(
                    f"Found {n_multi} plant-pollinator pairs with multiple minute-level visits in the same watch "
                    f"({n_extra_rows} extra rows collapsed into {len(watch_groups)} watch-level events)."
                ),
                impact="Explains the reduction from 61,025 raw positive rows to ~20,025 watch-level positive events.",
                remediation="Ensure GroupSum(..., value=NO_INT) is performed as prescribed in Algorithm 7.",
                metadata={"collapsed_extra_rows": n_extra_rows, "watch_level_events": len(watch_groups)},
            )
        return DiscrepancyResult(
            category="Granularity & Aggregation",
            check_name="within_watch_minute_level_duplicates",
            severity="INFO",
            total_evaluated=len(pos_df),
            discrepancy_count=0,
            discrepancy_pct=0.0,
            description="MINUTE column not present; skipping minute-level aggregation check.",
            impact="None",
            remediation="None",
        )

    @staticmethod
    def audit_negative_event_granularity(vis_df: pd.DataFrame) -> DiscrepancyResult:
        """Audits the known 13,917 (minute-level) vs 1,389 (watch-session) discrepancy in negatives."""
        neg_df = vis_df[vis_df["PPI_STATUS"] == 0]
        n_raw = len(neg_df)
        session_cols = ["PLOT_ID", "YEAR", "WATCH"]
        n_session = neg_df[session_cols].drop_duplicates().shape[0]

        return DiscrepancyResult(
            category="Granularity & Aggregation",
            check_name="structural_negative_granularity_discrepancy",
            severity="WARNING" if n_raw != n_session else "INFO",
            total_evaluated=n_raw,
            discrepancy_count=abs(n_raw - n_session),
            discrepancy_pct=(abs(n_raw - n_session) / n_raw) * 100 if n_raw else 0.0,
            description=(
                f"Negative events exist at two different granularities: {n_raw} raw minute-level rows "
                f"vs {n_session} deduplicated watch sessions. (Paper Table 1 cited 1,389; Section 2.2 cited 13,917)."
            ),
            impact="Users expecting 1,389 negative rows will encounter schema confusion when seeing 13,917 rows.",
            remediation="Export both granularities explicitly or document that 1,389 is deduplicated by (PLOT_ID, YEAR, WATCH).",
            metadata={"raw_minute_rows": n_raw, "deduplicated_watch_sessions": n_session},
        )


# ==============================================================================
# 5. Distributional & Parameter Boundary Detector
# ==============================================================================

class DistributionalDetector:
    """Audits parameter grid-capping risks and degenerate plant species."""

    @staticmethod
    def audit_lambda_grid_capping_risk(vis_df: pd.DataFrame, lam_ceiling: float = 80.0, p_hat: float = 0.15) -> DiscrepancyResult:
        """Flags high-abundance interaction groups whose raw count would pin lambda_hat near lam_ceiling."""
        pos_df = vis_df[vis_df["PPI_STATUS"] == 1].copy()
        if "NO_INT" not in pos_df.columns:
            return DiscrepancyResult(
                category="Distributional Boundary",
                check_name="lambda_grid_capping_risk",
                severity="INFO",
                total_evaluated=0,
                discrepancy_count=0,
                discrepancy_pct=0.0,
                description="NO_INT column missing.",
                impact="None",
                remediation="Ensure NO_INT is present.",
            )

        pos_df["NO_INT"] = pos_df["NO_INT"].fillna(1)
        group_cols = ["PLOT_ID", "YEAR", "PLTSP_CODE", "VISSP_CODE", "WATCH"]
        watch_totals = pos_df.groupby(group_cols)["NO_INT"].sum().reset_index()

        # Group-level maximum count
        group_keys = ["PLOT_ID", "YEAR", "PLTSP_CODE", "VISSP_CODE"]
        group_max = watch_totals.groupby(group_keys)["NO_INT"].max().reset_index()

        # If observed count / p_hat approaches or exceeds lam_ceiling, the group is at high risk of pinning
        pinned_risk = group_max[group_max["NO_INT"] >= (0.5 * lam_ceiling * p_hat)]
        n_risk = len(pinned_risk)
        pct_risk = (n_risk / len(group_max)) * 100 if len(group_max) else 0.0

        max_count = int(group_max["NO_INT"].max()) if len(group_max) else 0

        return DiscrepancyResult(
            category="Distributional Boundary",
            check_name="lambda_grid_capping_risk",
            severity="ERROR" if max_count > lam_ceiling else ("WARNING" if n_risk > 0 else "INFO"),
            total_evaluated=len(group_max),
            discrepancy_count=n_risk,
            discrepancy_pct=pct_risk,
            description=(
                f"Evaluated lambda-grid ceiling of {lam_ceiling}. Maximum single-watch count observed = {max_count}. "
                f"Groups at risk of boundary pinning: {n_risk} ({pct_risk:.2f}%)."
            ),
            impact="Silently pins posterior mean lambda_hat at grid ceiling, artificially squashing strongest edges.",
            remediation=f"Widen PipelineConfig.lam_grid upper ceiling to at least 5000 (currently evaluated: {lam_ceiling}).",
            offending_data=pinned_risk.sort_values("NO_INT", ascending=False),
            metadata={"max_single_watch_count": max_count, "evaluated_ceiling": lam_ceiling},
        )

    @staticmethod
    def audit_degenerate_plant_species(flw_df: pd.DataFrame) -> DiscrepancyResult:
        """Finds species with zero variance in flower counts across the entire survey."""
        if "NO_FLWS" not in flw_df.columns or "PLTSP_CODE" not in flw_df.columns:
            return DiscrepancyResult(
                category="Distributional Boundary",
                check_name="degenerate_flowering_species",
                severity="INFO",
                total_evaluated=0,
                discrepancy_count=0,
                discrepancy_pct=0.0,
                description="Columns missing.",
                impact="None",
                remediation="Ensure NO_FLWS and PLTSP_CODE exist.",
            )

        sp_span = flw_df.groupby("PLTSP_CODE")["NO_FLWS"].agg(lambda s: s.max() - s.min())
        degenerate = sp_span[sp_span < 1e-9]
        n_degenerate = len(degenerate)
        pct = (n_degenerate / len(sp_span)) * 100 if len(sp_span) else 0.0

        return DiscrepancyResult(
            category="Distributional Boundary",
            check_name="degenerate_flowering_species",
            severity="WARNING" if n_degenerate > 0 else "INFO",
            total_evaluated=len(sp_span),
            discrepancy_count=n_degenerate,
            discrepancy_pct=pct,
            description=f"Species with zero variance in flower counts: {n_degenerate} ({pct:.2f}%).",
            impact="Min-max normalization yields 0/0; triggers neutral 0.5 confidence fallback in Algorithm 5.",
            remediation="Confirm Algorithm 5 handles degenerate species via the 0.5 fallback condition.",
            metadata={"degenerate_species": list(degenerate.index)},
        )


# ==============================================================================
# 6. Temporal Continuity & Hiatus Detector
# ==============================================================================

class TemporalContinuityDetector:
    """Audits multi-year survey gaps, out-of-order records, and split leakage."""

    @staticmethod
    def audit_calendar_gaps(dates: pd.Series) -> DiscrepancyResult:
        valid_dates = dates.dropna()
        if len(valid_dates) == 0:
            return DiscrepancyResult(
                category="Temporal Continuity",
                check_name="calendar_year_gaps",
                severity="ERROR",
                total_evaluated=0,
                discrepancy_count=0,
                discrepancy_pct=0.0,
                description="No valid dates found to evaluate.",
                impact="Cannot construct temporal axis.",
                remediation="Provide valid date series.",
            )

        years = valid_dates.dt.year
        min_year, max_year = int(years.min()), int(years.max())
        expected_years = set(range(min_year, max_year + 1))
        observed_years = set(years.unique())

        missing_years = sorted(list(expected_years - observed_years))
        n_missing = len(missing_years)

        return DiscrepancyResult(
            category="Temporal Continuity",
            check_name="calendar_year_gaps",
            severity="WARNING" if n_missing > 0 else "INFO",
            total_evaluated=len(expected_years),
            discrepancy_count=n_missing,
            discrepancy_pct=(n_missing / len(expected_years)) * 100 if expected_years else 0.0,
            description=(
                f"Multi-year hiatus detected over survey horizon [{min_year}-{max_year}]. "
                f"Unsampled years: {missing_years}."
            ),
            impact="Naive calendar-based train/val/test splits produce empty partitions or invalid temporal decay.",
            remediation="Use cumulative event volume splitting (Algorithm 8) over continuous elapsed time tau.",
            metadata={"missing_years": missing_years, "observed_years": sorted(list(observed_years))},
        )
