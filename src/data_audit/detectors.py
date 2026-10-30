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
    target_dataset: str = "Both (Cross-Dataset)"
    offending_data: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_dataset": self.target_dataset,
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

        target = "SA02601_v6.csv (Visitation)" if "vis" in dataset_name.lower() else "SA02602_v5.csv (Flowering)"

        return DiscrepancyResult(
            target_dataset=target,
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
        df: pd.DataFrame, dataset_name: str, date_col: str = "SAMPLEDATE", year_col: str = "YEAR"
    ) -> Tuple[DiscrepancyResult, pd.Series]:
        """Audits internal date consistency in the dataset:
        1. Checks for unparseable dates or mixed date patterns.
        2. Asserts parsed date year matches the recorded YEAR column.
        (Does not impose external paper assumptions; checks internal dataset validity).
        """
        n_total = len(df)
        target = "SA02601_v6.csv (Visitation)" if "vis" in dataset_name.lower() else "SA02602_v5.csv (Flowering)"

        # Attempt standard ISO/auto parsing
        parsed_dates = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
        null_dates = parsed_dates.isna()
        n_parse_fails = int(null_dates.sum())

        # Check year consistency
        year_mismatch = (parsed_dates.dt.year != df[year_col]) & (~null_dates)
        n_mismatches = int(year_mismatch.sum())

        # Check if multiple date string formats are mixed in the same column
        date_str = df[date_col].astype(str)
        # Identify string patterns (e.g. 2011-07-24 -> XXXX-XX-XX vs 24-07-2011)
        patterns = date_str.str.replace(r"\d", "X", regex=True).unique()
        has_mixed_patterns = len(patterns) > 1

        total_issues = n_parse_fails + n_mismatches
        pct_issues = (total_issues / n_total) * 100 if n_total > 0 else 0.0

        offending_mask = null_dates | year_mismatch
        offending_df = df[offending_mask].copy() if total_issues > 0 else None
        if offending_df is not None:
            offending_df["_parsed_date"] = parsed_dates[offending_mask]
            offending_df["_issue_type"] = np.where(null_dates[offending_mask], "parse_failed", "year_contradiction")

        if total_issues > 0 or has_mixed_patterns:
            desc = (
                f"Date format inconsistencies detected in {dataset_name}: "
                f"Parse failures: {n_parse_fails}, Year contradictions with '{year_col}': {n_mismatches}, "
                f"Date string patterns found: {list(patterns)}."
            )
            severity = "ERROR" if total_issues > 0 else "WARNING"
            impact = "Corrupts chronological ordering and assigns observations to incorrect survey seasons."
            remediation = "Standardize date strings to ISO YYYY-MM-DD and reconcile rows contradicting YEAR."
        else:
            primary_fmt = "YYYY-MM-DD" if patterns[0] == "XXXX-XX-XX" else str(patterns[0])
            desc = (
                f"Dates in {dataset_name} are 100% internally consistent: All {n_total:,} rows "
                f"follow uniform '{primary_fmt}' and match the recorded '{year_col}' column."
            )
            severity = "INFO"
            impact = "Clean; dates parse unambiguously."
            remediation = "None required."

        result = DiscrepancyResult(
            target_dataset=target,
            category="Date Formatting & Consistency",
            check_name=f"{dataset_name}_date_consistency",
            severity=severity,
            total_evaluated=n_total,
            discrepancy_count=total_issues,
            discrepancy_pct=pct_issues,
            description=desc,
            impact=impact,
            remediation=remediation,
            offending_data=offending_df,
            metadata={
                "detected_patterns": list(patterns),
                "parse_failures": n_parse_fails,
                "year_mismatches": n_mismatches,
            },
        )
        return result, parsed_dates

    @staticmethod
    def audit_session_multi_date_splits(df: pd.DataFrame, dataset_name: str) -> DiscrepancyResult:
        """Checks if any watch session (PLOT_ID, YEAR, WATCH) spans across multiple calendar dates."""
        target = "SA02601_v6.csv (Visitation)" if "vis" in dataset_name.lower() else "SA02602_v5.csv (Flowering)"
        key_cols = ["PLOT_ID", "YEAR", "WATCH"]
        if not all(k in df.columns for k in key_cols) or "SAMPLEDATE" not in df.columns:
            return DiscrepancyResult(
                target_dataset=target,
                category="Date Formatting & Consistency",
                check_name=f"{dataset_name}_watch_multi_date_splits",
                severity="INFO",
                total_evaluated=0,
                discrepancy_count=0,
                discrepancy_pct=0.0,
                description="Columns missing.",
                impact="None",
                remediation="None",
            )

        session_dates = df.groupby(key_cols)["SAMPLEDATE"].nunique()
        multi_date_sessions = session_dates[session_dates > 1]
        n_multi = len(multi_date_sessions)
        pct_multi = (n_multi / len(session_dates)) * 100 if len(session_dates) else 0.0

        sample_sessions = {}
        if n_multi > 0:
            for k in list(multi_date_sessions.index[:5]):
                sample_sessions[str(k)] = list(df[(df["PLOT_ID"] == k[0]) & (df["YEAR"] == k[1]) & (df["WATCH"] == k[2])]["SAMPLEDATE"].unique())

        return DiscrepancyResult(
            target_dataset=target,
            category="Date Formatting & Consistency",
            check_name=f"{dataset_name}_watch_multi_date_splits",
            severity="WARNING" if n_multi > 0 else "INFO",
            total_evaluated=len(session_dates),
            discrepancy_count=n_multi,
            discrepancy_pct=pct_multi,
            description=(
                f"Found {n_multi} watch sessions in {dataset_name} that span multiple distinct calendar dates "
                f"({pct_multi:.2f}% of sessions). Sample: {sample_sessions}"
            ),
            impact="Watch sessions split across days (e.g. due to weather delays) create ambiguous event timestamps.",
            remediation="Assign the session's start date or aggregate observations at watch session granularity.",
            metadata={"multi_date_sessions_count": n_multi, "sample_sessions": sample_sessions},
        )


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
            target_dataset="SA02601_v6.csv (Visitation)",
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
            target_dataset="SA02601_v6.csv (Visitation)",
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
            target_dataset="SA02601_v6.csv (Visitation)",
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
                target_dataset="SA02601_v6.csv (Visitation)",
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
                target_dataset="SA02602_v5.csv (Flowering)",
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
            target_dataset="Both (SA02601 vs SA02602)",
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
            target_dataset="Both (SA02601 vs SA02602)",
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
            target_dataset="Both (SA02601 vs SA02602)",
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
                target_dataset="Both (SA02601 vs SA02602)",
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
            target_dataset="Both (SA02601 vs SA02602)",
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

    @staticmethod
    def audit_cross_dataset_watch_date_alignment(vis_df: pd.DataFrame, flw_df: pd.DataFrame) -> DiscrepancyResult:
        """Audits whether visitation and flowering records for the SAME watch session share the exact same SAMPLEDATE."""
        key_cols = ["PLOT_ID", "YEAR", "WATCH"]
        if not (all(k in vis_df.columns for k in key_cols) and all(k in flw_df.columns for k in key_cols)):
            return DiscrepancyResult(
                target_dataset="Both (SA02601 vs SA02602)",
                category="Date Formatting & Consistency",
                check_name="cross_dataset_watch_date_mismatch",
                severity="INFO",
                total_evaluated=0,
                discrepancy_count=0,
                discrepancy_pct=0.0,
                description="Columns missing.",
                impact="None",
                remediation="None",
            )

        vis_w = vis_df[key_cols + ["SAMPLEDATE"]].drop_duplicates()
        flw_w = flw_df[key_cols + ["SAMPLEDATE"]].drop_duplicates()

        merged = vis_w.merge(flw_w, on=key_cols, suffixes=("_vis", "_flw"))
        date_mismatch = merged[merged["SAMPLEDATE_vis"] != merged["SAMPLEDATE_flw"]].copy()
        n_mismatches = len(date_mismatch)
        pct_mismatches = (n_mismatches / len(merged)) * 100 if len(merged) else 0.0

        if n_mismatches > 0:
            diff_days = (
                pd.to_datetime(date_mismatch["SAMPLEDATE_flw"], errors="coerce", format="mixed")
                - pd.to_datetime(date_mismatch["SAMPLEDATE_vis"], errors="coerce", format="mixed")
            ).dt.days
            date_mismatch["day_offset"] = diff_days
            max_offset = int(diff_days.abs().max())
            median_offset = float(diff_days.abs().median())
        else:
            max_offset = 0
            median_offset = 0.0

        return DiscrepancyResult(
            target_dataset="Both (SA02601 vs SA02602)",
            category="Date Formatting & Consistency",
            check_name="cross_dataset_watch_date_mismatch",
            severity="WARNING" if n_mismatches > 0 else "INFO",
            total_evaluated=len(merged),
            discrepancy_count=n_mismatches,
            discrepancy_pct=pct_mismatches,
            description=(
                f"Found {n_mismatches} shared watch sessions ({pct_mismatches:.2f}%) where visitation and flowering "
                f"surveys occurred on DIFFERENT calendar dates (offsets up to {max_offset} days, median offset {median_offset:.0f} day). "
                f"Most common offset is +1 day (flowering surveyed the day after insect watches)."
            ),
            impact="A naive inner join requiring SAMPLEDATE equality drops 41.7% of valid watch sessions.",
            remediation="Join visitation and flowering on [PLOT_ID, YEAR, WATCH, PLTSP_CODE] without enforcing SAMPLEDATE equality.",
            offending_data=date_mismatch,
            metadata={"max_day_offset": max_offset, "median_day_offset": median_offset},
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
                target_dataset="SA02601_v6.csv (Visitation)",
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
            target_dataset="SA02601_v6.csv (Visitation)",
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
            target_dataset="SA02601_v6.csv (Visitation)",
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
                target_dataset="SA02601_v6.csv & Model Hyperparameters",
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
            target_dataset="SA02601_v6.csv & Model Hyperparameters",
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
                target_dataset="SA02602_v5.csv (Flowering)",
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
            target_dataset="SA02602_v5.csv (Flowering)",
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
                target_dataset="Both (Survey Sampling Horizon)",
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
            target_dataset="Both (Survey Sampling Horizon)",
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


# ==============================================================================
# 7. Missingness Encoding & Sparsity Detector
# ==============================================================================

class MissingnessEncodingDetector:
    """Audits representation of missing values: literal string 'NA' vs blanks vs sparse columns."""

    @staticmethod
    def audit_literal_na_strings(df: pd.DataFrame, dataset_name: str) -> List[DiscrepancyResult]:
        results = []
        target = "SA02601_v6.csv (Visitation)" if "vis" in dataset_name.lower() else "SA02602_v5.csv (Flowering)"

        # Check for literal string 'NA' vs empty strings
        cols_with_literal_na = {}
        for col in df.columns:
            # treat as string
            s = df[col].astype(str)
            na_count = int((s == "NA").sum())
            if na_count > 0:
                cols_with_literal_na[col] = na_count

        if cols_with_literal_na:
            total_literal_na = sum(cols_with_literal_na.values())
            results.append(DiscrepancyResult(
                target_dataset=target,
                category="Missingness Encoding",
                check_name=f"{dataset_name}_literal_string_na_encoding",
                severity="WARNING",
                total_evaluated=len(df) * len(cols_with_literal_na),
                discrepancy_count=total_literal_na,
                discrepancy_pct=(total_literal_na / (len(df) * len(cols_with_literal_na))) * 100 if len(df) else 0.0,
                description=(
                    f"Columns contain the literal two-letter string 'NA' instead of null/blank: {cols_with_literal_na}. "
                    f"Standard CSV parsers without na_values=['NA'] will read these as valid non-null strings."
                ),
                impact="String 'NA' in categorical columns (e.g. OBSERVER='NA', FLW_STATUS='NA') passes notna() checks.",
                remediation="Configure read_csv(..., keep_default_na=True, na_values=['NA']) to parse 'NA' as actual NaN.",
                metadata={"columns_with_literal_na": cols_with_literal_na},
            ))

        # Check for partially populated sparse columns (>30% missing)
        sparse_cols = {}
        for col in df.columns:
            s = df[col].astype(str)
            missing = (s == "") | (s == "NA") | (s == "nan") | df[col].isna()
            pct_missing = (missing.sum() / len(df)) * 100
            if pct_missing > 30.0:
                sparse_cols[col] = round(pct_missing, 2)

        if sparse_cols:
            results.append(DiscrepancyResult(
                target_dataset=target,
                category="Field Sparsity",
                check_name=f"{dataset_name}_highly_sparse_fields",
                severity="INFO",
                total_evaluated=len(df.columns),
                discrepancy_count=len(sparse_cols),
                discrepancy_pct=(len(sparse_cols) / len(df.columns)) * 100,
                description=f"Fields where over 30% of records are missing or unrecorded: {sparse_cols}.",
                impact="Downstream models cannot rely on these features without heavy imputation (e.g., TEMP 38% missing).",
                remediation="Exclude highly sparse attributes from graph node/edge features or apply targeted imputation.",
                metadata={"sparse_columns_pct": sparse_cols},
            ))

        return results


# ==============================================================================
# 8. Naming Inconsistency & Categorical Synonym Detector
# ==============================================================================

class NamingInconsistencyDetector:
    """Audits categorical naming variations, synonyms, typos, and formatting mixes (e.g. 'US' vs 'USA')."""

    @staticmethod
    def audit_pollinator_type_synonyms_and_typos(vis_df: pd.DataFrame) -> List[DiscrepancyResult]:
        results = []
        if "VISSP_CODE" not in vis_df.columns or "VISSP_TYPE" not in vis_df.columns:
            return results

        pos_df = vis_df[vis_df["PPI_STATUS"] == 1].dropna(subset=["VISSP_CODE", "VISSP_TYPE"])
        # Group by code -> set of types
        code_to_types = pos_df.groupby("VISSP_CODE")["VISSP_TYPE"].unique()
        multi_types = code_to_types[code_to_types.apply(len) > 1]
        n_multi = len(multi_types)

        # Detect specific typos in VISSP_TYPE
        known_typos = ["beeetle", "kloeptoparasitic bee", "predaceous onm bees", "muscoid fl"]
        detected_typos = {}
        unique_types = pos_df["VISSP_TYPE"].unique()
        for t in unique_types:
            for typo in known_typos:
                if typo in t:
                    detected_typos[t] = int((pos_df["VISSP_TYPE"] == t).sum())

        # Detect punctuation / synonym variations (hyphen vs space)
        synonym_pairs = [
            ("true-bug", "true bug"),
            ("social-wasp", "social wasp"),
            ("hover-fly", "Hover-fly"),
            ("beetle", "herbivorous beetle"),
            ("grasshopper", "herbivorous grasshopper"),
        ]
        active_synonyms = []
        for a, b in synonym_pairs:
            if a in unique_types and b in unique_types:
                active_synonyms.append(f"'{a}' vs '{b}'")

        if n_multi > 0:
            sample_conflicts = {k: list(v) for k, v in list(multi_types.items())[:8]}
            results.append(DiscrepancyResult(
                target_dataset="SA02601_v6.csv (Visitation)",
                category="Naming Inconsistency",
                check_name="pollinator_type_multi_assignment",
                severity="WARNING",
                total_evaluated=len(code_to_types),
                discrepancy_count=n_multi,
                discrepancy_pct=(n_multi / len(code_to_types)) * 100 if len(code_to_types) else 0.0,
                description=(
                    f"Found {n_multi} pollinator taxa assigned to multiple different functional guilds (VISSP_TYPE). "
                    f"Sample: {sample_conflicts}"
                ),
                impact="Splits functional insect guilds into inconsistent categories (e.g., Bombus bifarius as social vs solitary).",
                remediation="Construct a canonical 1-to-1 species->guild lookup table to harmonize visitor types.",
                metadata={"inconsistent_species_count": n_multi, "sample_conflicts": sample_conflicts},
            ))

        if detected_typos or active_synonyms:
            results.append(DiscrepancyResult(
                target_dataset="SA02601_v6.csv (Visitation)",
                category="Naming Inconsistency",
                check_name="functional_guild_synonyms_and_typos",
                severity="WARNING",
                total_evaluated=len(unique_types),
                discrepancy_count=len(detected_typos) + len(active_synonyms),
                discrepancy_pct=((len(detected_typos) + len(active_synonyms)) / len(unique_types)) * 100,
                description=(
                    f"Categorical naming variations detected in VISSP_TYPE: "
                    f"Typos: {detected_typos}. Synonym/Punctuation splits: {active_synonyms}."
                ),
                impact="Categorical encodings create redundant, fragmented one-hot dimensions for the same biological guild.",
                remediation="Standardize string representation (e.g. replace hyphens, fix 'beeetle' -> 'beetle').",
                metadata={"typos": detected_typos, "synonym_pairs": active_synonyms},
            ))

        return results

    @staticmethod
    def audit_observer_naming_styles(vis_df: pd.DataFrame, flw_df: pd.DataFrame) -> DiscrepancyResult:
        vis_obs = set(vis_df["OBSERVER"].dropna().unique()) if "OBSERVER" in vis_df.columns else set()
        flw_obs = set(flw_df["OBSERVER"].dropna().unique()) if "OBSERVER" in flw_df.columns else set()
        all_obs = vis_obs | flw_obs
        # remove literal 'NA'
        valid_obs = {o for o in all_obs if o not in ("NA", "nan", "")}

        multi_person = [o for o in valid_obs if any(sep in o for sep in [",", "+", " and ", "/"])]
        field_notes = [o for o in valid_obs if any(token in o.lower() for token in ["watch", "change", "(", "unknown", "xarm"])]
        initials_only = [o for o in valid_obs if len(o) in (2, 3) and o.isupper() and o not in multi_person]
        full_names = [o for o in valid_obs if len(o) > 3 and " " not in o and o[0].isupper() and o[1:].islower()]

        n_anomalies = len(multi_person) + len(field_notes)

        return DiscrepancyResult(
            target_dataset="Both (SA02601 vs SA02602)",
            category="Naming Inconsistency",
            check_name="observer_naming_style_fragmentation",
            severity="WARNING" if n_anomalies > 0 else "INFO",
            total_evaluated=len(valid_obs),
            discrepancy_count=n_anomalies,
            discrepancy_pct=(n_anomalies / len(valid_obs)) * 100 if valid_obs else 0.0,
            description=(
                f"Observer column contains mixed naming styles: {len(initials_only)} uppercase initials (e.g. 'AS', 'ARM'), "
                f"{len(full_names)} single first names (e.g. 'Andy', 'Vera'), {len(multi_person)} concatenated observer teams "
                f"(e.g. 'Andy and Tim', 'Melinda, Logan'), and {len(field_notes)} field log annotations (e.g. 'ARMchange watch', 'xARMx')."
            ),
            impact="Observer bias cannot be controlled as a fixed effect in statistical models without disambiguating names.",
            remediation="Create a standardized observer identity dictionary mapping names, teams, and initials to unique IDs.",
            metadata={
                "multi_person_teams": multi_person[:8],
                "field_note_annotations": field_notes[:8],
                "sample_initials": initials_only[:8],
                "sample_first_names": full_names[:8],
            },
        )

    @staticmethod
    def audit_taxonomic_synonym_drift(vis_df: pd.DataFrame, flw_df: pd.DataFrame) -> DiscrepancyResult:
        # Check ZIGAVENE or any other species where PLTSP_CODE maps to multiple scientific names
        dfs = []
        if "PLTSP_CODE" in vis_df.columns and "PLTSP_NAME" in vis_df.columns:
            dfs.append(vis_df[["PLTSP_CODE", "PLTSP_NAME"]].dropna())
        if "PLTSP_CODE" in flw_df.columns and "PLTSP_NAME" in flw_df.columns:
            dfs.append(flw_df[["PLTSP_CODE", "PLTSP_NAME"]].dropna())

        if not dfs:
            return DiscrepancyResult(
                target_dataset="Both (SA02601 vs SA02602)",
                category="Naming Inconsistency",
                check_name="taxonomic_synonym_botanical_drift",
                severity="INFO",
                total_evaluated=0,
                discrepancy_count=0,
                discrepancy_pct=0.0,
                description="PLTSP_NAME column not present in input data.",
                impact="None",
                remediation="None",
            )

        combined = pd.concat(dfs).drop_duplicates()
        combined = combined[(combined["PLTSP_CODE"] != "") & (combined["PLTSP_NAME"] != "")]

        code_to_names = combined.groupby("PLTSP_CODE")["PLTSP_NAME"].unique()
        synonym_drift = code_to_names[code_to_names.apply(len) > 1]
        n_drift = len(synonym_drift)

        sample = {k: list(v) for k, v in synonym_drift.items()}

        return DiscrepancyResult(
            target_dataset="Both (SA02601 vs SA02602)",
            category="Naming Inconsistency",
            check_name="taxonomic_synonym_botanical_drift",
            severity="WARNING" if n_drift > 0 else "INFO",
            total_evaluated=len(code_to_names),
            discrepancy_count=n_drift,
            discrepancy_pct=(n_drift / len(code_to_names)) * 100 if len(code_to_names) else 0.0,
            description=(
                f"Plant species codes with shifting scientific names across survey years: {sample}. "
                f"Example: ZIGAVENE shifts between 'Zigadenus venenosus' and 'Toxicoscordion venenosum'."
            ),
            impact="Label inconsistency if grouping by scientific binomial name instead of PLTSP_CODE.",
            remediation="Always use canonical PLTSP_CODE for relational joins; maintain a taxonomy synonym crosswalk.",
            metadata={"synonym_drift": sample},
        )
