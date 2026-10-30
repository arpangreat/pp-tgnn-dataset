"""
DatasetAuditor orchestrator: runs all discrepancy detectors across both datasets and outputs reports.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .detectors import (
    DiscrepancyResult,
    DistributionalDetector,
    GranularityDetector,
    MissingnessEncodingDetector,
    NamingInconsistencyDetector,
    ReferentialIntegrityDetector,
    SchemaAndDateDetector,
    SemanticStatusDetector,
    TemporalContinuityDetector,
)
from .reporter import AuditReporter

log = logging.getLogger("data_audit")


class DatasetAuditor:
    """Coordinates discrepancy audits across SA02601 and SA02602."""

    def __init__(
        self,
        vis_df: pd.DataFrame,
        flw_df: pd.DataFrame,
        lam_ceiling_to_test: float = 80.0,
    ):
        self.vis = vis_df.copy()
        self.flw = flw_df.copy()
        self.lam_ceiling_to_test = lam_ceiling_to_test

    def run_all_checks(self) -> List[DiscrepancyResult]:
        """Runs the complete battery of discrepancy checks across both datasets."""
        results: List[DiscrepancyResult] = []

        log.info("Starting comprehensive dataset discrepancy audit...")

        # 1. Schema & Required Columns
        results.append(
            SchemaAndDateDetector.audit_columns(
                self.vis, "SA02601_visitation", SchemaAndDateDetector.EXPECTED_VIS_COLS
            )
        )
        results.append(
            SchemaAndDateDetector.audit_columns(
                self.flw, "SA02602_flowering", SchemaAndDateDetector.EXPECTED_FLW_COLS
            )
        )

        # 2. Date Formatting & Year Invariants (Empirical - no false paper format assumptions)
        vis_date_res, vis_dates = SchemaAndDateDetector.audit_date_format(
            self.vis, "SA02601_visitation"
        )
        results.append(vis_date_res)

        flw_date_res, flw_dates = SchemaAndDateDetector.audit_date_format(
            self.flw, "SA02602_flowering"
        )
        results.append(flw_date_res)

        # Multi-date session splits within datasets
        results.append(
            SchemaAndDateDetector.audit_session_multi_date_splits(
                self.vis, "SA02601_visitation"
            )
        )
        results.append(
            SchemaAndDateDetector.audit_session_multi_date_splits(
                self.flw, "SA02602_flowering"
            )
        )

        # 3. Semantic & Status Invariants
        results.extend(SemanticStatusDetector.audit_ppi_status_invariants(self.vis))
        results.extend(SemanticStatusDetector.audit_flw_status_invariants(self.flw))

        # 4. Referential Integrity & Relational Overlap
        results.append(ReferentialIntegrityDetector.audit_species_vocabulary_overlap(self.vis, self.flw))
        watch_cov, event_cov = ReferentialIntegrityDetector.audit_watch_level_join_coverage(self.vis, self.flw)
        results.append(watch_cov)
        results.append(event_cov)
        results.append(ReferentialIntegrityDetector.audit_spatial_hierarchy_consistency(self.vis, self.flw))
        results.append(ReferentialIntegrityDetector.audit_cross_dataset_watch_date_alignment(self.vis, self.flw))

        # 5. Granularity & Deduplication
        results.append(GranularityDetector.audit_minute_level_duplicates(self.vis))
        results.append(GranularityDetector.audit_negative_event_granularity(self.vis))

        # 6. Distributional & Parameter Boundaries
        results.append(
            DistributionalDetector.audit_lambda_grid_capping_risk(
                self.vis, lam_ceiling=self.lam_ceiling_to_test
            )
        )
        results.append(DistributionalDetector.audit_degenerate_plant_species(self.flw))

        # 7. Temporal Continuity & Sampling Gaps
        results.append(TemporalContinuityDetector.audit_calendar_gaps(vis_dates))

        # 8. Missingness Representation & Field Sparsity (String 'NA' vs blanks)
        results.extend(MissingnessEncodingDetector.audit_literal_na_strings(self.vis, "SA02601_visitation"))
        results.extend(MissingnessEncodingDetector.audit_literal_na_strings(self.flw, "SA02602_flowering"))

        # 9. Naming Inconsistencies & Categorical Synonyms
        results.extend(NamingInconsistencyDetector.audit_pollinator_type_synonyms_and_typos(self.vis))
        results.append(NamingInconsistencyDetector.audit_observer_naming_styles(self.vis, self.flw))
        results.append(NamingInconsistencyDetector.audit_taxonomic_synonym_drift(self.vis, self.flw))

        return results


def run_full_audit(
    vis_path: Path | str,
    flw_path: Path | str,
    out_dir: Path | str,
    lam_ceiling_to_test: float = 80.0,
) -> Dict[str, Any]:
    """Loads raw datasets, executes all discrepancy detectors, and exports full report artifacts."""
    vis_path, flw_path, out_dir = Path(vis_path), Path(flw_path), Path(out_dir)

    log.info("Loading raw visitation: %s", vis_path)
    vis = pd.read_csv(vis_path, low_memory=False)

    log.info("Loading raw flowering: %s", flw_path)
    flw = pd.read_csv(flw_path, low_memory=False)

    auditor = DatasetAuditor(
        vis, flw, lam_ceiling_to_test=lam_ceiling_to_test
    )
    findings = auditor.run_all_checks()

    reporter = AuditReporter(out_dir)
    reporter.log_results_to_console(findings)
    json_path = reporter.export_json(findings)
    offending_csvs = reporter.export_offending_csvs(findings)
    md_path = reporter.generate_markdown_report(findings)

    return {
        "findings": findings,
        "json_report": json_path,
        "markdown_report": md_path,
        "offending_csvs": offending_csvs,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit and log data discrepancies in SA02601 and SA02602 datasets.")
    parser.add_argument("--vis", type=Path, default=Path("data/raw/SA02601_v6.csv"), help="Path to raw visitation CSV")
    parser.add_argument("--flw", type=Path, default=Path("data/raw/SA02602_v5.csv"), help="Path to raw flowering CSV")
    parser.add_argument("--outdir", type=Path, default=Path("data_discrepancy_audit"), help="Output directory for reports")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic test data with deliberate discrepancies")
    parser.add_argument("--lam-ceiling", type=float, default=80.0, help="Candidate lambda ceiling to audit for capping risk")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.synthetic:
        from .synthetic_data import generate_audit_test_datasets
        syn_dir = args.outdir / "synthetic_input"
        log.info("Generating synthetic test dataset at %s with injected discrepancies...", syn_dir)
        vis_path, flw_path = generate_audit_test_datasets(syn_dir, inject_discrepancies=True)
    else:
        vis_path, flw_path = args.vis, args.flw
        if not vis_path.exists() or not flw_path.exists():
            log.warning("Raw files not found at %s or %s. Falling back to synthetic demonstration mode.", vis_path, flw_path)
            from .synthetic_data import generate_audit_test_datasets
            syn_dir = args.outdir / "synthetic_input"
            vis_path, flw_path = generate_audit_test_datasets(syn_dir, inject_discrepancies=True)

    run_full_audit(vis_path, flw_path, args.outdir, lam_ceiling_to_test=args.lam_ceiling)


if __name__ == "__main__":
    main()
