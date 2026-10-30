"""
Unit tests for data_audit package and discrepancy detection algorithms.
"""

from pathlib import Path

import pandas as pd
import pytest

from data_audit.audit import DatasetAuditor, run_full_audit
from data_audit.detectors import (
    DistributionalDetector,
    GranularityDetector,
    MissingnessEncodingDetector,
    NamingInconsistencyDetector,
    ReferentialIntegrityDetector,
    SchemaAndDateDetector,
    SemanticStatusDetector,
    TemporalContinuityDetector,
)
from data_audit.synthetic_data import generate_audit_test_datasets


@pytest.fixture
def test_datasets(tmp_path: Path):
    syn_dir = tmp_path / "synthetic_audit"
    vis_path, flw_path = generate_audit_test_datasets(syn_dir, inject_discrepancies=True)
    return vis_path, flw_path


def test_schema_and_date_detector(test_datasets):
    vis_path, flw_path = test_datasets
    vis = pd.read_csv(vis_path)
    flw = pd.read_csv(flw_path)

    # Injected a 2099 date with YEAR=2011
    res, _ = SchemaAndDateDetector.audit_date_format(vis, "vis")
    assert res.severity == "ERROR"
    assert res.discrepancy_count >= 1
    assert res.offending_data is not None

    # Multi-date session split check
    split_res = SchemaAndDateDetector.audit_session_multi_date_splits(vis, "vis")
    assert split_res.category == "Date Formatting & Consistency"

    # Cross-dataset watch date alignment check
    align_res = ReferentialIntegrityDetector.audit_cross_dataset_watch_date_alignment(vis, flw)
    assert align_res.category == "Date Formatting & Consistency"


def test_semantic_status_detector(test_datasets):
    vis_path, flw_path = test_datasets
    vis = pd.read_csv(vis_path)
    flw = pd.read_csv(flw_path)

    vis_res = SemanticStatusDetector.audit_ppi_status_invariants(vis)
    domain_check = next(r for r in vis_res if r.check_name == "vis_ppi_status_valid_domain")
    assert domain_check.severity == "ERROR"
    assert domain_check.discrepancy_count >= 1  # status 99

    flw_res = SemanticStatusDetector.audit_flw_status_invariants(flw)
    flw_count_check = next(r for r in flw_res if r.check_name == "flw_negative_or_null_flower_count")
    assert flw_count_check.discrepancy_count >= 1  # count -10


def test_referential_integrity_detector(test_datasets):
    vis_path, flw_path = test_datasets
    vis = pd.read_csv(vis_path)
    flw = pd.read_csv(flw_path)

    species_res = ReferentialIntegrityDetector.audit_species_vocabulary_overlap(vis, flw)
    assert species_res.severity == "ERROR"
    assert "PL_ALIEN" in species_res.metadata["unmatched_species"]


def test_distributional_detector(test_datasets):
    vis_path, flw_path = test_datasets
    vis = pd.read_csv(vis_path)

    # Injected a count of 1500; evaluating against ceiling of 80 should flag an ERROR
    res = DistributionalDetector.audit_lambda_grid_capping_risk(vis, lam_ceiling=80.0, p_hat=0.15)
    assert res.severity == "ERROR"
    assert res.discrepancy_count >= 1
    assert res.metadata["max_single_watch_count"] >= 1500


def test_temporal_continuity_detector(test_datasets):
    vis_path, flw_path = test_datasets
    vis = pd.read_csv(vis_path)
    _, parsed_dates = SchemaAndDateDetector.audit_date_format(vis, "vis")

    res = TemporalContinuityDetector.audit_calendar_gaps(parsed_dates)
    assert res.severity == "WARNING"
    assert len(res.metadata["missing_years"]) > 0


def test_run_full_audit_produces_artifacts(test_datasets, tmp_path: Path):
    vis_path, flw_path = test_datasets
    out_dir = tmp_path / "audit_output"

    output = run_full_audit(vis_path, flw_path, out_dir, lam_ceiling_to_test=80.0)

    assert output["json_report"].exists()
    assert output["markdown_report"].exists()
    assert len(output["offending_csvs"]) > 0
    assert (out_dir / "offending_records").exists()


def test_missingness_and_naming_detectors(test_datasets):
    vis_path, flw_path = test_datasets
    vis = pd.read_csv(vis_path)
    flw = pd.read_csv(flw_path)

    missing_res = MissingnessEncodingDetector.audit_literal_na_strings(vis, "vis")
    assert isinstance(missing_res, list)

    obs_res = NamingInconsistencyDetector.audit_observer_naming_styles(vis, flw)
    assert obs_res.category == "Naming Inconsistency"
    assert obs_res.check_name == "observer_naming_style_fragmentation"
