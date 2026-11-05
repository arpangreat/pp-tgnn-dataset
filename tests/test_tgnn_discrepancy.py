"""
Unit and regression tests for the TGNN Discrepancy & Integrity Audit toolkit.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from tgnn_discrepancy.audit_temporal import measure_temporal_degeneracy
from tgnn_discrepancy.audit_splits import (
    detect_split_boundary_leakage,
    generate_leakage_free_splits,
)
from tgnn_discrepancy.audit_topology import analyze_node_and_edge_turnover
from tgnn_discrepancy.audit_negatives import (
    build_meadow_seasonal_pollinator_pool,
    materialize_grounded_structural_negatives,
)
from tgnn_discrepancy.audit_weights import analyze_weight_distribution


@pytest.fixture
def synthetic_events():
    """Small synthetic event stream with known temporal, split, and topological properties."""
    dates = ["2015-06-01", "2015-06-01", "2015-06-02", "2015-06-10", "2015-06-10", "2015-06-10"]
    taus = [0, 0, 1, 9, 9, 9]
    plants = ["PL1", "PL2", "PL1", "PL3", "PL1", "PL2"]
    polls = ["PO1", "PO2", "PO1", "PO3", "PO2", "PO1"]
    return pd.DataFrame({
        "tau": taus,
        "date": dates,
        "plant_id": plants,
        "pollinator_id": polls,
        "NO_INT": [1, 2, 1, 3, 1, 2],
        "confidence_posterior": [0.4, 0.5, 0.3, 0.8, 0.2, 0.6],
        "confidence_flowering": [0.5, 0.6, 0.4, 0.9, 0.3, 0.7],
        "c": [0.20, 0.30, 0.12, 0.72, 0.06, 0.42],
        "is_imputed_flowering": [False, False, False, False, False, True],
        "PLOT_ID": ["CPR01", "CPR01", "CPR02", "BGD01", "BGD01", "BGD02"],
        "YEAR": [2015, 2015, 2015, 2015, 2015, 2015],
        "WATCH": [1, 1, 1, 2, 2, 2],
    })


def test_measure_temporal_degeneracy(synthetic_events):
    res = measure_temporal_degeneracy(synthetic_events)
    assert res["n_events"] == 6
    assert res["n_unique_tau_days"] == 3
    # Consecutive events with dt=0: (index 1 is 0, index 4 is 9, index 5 is 9) -> 3 out of 6
    assert res["zero_delta_tau_events"] >= 3
    assert 0.0 < res["zero_delta_tau_fraction"] <= 1.0


def test_detect_split_boundary_leakage():
    # Construct splits where train and val share 2015-06-02
    train = pd.DataFrame({
        "tau": [0, 1], "date": ["2015-06-01", "2015-06-02"],
        "plant_id": ["PL1", "PL2"], "pollinator_id": ["PO1", "PO2"],
    })
    val = pd.DataFrame({
        "tau": [1, 5], "date": ["2015-06-02", "2015-06-06"],
        "plant_id": ["PL2", "PL3"], "pollinator_id": ["PO2", "PO3"],
    })
    test = pd.DataFrame({
        "tau": [10, 11], "date": ["2015-06-11", "2015-06-12"],
        "plant_id": ["PL3", "PL4"], "pollinator_id": ["PO3", "PO4"],
    })

    leakage = detect_split_boundary_leakage(train, val, test)
    assert leakage["has_leakage"] is True
    assert leakage["train_val_boundary"]["date_collision"] is True
    assert leakage["train_val_boundary"]["shared_pair_count"] == 1  # ('PL2', 'PO2')
    assert leakage["val_test_boundary"]["date_collision"] is False


def test_generate_leakage_free_splits(synthetic_events):
    train_c, val_c, test_c, info = generate_leakage_free_splits(
        synthetic_events, train_frac=0.5, val_frac=0.25
    )
    assert len(train_c) + len(val_c) + len(test_c) == len(synthetic_events)
    # Assert strict temporal separation without intra-day overlap
    assert train_c["tau"].max() < val_c["tau"].min()
    assert val_c["tau"].max() < test_c["tau"].min()


def test_analyze_node_and_edge_turnover():
    train = pd.DataFrame({
        "plant_id": ["P1", "P2"], "pollinator_id": ["Q1", "Q2"],
    })
    val = pd.DataFrame({
        "plant_id": ["P1", "P3"], "pollinator_id": ["Q1", "Q3"],
    })
    test = pd.DataFrame({
        "plant_id": ["P1", "P2", "P4"], "pollinator_id": ["Q1", "Q1", "Q4"],
    })

    res = analyze_node_and_edge_turnover(train, val, test)
    assert res["val"]["unseen_plants"] == 1  # P3
    assert res["val"]["unseen_polls"] == 1   # Q3
    assert res["test"]["recurrent_events"] == 1  # (P1, Q1)
    assert res["test"]["semi_novel_events"] == 1 # (P2, Q1)
    assert res["test"]["inductive_events"] == 1  # (P4, Q4)


def test_materialize_grounded_structural_negatives(synthetic_events):
    flw_conf = pd.DataFrame({
        "PLOT_ID": ["CPR01", "CPR01"],
        "YEAR": [2015, 2015],
        "WATCH": [1, 1],
        "PLTSP_CODE": ["PL1", "PL2"],
    })
    neg_events = pd.DataFrame({
        "PLOT_ID": ["CPR01"],
        "YEAR": [2015],
        "WATCH": [1],
        "date": ["2015-06-01"],
    })

    grounded = materialize_grounded_structural_negatives(
        neg_events, flw_conf, synthetic_events, max_samples_per_session=2, seed=42
    )
    assert len(grounded) > 0
    assert "plant_id" in grounded.columns
    assert "pollinator_id" in grounded.columns
    assert set(grounded["plant_id"]).issubset({"PL1", "PL2"})


def test_analyze_weight_distribution(synthetic_events):
    res = analyze_weight_distribution(synthetic_events)
    assert res["n_events"] == 6
    assert 0.0 <= res["c_summary"]["median"] <= 1.0
    assert res["imputed_flowering_count"] == 1
