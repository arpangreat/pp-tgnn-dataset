"""Tests for pp_tgnn.pipeline — validate Algorithms 1-8 against synthetic
data mirroring the real raw schema."""

import numpy as np
import pandas as pd
import pytest

from pp_tgnn.config import PipelineConfig
from pp_tgnn.pipeline import (
    build_event_stream,
    build_replicate_groups,
    chronological_split,
    fit_detection_probability,
    flowering_confidence,
    harmonize_raw_data,
    join_with_fallback,
    log_marginal_over_lamgrid,
    partition_interactions,
    posterior_confidence,
    run_pipeline,
)


@pytest.fixture
def cfg() -> PipelineConfig:
    return PipelineConfig(poll_col="VISSP_CODE")


# --------------------------------------------------------------------
# Algorithm 1 — HarmonizeRawData
# --------------------------------------------------------------------

def test_harmonize_raw_data_parses_both_date_formats(synthetic_vis_path, synthetic_flw_path, cfg):
    vis, flw = harmonize_raw_data(synthetic_vis_path, synthetic_flw_path, cfg)
    assert "date" in vis.columns and "date" in flw.columns
    assert (vis["date"].dt.year == vis[cfg.year_col]).all()
    assert (flw["date"].dt.year == flw[cfg.year_col]).all()


def test_harmonize_raw_data_catches_date_format_mismatch(synthetic_vis_path, synthetic_flw_path, cfg):
    """Swapping the two date formats should trip the year-consistency assertion —
    this is the exact silent-corruption failure mode documented in the paper."""
    bad_cfg = PipelineConfig(
        poll_col="VISSP_CODE",
        vis_date_fmt="%Y-%m-%d",  # wrong on purpose (vis is actually %d-%m-%Y)
    )
    with pytest.raises((AssertionError, ValueError)):
        harmonize_raw_data(synthetic_vis_path, synthetic_flw_path, bad_cfg)


# --------------------------------------------------------------------
# Algorithm 2 — PartitionInteractions
# --------------------------------------------------------------------

def test_partition_interactions_splits_correctly(synthetic_vis_path, synthetic_flw_path, cfg):
    vis, _ = harmonize_raw_data(synthetic_vis_path, synthetic_flw_path, cfg)
    e_pos, e_neg = partition_interactions(vis, cfg)

    assert len(e_pos) + len(e_neg) == len(vis)
    assert e_pos[cfg.plant_col].notna().all()
    assert e_neg[cfg.plant_col].isna().all()
    assert not e_pos[cfg.no_int_col].isna().any()  # filled with 1


# --------------------------------------------------------------------
# Algorithm 3 — BuildReplicateGroups
# --------------------------------------------------------------------

def test_build_replicate_groups_replicate_count_matches_watches(synthetic_vis_path, synthetic_flw_path, cfg):
    vis, _ = harmonize_raw_data(synthetic_vis_path, synthetic_flw_path, cfg)
    e_pos, _ = partition_interactions(vis, cfg)
    groups = build_replicate_groups(e_pos, vis, cfg)

    assert len(groups) > 0
    for g in groups:
        expected_K = len(vis[(vis[cfg.plot_col] == g.plot) & (vis[cfg.year_col] == g.year)][cfg.watch_col].unique())
        assert g.K == expected_K
        assert len(g.y) == g.K
        assert (g.y >= 0).all()


# --------------------------------------------------------------------
# Listing 1 + Algorithm 4 — Detection-bias correction
# --------------------------------------------------------------------

def test_log_marginal_over_lamgrid_shape_and_finiteness():
    y = np.array([0.0, 1.0, 2.0])
    lam_grid = np.geomspace(0.01, 80, 50)
    ell = log_marginal_over_lamgrid(y, lam_grid, p=0.15)
    assert ell.shape == lam_grid.shape
    assert np.isfinite(ell).all()


def test_fit_detection_probability_returns_valid_phat_and_lambda(synthetic_vis_path, synthetic_flw_path, cfg):
    vis, _ = harmonize_raw_data(synthetic_vis_path, synthetic_flw_path, cfg)
    e_pos, _ = partition_interactions(vis, cfg)
    groups = build_replicate_groups(e_pos, vis, cfg)
    p_hat, lambda_hat = fit_detection_probability(groups, cfg)

    assert cfg.p_grid.min() <= p_hat <= cfg.p_grid.max()
    assert len(lambda_hat) == len(groups)
    assert all(v >= 0 for v in lambda_hat.values())


def test_posterior_confidence_in_unit_range():
    lambda_hat = {("p1", 2011, "PL0", "PO0"): 0.1, ("p1", 2011, "PL0", "PO1"): 5.0}
    conf = posterior_confidence(lambda_hat)
    assert all(0.0 <= v <= 1.0 for v in conf.values())


# --------------------------------------------------------------------
# Algorithm 5 — FloweringConfidence
# --------------------------------------------------------------------

def test_flowering_confidence_in_unit_range(synthetic_vis_path, synthetic_flw_path, cfg):
    _, flw = harmonize_raw_data(synthetic_vis_path, synthetic_flw_path, cfg)
    flw_conf = flowering_confidence(flw, cfg)
    assert flw_conf["confidence_flowering"].between(0, 1).all()


def test_flowering_confidence_degenerate_species_gets_neutral_value(cfg):
    flw = pd.DataFrame({
        cfg.plant_col: ["PLX"] * 3,
        cfg.no_flws_col: [5, 5, 5],  # no variation -> degenerate
        cfg.plot_col: ["P0"] * 3,
        cfg.year_col: [2011] * 3,
        cfg.watch_col: [1, 2, 3],
        "date": pd.to_datetime(["2011-06-01", "2011-06-11", "2011-06-21"]),
    })
    flw_conf = flowering_confidence(flw, cfg)
    assert (flw_conf["confidence_flowering"] == 0.5).all()


# --------------------------------------------------------------------
# Algorithm 8 — ChronologicalSplit
# --------------------------------------------------------------------

def test_chronological_split_is_leakage_safe():
    events = pd.DataFrame({"tau": np.arange(100), "value": np.arange(100)})
    train, val, test = chronological_split(events, PipelineConfig())

    assert len(train) + len(val) + len(test) == len(events)
    assert train["tau"].max() < val["tau"].min()
    assert val["tau"].max() < test["tau"].min()
    assert len(train) == 80 and len(val) == 10 and len(test) == 10


def test_chronological_split_robust_to_sampling_gap():
    """Mirrors the paper's 2019-2020 gap: a chunk of tau values is simply
    absent. The cumulative-volume split must not produce an empty partition."""
    tau_before_gap = np.arange(0, 50)
    tau_after_gap = np.arange(1000, 1050)  # large jump, no events in between
    events = pd.DataFrame({"tau": np.concatenate([tau_before_gap, tau_after_gap])})

    train, val, test = chronological_split(events, PipelineConfig())
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    assert train["tau"].max() <= val["tau"].min()
    assert val["tau"].max() <= test["tau"].min()


# --------------------------------------------------------------------
# End-to-end pipeline
# --------------------------------------------------------------------

def test_run_pipeline_end_to_end(synthetic_vis_path, synthetic_flw_path, tmp_path, cfg):
    outputs = run_pipeline(synthetic_vis_path, synthetic_flw_path, tmp_path, cfg)

    assert set(outputs.keys()) >= {"events", "negatives", "train", "val", "test"}
    events = outputs["events"]

    # Schema checks
    for col in ["tau", "date", "plant_id", "pollinator_id", "c", "is_imputed_flowering"]:
        assert col in events.columns

    # Value-range checks
    assert events["c"].between(0, 1).all()
    assert (events["tau"] >= 0).all()

    # Split checks
    n = len(events)
    assert len(outputs["train"]) + len(outputs["val"]) + len(outputs["test"]) == n

    # Output files actually written
    for fname in [
        "plant_pollinator_events.csv", "plant_pollinator_negative_events.csv",
        "events_train.csv", "events_val.csv", "events_test.csv",
        "lambda_hat.csv", "flw_conf.csv",
    ]:
        assert (tmp_path / fname).exists(), f"Missing output file: {fname}"

    # Re-read written CSV round-trips cleanly
    reread = pd.read_csv(tmp_path / "plant_pollinator_events.csv")
    assert len(reread) == len(events)
