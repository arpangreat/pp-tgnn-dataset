"""
Audit train/val/test chronological partition boundaries for intra-day event splitting,
identical-pair leakage, and generate strictly leakage-free calendar-date splits.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("tgnn_discrepancy.splits")


def detect_split_boundary_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Detects date collisions and active interaction pair leakage across train/val and val/test boundaries."""
    train_dates = set(train_df["date"])
    val_dates = set(val_df["date"])
    test_dates = set(test_df["date"])

    train_val_shared_dates = sorted(list(train_dates.intersection(val_dates)))
    val_test_shared_dates = sorted(list(val_dates.intersection(test_dates)))

    train_tau_max = int(train_df["tau"].max())
    val_tau_min = int(val_df["tau"].min())
    val_tau_max = int(val_df["tau"].max())
    test_tau_min = int(test_df["tau"].min())

    # Check pair overlap on shared dates
    tv_leakage_pairs: Set[Tuple[str, str]] = set()
    tv_events_train = 0
    tv_events_val = 0
    if train_val_shared_dates:
        t_sub = train_df[train_df["date"].isin(train_val_shared_dates)]
        v_sub = val_df[val_df["date"].isin(val_val_shared_dates if False else train_val_shared_dates)]
        tv_events_train = len(t_sub)
        tv_events_val = len(v_sub)
        pairs_t = set(zip(t_sub["plant_id"], t_sub["pollinator_id"]))
        pairs_v = set(zip(v_sub["plant_id"], v_sub["pollinator_id"]))
        tv_leakage_pairs = pairs_t.intersection(pairs_v)

    vt_leakage_pairs: Set[Tuple[str, str]] = set()
    vt_events_val = 0
    vt_events_test = 0
    if val_test_shared_dates:
        v_sub = val_df[val_df["date"].isin(val_test_shared_dates)]
        t_sub = test_df[test_df["date"].isin(val_test_shared_dates)]
        vt_events_val = len(v_sub)
        vt_events_test = len(t_sub)
        pairs_v = set(zip(v_sub["plant_id"], v_sub["pollinator_id"]))
        pairs_t = set(zip(t_sub["plant_id"], t_sub["pollinator_id"]))
        vt_leakage_pairs = pairs_v.intersection(pairs_t)

    has_leakage = bool(len(train_val_shared_dates) > 0 or len(val_test_shared_dates) > 0)

    return {
        "has_leakage": has_leakage,
        "train_tau_range": [int(train_df["tau"].min()), train_tau_max],
        "val_tau_range": [val_tau_min, val_tau_max],
        "test_tau_range": [test_tau_min, int(test_df["tau"].max())],
        "train_val_boundary": {
            "date_collision": len(train_val_shared_dates) > 0,
            "shared_dates": train_val_shared_dates,
            "train_tau_max": train_tau_max,
            "val_tau_min": val_tau_min,
            "train_events_on_shared_dates": tv_events_train,
            "val_events_on_shared_dates": tv_events_val,
            "shared_pair_count": len(tv_leakage_pairs),
            "sample_shared_pairs": [f"{p}->{q}" for p, q in list(tv_leakage_pairs)[:10]],
        },
        "val_test_boundary": {
            "date_collision": len(val_test_shared_dates) > 0,
            "shared_dates": val_test_shared_dates,
            "val_tau_max": val_tau_max,
            "test_tau_min": test_tau_min,
            "val_events_on_shared_dates": vt_events_val,
            "test_events_on_shared_dates": vt_events_test,
            "shared_pair_count": len(vt_leakage_pairs),
            "sample_shared_pairs": [f"{p}->{q}" for p, q in list(vt_leakage_pairs)[:10]],
        },
    }


def generate_leakage_free_splits(
    events_df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Partitions events by whole calendar dates so that tau_train < tau_val < tau_test strictly.

    Guarantees that no calendar day is split across partitions, eliminating intra-day leakage.
    """
    df = events_df.sort_values(["tau", "date"]).reset_index(drop=True)
    total_events = len(df)
    target_train_n = int(round(total_events * train_frac))
    target_val_n = int(round(total_events * val_frac))

    # Daily event counts sorted chronologically
    daily = (
        df.groupby(["tau", "date"])
        .size()
        .reset_index(name="count")
        .sort_values("tau")
        .reset_index(drop=True)
    )
    daily["cum_count"] = daily["count"].cumsum()
    n_days = len(daily)
    if n_days < 3:
        raise ValueError(f"Need at least 3 distinct calendar dates to split train/val/test, got {n_days}")

    # Find date cutoff for train (must leave at least 1 day for val and 1 day for test)
    candidate_train_indices = np.arange(0, n_days - 2)
    diffs_train = (daily.loc[candidate_train_indices, "cum_count"] - target_train_n).abs()
    best_train_idx = int(candidate_train_indices[diffs_train.argmin()])
    train_cutoff_tau = daily.loc[best_train_idx, "tau"]

    # Find date cutoff for val (must leave at least 1 day for test)
    target_train_val_n = target_train_n + target_val_n
    candidate_val_indices = np.arange(best_train_idx + 1, n_days - 1)
    diffs_val = (daily.loc[candidate_val_indices, "cum_count"] - target_train_val_n).abs()
    best_val_idx = int(candidate_val_indices[diffs_val.argmin()])
    val_cutoff_tau = daily.loc[best_val_idx, "tau"]

    train_mask = df["tau"] <= train_cutoff_tau
    val_mask = (df["tau"] > train_cutoff_tau) & (df["tau"] <= val_cutoff_tau)
    test_mask = df["tau"] > val_cutoff_tau

    train_clean = df[train_mask].reset_index(drop=True)
    val_clean = df[val_mask].reset_index(drop=True)
    test_clean = df[test_mask].reset_index(drop=True)

    # Verification: assert strict inequality
    assert train_clean["tau"].max() < val_clean["tau"].min(), "Train/val strict separation failed!"
    assert val_clean["tau"].max() < test_clean["tau"].min(), "Val/test strict separation failed!"

    split_info = {
        "train_n": len(train_clean),
        "train_pct": len(train_clean) / total_events,
        "train_date_range": [str(train_clean["date"].min()), str(train_clean["date"].max())],
        "train_tau_max": int(train_clean["tau"].max()),
        "val_n": len(val_clean),
        "val_pct": len(val_clean) / total_events,
        "val_date_range": [str(val_clean["date"].min()), str(val_clean["date"].max())],
        "val_tau_min": int(val_clean["tau"].min()),
        "val_tau_max": int(val_clean["tau"].max()),
        "test_n": len(test_clean),
        "test_pct": len(test_clean) / total_events,
        "test_date_range": [str(test_clean["date"].min()), str(test_clean["date"].max())],
        "test_tau_min": int(test_clean["tau"].min()),
    }

    return train_clean, val_clean, test_clean, split_info
