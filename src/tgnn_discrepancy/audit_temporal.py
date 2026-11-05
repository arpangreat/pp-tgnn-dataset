"""
Audit temporal dynamics, timestamp degeneracy (delta_tau = 0),
intra-day permutation sensitivity, and multi-year hiatus impact.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from pp_tgnn.model import TGNNLite
from pp_tgnn.train import encode_events, negative_sample

log = logging.getLogger("tgnn_discrepancy.temporal")


def measure_temporal_degeneracy(events_df: pd.DataFrame) -> Dict[str, Any]:
    """Quantifies zero-interval timestamp collapse (events sharing identical integer tau)."""
    df = events_df.sort_values("tau").reset_index(drop=True)
    n_events = len(df)
    unique_tau = df["tau"].nunique()
    tau_counts = df["tau"].value_counts()

    delta_tau = df["tau"].diff().fillna(0)
    zero_dt_count = int((delta_tau == 0).sum())
    zero_dt_frac = float(zero_dt_count / max(n_events, 1))

    return {
        "n_events": n_events,
        "n_unique_tau_days": unique_tau,
        "mean_events_per_active_day": float(n_events / max(unique_tau, 1)),
        "median_events_per_active_day": float(tau_counts.median()),
        "max_events_on_single_day": int(tau_counts.max()),
        "zero_delta_tau_events": zero_dt_count,
        "zero_delta_tau_fraction": zero_dt_frac,
        "inter_arrival_days_quantiles": {
            "25%": float(delta_tau.quantile(0.25)),
            "50%": float(delta_tau.quantile(0.50)),
            "75%": float(delta_tau.quantile(0.75)),
            "90%": float(delta_tau.quantile(0.90)),
            "99%": float(delta_tau.quantile(0.99)),
            "max": float(delta_tau.max()),
        },
    }


def evaluate_intraday_permutation_sensitivity(
    model: TGNNLite,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    plant2idx: Dict[str, int],
    poll2idx: Dict[str, int],
    device: torch.device,
    seeds: Optional[List[int]] = None,
    batch_size: int = 200,
) -> Dict[str, Any]:
    """Tests TGNN stability under permutations of events sharing the exact same tau timestamp."""
    if seeds is None:
        seeds = [42, 100, 2024, 777, 999]

    n_plants = len(plant2idx)
    n_polls = len(poll2idx)

    train_enc = encode_events(train_df, plant2idx, poll2idx)
    val_enc = encode_events(val_df, plant2idx, poll2idx)
    test_enc = encode_events(test_df, plant2idx, poll2idx)

    def _eval_sequence(eval_df: pd.DataFrame) -> Tuple[float, float]:
        # Reset and warm up memory through train + val
        model.reset_memory()
        with torch.no_grad():
            for df in [train_enc, val_enc]:
                for start in range(0, len(df), batch_size):
                    b = df.iloc[start : start + batch_size]
                    model.update(
                        torch.tensor(b["u"].values, dtype=torch.long, device=device),
                        torch.tensor(b["v"].values, dtype=torch.long, device=device),
                        torch.tensor(b["tau"].values, dtype=torch.float, device=device),
                        torch.tensor(b["c"].values, dtype=torch.float, device=device),
                    )

            all_scores, all_labels = [], []
            for start in range(0, len(eval_df), batch_size):
                b = eval_df.iloc[start : start + batch_size]
                u = torch.tensor(b["u"].values, dtype=torch.long, device=device)
                v = torch.tensor(b["v"].values, dtype=torch.long, device=device)
                t = torch.tensor(b["tau"].values, dtype=torch.float, device=device)
                c = torch.tensor(b["c"].values, dtype=torch.float, device=device)

                neg_v = negative_sample(u, n_plants, n_polls, device)
                pos_s = torch.sigmoid(model.score(u, v, t)).cpu().numpy()
                neg_s = torch.sigmoid(model.score(u, neg_v, t)).cpu().numpy()

                all_scores.extend(pos_s)
                all_scores.extend(neg_s)
                all_labels.extend([1.0] * len(pos_s))
                all_labels.extend([0.0] * len(neg_s))

                model.update(u, v, t, c)

        ap = float(average_precision_score(all_labels, all_scores))
        auc = float(roc_auc_score(all_labels, all_scores))
        return ap, auc

    orig_ap, orig_auc = _eval_sequence(test_enc)
    results = [{"seed": "original", "ap": orig_ap, "auc": orig_auc}]

    perm_aps, perm_aucs = [], []
    for s in seeds:
        shuffled_test = (
            test_enc.sample(frac=1.0, random_state=s)
            .sort_values("tau", kind="stable")
            .reset_index(drop=True)
        )
        s_ap, s_auc = _eval_sequence(shuffled_test)
        results.append({"seed": s, "ap": s_ap, "auc": s_auc})
        perm_aps.append(s_ap)
        perm_aucs.append(s_auc)

    ap_range = max(perm_aps) - min(perm_aps)
    auc_range = max(perm_aucs) - min(perm_aucs)

    return {
        "original_ap": orig_ap,
        "original_auc": orig_auc,
        "permutation_runs": results,
        "ap_mean": float(np.mean(perm_aps)),
        "ap_std": float(np.std(perm_aps)),
        "ap_min": float(min(perm_aps)),
        "ap_max": float(max(perm_aps)),
        "ap_spread": float(ap_range),
        "auc_mean": float(np.mean(perm_aucs)),
        "auc_std": float(np.std(perm_aucs)),
        "auc_spread": float(auc_range),
    }


def evaluate_hiatus_impact(
    model: TGNNLite,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    plant2idx: Dict[str, int],
    poll2idx: Dict[str, int],
    device: torch.device,
    hiatus_years: Tuple[int, int] = (2019, 2020),
    batch_size: int = 200,
) -> Dict[str, Any]:
    """Measures TGNN performance collapse across multi-year unobserved hiatus."""
    n_plants = len(plant2idx)
    n_polls = len(poll2idx)

    train_enc = encode_events(train_df, plant2idx, poll2idx)
    val_enc = encode_events(val_df, plant2idx, poll2idx)
    test_enc = encode_events(test_df, plant2idx, poll2idx).sort_values("tau").reset_index(drop=True)

    # Warm up memory
    model.reset_memory()
    with torch.no_grad():
        for df in [train_enc, val_enc]:
            for start in range(0, len(df), batch_size):
                b = df.iloc[start : start + batch_size]
                model.update(
                    torch.tensor(b["u"].values, dtype=torch.long, device=device),
                    torch.tensor(b["v"].values, dtype=torch.long, device=device),
                    torch.tensor(b["tau"].values, dtype=torch.float, device=device),
                    torch.tensor(b["c"].values, dtype=torch.float, device=device),
                )

        pre_scores, pre_labels = [], []
        post_scores, post_labels = [], []

        for start in range(0, len(test_enc), batch_size):
            b = test_enc.iloc[start : start + batch_size]
            u = torch.tensor(b["u"].values, dtype=torch.long, device=device)
            v = torch.tensor(b["v"].values, dtype=torch.long, device=device)
            t = torch.tensor(b["tau"].values, dtype=torch.float, device=device)
            c = torch.tensor(b["c"].values, dtype=torch.float, device=device)
            years = b["YEAR"].values

            neg_v = negative_sample(u, n_plants, n_polls, device)
            pos_s = torch.sigmoid(model.score(u, v, t)).cpu().numpy()
            neg_s = torch.sigmoid(model.score(u, neg_v, t)).cpu().numpy()

            for i in range(len(u)):
                y = years[i]
                if y < min(hiatus_years):
                    pre_scores.extend([pos_s[i], neg_s[i]])
                    pre_labels.extend([1.0, 0.0])
                elif y > max(hiatus_years):
                    post_scores.extend([pos_s[i], neg_s[i]])
                    post_labels.extend([1.0, 0.0])

            model.update(u, v, t, c)

    pre_ap = float(average_precision_score(pre_labels, pre_scores)) if pre_labels else 0.0
    pre_auc = float(roc_auc_score(pre_labels, pre_scores)) if pre_labels else 0.0
    post_ap = float(average_precision_score(post_labels, post_scores)) if post_labels else 0.0
    post_auc = float(roc_auc_score(post_labels, post_scores)) if post_labels else 0.0

    test_years = sorted(test_df["YEAR"].unique())
    pre_sub = test_df[test_df["YEAR"] < min(hiatus_years)]
    post_sub = test_df[test_df["YEAR"] > max(hiatus_years)]

    gap_days = 0
    if len(pre_sub) > 0 and len(post_sub) > 0:
        gap_days = int(post_sub["tau"].min() - pre_sub["tau"].max())

    return {
        "test_years_present": [int(y) for y in test_years],
        "hiatus_duration_days": gap_days,
        "pre_hiatus_n_events": len(pre_sub),
        "post_hiatus_n_events": len(post_sub),
        "pre_hiatus_ap": pre_ap,
        "pre_hiatus_auc": pre_auc,
        "post_hiatus_ap": post_ap,
        "post_hiatus_auc": post_auc,
        "ap_drop_absolute": pre_ap - post_ap,
        "ap_drop_percent": ((pre_ap - post_ap) / max(pre_ap, 1e-6)) * 100.0,
        "auc_drop_absolute": pre_auc - post_auc,
        "auc_drop_percent": ((pre_auc - post_auc) / max(pre_auc, 1e-6)) * 100.0,
    }
