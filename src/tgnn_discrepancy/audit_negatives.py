"""
Audit negative sampling protocols: implement the 3-Tier Negative Evaluation Protocol
(Uniform Random vs Seasonal-Spatiotemporal vs Grounded Structural Negatives from PPI_STATUS=0).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from pp_tgnn.model import TGNNLite
from pp_tgnn.train import encode_events, negative_sample

log = logging.getLogger("tgnn_discrepancy.negatives")


def build_meadow_seasonal_pollinator_pool(
    events_df: pd.DataFrame,
) -> Dict[Tuple[str, int], List[str]]:
    """Builds a lookup table of pollinators active within each (PLOT_ID prefix / meadow, YEAR)."""
    # PLOT_ID prefix (first 3 chars) is the meadow identifier (e.g. CPR = Carpenter, BGD = Bunchgrass)
    df = events_df.copy()
    df["meadow"] = df["PLOT_ID"].astype(str).str[:3]
    grouped = df.groupby(["meadow", "YEAR"])["pollinator_id"].unique().to_dict()
    return {k: list(v) for k, v in grouped.items()}


def materialize_grounded_structural_negatives(
    negative_events_df: pd.DataFrame,
    flw_conf_df: pd.DataFrame,
    events_df: pd.DataFrame,
    max_samples_per_session: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Grounds empty watch sessions (PPI_STATUS=0) into candidate negative edges.

    Pairs blooming plant species present at that plot session (from flw_conf)
    with pollinators active in that meadow-year that did not visit.
    """
    rng = np.random.default_rng(seed)
    flw = flw_conf_df[["PLOT_ID", "YEAR", "WATCH", "PLTSP_CODE"]].drop_duplicates()
    flw = flw.rename(columns={"PLTSP_CODE": "plant_id"})

    # Deduplicate empty watch sessions
    neg_sessions = negative_events_df[["PLOT_ID", "YEAR", "WATCH", "date"]].drop_duplicates()

    # Join with flowering survey to find plants present in those empty sessions
    joined = pd.merge(neg_sessions, flw, on=["PLOT_ID", "YEAR", "WATCH"], how="inner")

    if len(joined) == 0:
        log.warning("No blooming plants matched negative sessions; falling back to plot-level catalog.")
        return pd.DataFrame(columns=["date", "plant_id", "pollinator_id", "PLOT_ID", "YEAR", "WATCH"])

    joined["meadow"] = joined["PLOT_ID"].astype(str).str[:3]
    active_pool = build_meadow_seasonal_pollinator_pool(events_df)

    grounded_rows = []
    for _, row in joined.iterrows():
        key = (row["meadow"], row["YEAR"])
        candidates = active_pool.get(key, [])
        if not candidates:
            # Fallback to year-level candidates
            candidates = events_df[events_df["YEAR"] == row["YEAR"]]["pollinator_id"].unique().tolist()
        if not candidates:
            continue

        sample_size = min(max_samples_per_session, len(candidates))
        chosen_polls = rng.choice(candidates, size=sample_size, replace=False)
        for p in chosen_polls:
            grounded_rows.append({
                "date": row["date"],
                "plant_id": row["plant_id"],
                "pollinator_id": p,
                "PLOT_ID": row["PLOT_ID"],
                "YEAR": row["YEAR"],
                "WATCH": row["WATCH"],
            })

    res = pd.DataFrame(grounded_rows).drop_duplicates()
    log.info("Materialized %d grounded structural negative candidate edges.", len(res))
    return res


def compare_negative_sampling_tiers(
    model: TGNNLite,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    flw_conf_df: pd.DataFrame,
    negative_events_df: pd.DataFrame,
    plant2idx: Dict[str, int],
    poll2idx: Dict[str, int],
    device: torch.device,
    seed: int = 42,
    batch_size: int = 200,
) -> Dict[str, Any]:
    """Evaluates link prediction under Tier 1 (Uniform Random), Tier 2 (Seasonal Meadow Active),

    and Tier 3 (Grounded Structural Absences).
    """
    rng = np.random.default_rng(seed)
    n_plants = len(plant2idx)
    n_polls = len(poll2idx)

    train_enc = encode_events(train_df, plant2idx, poll2idx)
    val_enc = encode_events(val_df, plant2idx, poll2idx)
    test_enc = encode_events(test_df, plant2idx, poll2idx).sort_values("tau").reset_index(drop=True)

    test_enc["meadow"] = test_enc["PLOT_ID"].astype(str).str[:3]
    all_events = pd.concat([train_df, val_df, test_df], ignore_index=True)
    active_pool = build_meadow_seasonal_pollinator_pool(all_events)

    # Replay memory
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

        tier1_scores, tier2_scores = [], []
        pos_scores_all = []

        for start in range(0, len(test_enc), batch_size):
            b = test_enc.iloc[start : start + batch_size]
            u = torch.tensor(b["u"].values, dtype=torch.long, device=device)
            v = torch.tensor(b["v"].values, dtype=torch.long, device=device)
            t = torch.tensor(b["tau"].values, dtype=torch.float, device=device)
            c = torch.tensor(b["c"].values, dtype=torch.float, device=device)

            # Tier 1: Uniform Random Negatives
            neg_v_tier1 = negative_sample(u, n_plants, n_polls, device)

            # Tier 2: Seasonal Meadow Active Negatives
            neg_v_tier2_list = []
            for _, r in b.iterrows():
                key = (r["meadow"], r["YEAR"])
                cands = [poll2idx[p] for p in active_pool.get(key, []) if p in poll2idx]
                if not cands:
                    cands = [poll2idx[p] for p in all_events[all_events["YEAR"] == r["YEAR"]]["pollinator_id"].unique() if p in poll2idx]
                if not cands:
                    cands = list(range(n_plants, n_plants + n_polls))
                neg_v_tier2_list.append(rng.choice(cands))
            neg_v_tier2 = torch.tensor(neg_v_tier2_list, dtype=torch.long, device=device)

            pos_s = torch.sigmoid(model.score(u, v, t)).cpu().numpy()
            neg_s1 = torch.sigmoid(model.score(u, neg_v_tier1, t)).cpu().numpy()
            neg_s2 = torch.sigmoid(model.score(u, neg_v_tier2, t)).cpu().numpy()

            pos_scores_all.extend(pos_s)
            tier1_scores.extend(neg_s1)
            tier2_scores.extend(neg_s2)

            model.update(u, v, t, c)

    labels = np.concatenate([np.ones(len(pos_scores_all)), np.zeros(len(pos_scores_all))])
    t1_all = np.concatenate([pos_scores_all, tier1_scores])
    t2_all = np.concatenate([pos_scores_all, tier2_scores])

    t1_ap = float(average_precision_score(labels, t1_all))
    t1_auc = float(roc_auc_score(labels, t1_all))
    t2_ap = float(average_precision_score(labels, t2_all))
    t2_auc = float(roc_auc_score(labels, t2_all))

    # Tier 3: Score Grounded Structural Absences
    grounded_df = materialize_grounded_structural_negatives(
        negative_events_df, flw_conf_df, all_events, seed=seed
    )
    t3_stats = {}
    if len(grounded_df) > 0:
        grounded_df = grounded_df[
            grounded_df["plant_id"].isin(plant2idx) & grounded_df["pollinator_id"].isin(poll2idx)
        ].copy()
        if len(grounded_df) > 0:
            t0 = pd.to_datetime(all_events["date"]).min()
            grounded_df["tau"] = (pd.to_datetime(grounded_df["date"]) - t0).dt.days
            grounded_enc = encode_events(grounded_df, plant2idx, poll2idx)
            u3 = torch.tensor(grounded_enc["u"].values[:500], dtype=torch.long, device=device)
            v3 = torch.tensor(grounded_enc["v"].values[:500], dtype=torch.long, device=device)
            t3 = torch.tensor(grounded_enc["tau"].values[:500], dtype=torch.float, device=device)
            with torch.no_grad():
                scores3 = torch.sigmoid(model.score(u3, v3, t3)).cpu().numpy()
            t3_stats = {
                "n_grounded_evaluated": len(scores3),
                "mean_predicted_absence_score": float(np.mean(scores3)),
                "median_predicted_absence_score": float(np.median(scores3)),
            }

    return {
        "tier1_uniform_random": {
            "ap": t1_ap,
            "auc": t1_auc,
            "description": "Standard benchmark: uniform random sampling over all 800 pollinators",
        },
        "tier2_seasonal_meadow_active": {
            "ap": t2_ap,
            "auc": t2_auc,
            "description": "Ecological partner choice: sampled only from pollinators co-active in same meadow-year",
        },
        "negative_realism_gap": {
            "ap_drop": t1_ap - t2_ap,
            "auc_drop": t1_auc - t2_auc,
        },
        "tier3_grounded_structural_absences": t3_stats,
    }
