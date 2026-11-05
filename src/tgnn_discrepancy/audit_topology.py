"""
Audit graph topology, bipartite turnover, transductive vs inductive node availability,
and recurrent vs novel link prediction performance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from pp_tgnn.model import TGNNLite
from pp_tgnn.train import encode_events, negative_sample

log = logging.getLogger("tgnn_discrepancy.topology")


def analyze_node_and_edge_turnover(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Analyzes species and interaction pair overlap and inductive drift across splits."""
    train_plants = set(train_df["plant_id"].unique())
    train_polls = set(train_df["pollinator_id"].unique())
    train_pairs = set(zip(train_df["plant_id"], train_df["pollinator_id"]))

    val_plants = set(val_df["plant_id"].unique())
    val_polls = set(val_df["pollinator_id"].unique())
    val_pairs = set(zip(val_df["plant_id"], val_df["pollinator_id"]))

    test_plants = set(test_df["plant_id"].unique())
    test_polls = set(test_df["pollinator_id"].unique())
    test_pairs = set(zip(test_df["plant_id"], test_df["pollinator_id"]))

    unseen_plants_val = val_plants - train_plants
    unseen_polls_val = val_polls - train_polls

    unseen_plants_test = test_plants - train_plants
    unseen_polls_test = test_polls - train_polls

    # Categorize test events
    test_records = list(zip(test_df["plant_id"], test_df["pollinator_id"]))
    recurrent_count = 0
    semi_novel_count = 0
    inductive_count = 0

    for p, q in test_records:
        if (p, q) in train_pairs:
            recurrent_count += 1
        elif p in train_plants and q in train_polls:
            semi_novel_count += 1
        else:
            inductive_count += 1

    n_test = len(test_df)

    return {
        "train": {
            "n_events": len(train_df),
            "n_plants": len(train_plants),
            "n_polls": len(train_polls),
            "n_pairs": len(train_pairs),
        },
        "val": {
            "n_events": len(val_df),
            "n_plants": len(val_plants),
            "n_polls": len(val_polls),
            "unseen_plants": len(unseen_plants_val),
            "unseen_plants_pct": len(unseen_plants_val) / max(len(val_plants), 1) * 100.0,
            "unseen_polls": len(unseen_polls_val),
            "unseen_polls_pct": len(unseen_polls_val) / max(len(val_polls), 1) * 100.0,
        },
        "test": {
            "n_events": n_test,
            "n_plants": len(test_plants),
            "n_polls": len(test_polls),
            "unseen_plants": len(unseen_plants_test),
            "unseen_plants_pct": len(unseen_plants_test) / max(len(test_plants), 1) * 100.0,
            "unseen_polls": len(unseen_polls_test),
            "unseen_polls_pct": len(unseen_polls_test) / max(len(test_polls), 1) * 100.0,
            "recurrent_events": recurrent_count,
            "recurrent_events_pct": (recurrent_count / max(n_test, 1)) * 100.0,
            "semi_novel_events": semi_novel_count,
            "semi_novel_events_pct": (semi_novel_count / max(n_test, 1)) * 100.0,
            "inductive_events": inductive_count,
            "inductive_events_pct": (inductive_count / max(n_test, 1)) * 100.0,
        },
    }


def evaluate_topological_stratification(
    model: TGNNLite,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    plant2idx: Dict[str, int],
    poll2idx: Dict[str, int],
    device: torch.device,
    batch_size: int = 200,
) -> Dict[str, Any]:
    """Evaluates TGNN link prediction accuracy separately on recurrent, semi-novel, and inductive edges."""
    n_plants = len(plant2idx)
    n_polls = len(poll2idx)

    train_plants = set(train_df["plant_id"].unique())
    train_polls = set(train_df["pollinator_id"].unique())
    train_pairs = set(zip(train_df["plant_id"], train_df["pollinator_id"]))

    train_enc = encode_events(train_df, plant2idx, poll2idx)
    val_enc = encode_events(val_df, plant2idx, poll2idx)
    test_enc = encode_events(test_df, plant2idx, poll2idx).sort_values("tau").reset_index(drop=True)

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

        rec_scores, rec_labels = [], []
        semi_scores, semi_labels = [], []
        ind_scores, ind_labels = [], []

        for start in range(0, len(test_enc), batch_size):
            b = test_enc.iloc[start : start + batch_size]
            u = torch.tensor(b["u"].values, dtype=torch.long, device=device)
            v = torch.tensor(b["v"].values, dtype=torch.long, device=device)
            t = torch.tensor(b["tau"].values, dtype=torch.float, device=device)
            c = torch.tensor(b["c"].values, dtype=torch.float, device=device)

            pairs = list(zip(b["plant_id"], b["pollinator_id"]))

            neg_v = negative_sample(u, n_plants, n_polls, device)
            pos_s = torch.sigmoid(model.score(u, v, t)).cpu().numpy()
            neg_s = torch.sigmoid(model.score(u, neg_v, t)).cpu().numpy()

            for i in range(len(u)):
                p_id, q_id = pairs[i]
                if (p_id, q_id) in train_pairs:
                    rec_scores.extend([pos_s[i], neg_s[i]])
                    rec_labels.extend([1.0, 0.0])
                elif p_id in train_plants and q_id in train_polls:
                    semi_scores.extend([pos_s[i], neg_s[i]])
                    semi_labels.extend([1.0, 0.0])
                else:
                    ind_scores.extend([pos_s[i], neg_s[i]])
                    ind_labels.extend([1.0, 0.0])

            model.update(u, v, t, c)

    def _safe_metrics(labels: List[float], scores: List[float]) -> Tuple[float, float]:
        if len(labels) == 0:
            return 0.0, 0.0
        return float(average_precision_score(labels, scores)), float(roc_auc_score(labels, scores))

    rec_ap, rec_auc = _safe_metrics(rec_labels, rec_scores)
    semi_ap, semi_auc = _safe_metrics(semi_labels, semi_scores)
    ind_ap, ind_auc = _safe_metrics(ind_labels, ind_scores)

    return {
        "recurrent_edges": {
            "n_pairs": len(rec_labels) // 2,
            "ap": rec_ap,
            "auc": rec_auc,
        },
        "semi_novel_edges": {
            "n_pairs": len(semi_labels) // 2,
            "ap": semi_ap,
            "auc": semi_auc,
        },
        "inductive_edges": {
            "n_pairs": len(ind_labels) // 2,
            "ap": ind_ap,
            "auc": ind_auc,
        },
        "recurrent_vs_inductive_ap_gap": rec_ap - ind_ap,
        "recurrent_vs_inductive_auc_gap": rec_auc - ind_auc,
    }
