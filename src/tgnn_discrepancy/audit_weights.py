"""
Audit confidence weight distributions (c = confidence_posterior * confidence_flowering)
and evaluate TGNN performance with unweighted BCE vs confidence-weighted loss objectives.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from pp_tgnn.model import TGNNLite
from pp_tgnn.train import encode_events, negative_sample

log = logging.getLogger("tgnn_discrepancy.weights")


def analyze_weight_distribution(events_df: pd.DataFrame) -> Dict[str, Any]:
    """Computes distribution statistics, skewness, and correlation for edge weight c."""
    c = events_df["c"]
    no_int = events_df["NO_INT"]
    c_post = events_df["confidence_posterior"]
    c_flw = events_df["confidence_flowering"]

    corr_c_noint = float(c.corr(no_int))
    corr_cpost_noint = float(c_post.corr(no_int))

    return {
        "n_events": len(events_df),
        "c_summary": {
            "mean": float(c.mean()),
            "std": float(c.std()),
            "median": float(c.median()),
            "iqr": [float(c.quantile(0.25)), float(c.quantile(0.75))],
            "min": float(c.min()),
            "max": float(c.max()),
            "fraction_below_0_01": float((c < 0.01).mean()),
            "fraction_below_0_05": float((c < 0.05).mean()),
            "fraction_above_0_5": float((c > 0.5).mean()),
        },
        "confidence_posterior_median": float(c_post.median()),
        "confidence_flowering_median": float(c_flw.median()),
        "imputed_flowering_count": int(events_df["is_imputed_flowering"].sum()),
        "imputed_flowering_fraction": float(events_df["is_imputed_flowering"].mean()),
        "pearson_corr_c_vs_raw_visits": corr_c_noint,
        "pearson_corr_c_post_vs_raw_visits": corr_cpost_noint,
    }


def train_and_eval_model(
    train_enc: pd.DataFrame,
    val_enc: pd.DataFrame,
    test_enc: pd.DataFrame,
    n_nodes: int,
    n_plants: int,
    n_polls: int,
    device: torch.device,
    use_weighted_loss: bool = False,
    epochs: int = 3,
    lr: float = 1e-3,
    batch_size: int = 200,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Trains a TGNNLite instance with either unweighted BCE or confidence-weighted loss,

    and returns (test_ap, test_auc, test_loss).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TGNNLite(n_nodes, mem_dim=64, time_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for start in range(0, len(train_enc), batch_size):
            b = train_enc.iloc[start : start + batch_size]
            u = torch.tensor(b["u"].values, dtype=torch.long, device=device)
            v = torch.tensor(b["v"].values, dtype=torch.long, device=device)
            t = torch.tensor(b["tau"].values, dtype=torch.float, device=device)
            c = torch.tensor(b["c"].values, dtype=torch.float, device=device)

            neg_v = negative_sample(u, n_plants, n_polls, device)

            pos_scores = model.score(u, v, t)
            neg_scores = model.score(u, neg_v, t)

            if use_weighted_loss:
                # Confidence-weighted positive loss
                pos_loss = F.binary_cross_entropy_with_logits(
                    pos_scores, torch.ones_like(pos_scores), weight=c
                )
                neg_loss = F.binary_cross_entropy_with_logits(
                    neg_scores, torch.zeros_like(neg_scores)
                )
                loss = pos_loss + neg_loss
            else:
                scores = torch.cat([pos_scores, neg_scores])
                labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
                loss = F.binary_cross_entropy_with_logits(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.update(u, v, t, c)

    # Evaluate on test after replay through val
    model.eval()
    with torch.no_grad():
        for start in range(0, len(val_enc), batch_size):
            b = val_enc.iloc[start : start + batch_size]
            model.update(
                torch.tensor(b["u"].values, dtype=torch.long, device=device),
                torch.tensor(b["v"].values, dtype=torch.long, device=device),
                torch.tensor(b["tau"].values, dtype=torch.float, device=device),
                torch.tensor(b["c"].values, dtype=torch.float, device=device),
            )

        all_scores, all_labels = [], []
        total_loss = 0.0
        n_b = 0

        for start in range(0, len(test_enc), batch_size):
            b = test_enc.iloc[start : start + batch_size]
            u = torch.tensor(b["u"].values, dtype=torch.long, device=device)
            v = torch.tensor(b["v"].values, dtype=torch.long, device=device)
            t = torch.tensor(b["tau"].values, dtype=torch.float, device=device)
            c = torch.tensor(b["c"].values, dtype=torch.float, device=device)

            neg_v = negative_sample(u, n_plants, n_polls, device)
            pos_scores = model.score(u, v, t)
            neg_scores = model.score(u, neg_v, t)

            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
            b_loss = F.binary_cross_entropy_with_logits(scores, labels).item()
            total_loss += b_loss
            n_b += 1

            all_scores.extend(torch.sigmoid(scores).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            model.update(u, v, t, c)

    test_ap = float(average_precision_score(all_labels, all_scores))
    test_auc = float(roc_auc_score(all_labels, all_scores))
    return test_ap, test_auc, total_loss / max(n_b, 1)


def evaluate_weighted_vs_unweighted_training(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    plant2idx: Dict[str, int],
    poll2idx: Dict[str, int],
    device: torch.device,
    epochs: int = 3,
    batch_size: int = 200,
) -> Dict[str, Any]:
    """Compares TGNN metrics between unweighted and confidence-weighted training objectives."""
    n_plants = len(plant2idx)
    n_polls = len(poll2idx)
    n_nodes = n_plants + n_polls

    train_enc = encode_events(train_df, plant2idx, poll2idx)
    val_enc = encode_events(val_df, plant2idx, poll2idx)
    test_enc = encode_events(test_df, plant2idx, poll2idx).sort_values("tau").reset_index(drop=True)

    unw_ap, unw_auc, unw_loss = train_and_eval_model(
        train_enc, val_enc, test_enc, n_nodes, n_plants, n_polls, device,
        use_weighted_loss=False, epochs=epochs, batch_size=batch_size
    )

    wgt_ap, wgt_auc, wgt_loss = train_and_eval_model(
        train_enc, val_enc, test_enc, n_nodes, n_plants, n_polls, device,
        use_weighted_loss=True, epochs=epochs, batch_size=batch_size
    )

    return {
        "unweighted_bce": {
            "test_ap": unw_ap,
            "test_auc": unw_auc,
            "test_loss": unw_loss,
        },
        "confidence_weighted_bce": {
            "test_ap": wgt_ap,
            "test_auc": wgt_auc,
            "test_loss": wgt_loss,
        },
        "ap_delta": wgt_ap - unw_ap,
        "auc_delta": wgt_auc - unw_auc,
    }
