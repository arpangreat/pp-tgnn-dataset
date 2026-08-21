"""Training / evaluation loop and metrics for TGNNLite, operating on the
chronologically-ordered event stream produced by pp_tgnn.pipeline."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from .model import TGNNLite

log = logging.getLogger("pp_tgnn.train")


def build_node_vocab(events: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int], int, int]:
    """Fit a global node vocabulary across the FULL event stream (standard
    practice for transductive dynamic-link-prediction benchmarks). Plants
    and pollinators share one contiguous index space."""
    plants = sorted(events["plant_id"].unique())
    polls = sorted(events["pollinator_id"].unique())
    plant2idx = {p: i for i, p in enumerate(plants)}
    poll2idx = {p: i + len(plants) for i, p in enumerate(polls)}
    return plant2idx, poll2idx, len(plants), len(polls)


def encode_events(df: pd.DataFrame, plant2idx: Dict[str, int], poll2idx: Dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    df["u"] = df["plant_id"].map(plant2idx)
    df["v"] = df["pollinator_id"].map(poll2idx)
    return df


def negative_sample(batch_u: torch.Tensor, n_plants: int, n_polls: int, device: torch.device) -> torch.Tensor:
    """Uniform random negative sampling over pollinator nodes."""
    neg_v = np.random.randint(0, n_polls, size=len(batch_u)) + n_plants
    return torch.tensor(neg_v, dtype=torch.long, device=device)


def run_epoch(
    model: TGNNLite,
    df: pd.DataFrame,
    n_plants: int,
    n_polls: int,
    device: torch.device,
    opt: torch.optim.Optimizer | None = None,
    batch_size: int = 200,
) -> Tuple[float, float, float]:
    """One chronological pass over `df`. If `opt` is provided, trains;
    otherwise evaluates only. Memory always advances chronologically
    regardless of mode. Returns (mean_loss, AP, AUC) against random negatives.
    """
    train_mode = opt is not None

    u_all = torch.tensor(df["u"].values, dtype=torch.long, device=device)
    v_all = torch.tensor(df["v"].values, dtype=torch.long, device=device)
    t_all = torch.tensor(df["tau"].values, dtype=torch.float, device=device)
    c_all = torch.tensor(df["c"].values, dtype=torch.float, device=device)

    total_loss, n_batches = 0.0, 0
    all_scores, all_labels = [], []

    for start in range(0, len(df), batch_size):
        end = start + batch_size
        u, v, t, c = u_all[start:end], v_all[start:end], t_all[start:end], c_all[start:end]
        if len(u) == 0:
            continue

        neg_v = negative_sample(u, n_plants, n_polls, device)

        pos_scores = model.score(u, v, t)
        neg_scores = model.score(u, neg_v, t)

        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        scores = torch.cat([pos_scores, neg_scores])
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        if train_mode:
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            model.update(u, v, t, c)

        total_loss += loss.item()
        n_batches += 1
        all_scores.append(torch.sigmoid(scores).detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    ap = average_precision_score(all_labels, all_scores)
    auc = roc_auc_score(all_labels, all_scores)
    return total_loss / max(n_batches, 1), ap, auc


def mrr_eval(
    model: TGNNLite,
    df: pd.DataFrame,
    n_plants: int,
    n_polls: int,
    device: torch.device,
    n_neg: int = 50,
    n_samples: int = 300,
    seed: int = 0,
) -> float:
    """Rank the true pollinator against `n_neg` random negatives for a
    sample of positive events; report Mean Reciprocal Rank."""
    rng = np.random.default_rng(seed)
    sample = df.sample(min(n_samples, len(df)), random_state=seed)
    u_all = torch.tensor(sample["u"].values, dtype=torch.long, device=device)
    v_all = torch.tensor(sample["v"].values, dtype=torch.long, device=device)
    t_all = torch.tensor(sample["tau"].values, dtype=torch.float, device=device)

    ranks = []
    with torch.no_grad():
        for i in range(len(sample)):
            u, v, t = u_all[i:i + 1], v_all[i:i + 1], t_all[i:i + 1]
            neg_v = torch.tensor(
                rng.choice(n_polls, size=n_neg, replace=False) + n_plants,
                dtype=torch.long, device=device,
            )
            cand_v = torch.cat([v, neg_v])
            cand_u = u.repeat(len(cand_v))
            cand_t = t.repeat(len(cand_v))
            scores = model.score(cand_u, cand_v, cand_t)
            rank = (scores > scores[0]).sum().item() + 1
            ranks.append(1.0 / rank)
    return float(np.mean(ranks))
