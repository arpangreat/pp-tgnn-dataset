"""
Detection-Bias-Corrected Temporal Event-Stream Pipeline
=========================================================

Implements Algorithms 1-8 from the accompanying data paper:

    1. harmonize_raw_data        (Algorithm 1)
    2. partition_interactions    (Algorithm 2)
    3. build_replicate_groups    (Algorithm 3)
    4. fit_detection_probability (Algorithm 4, uses log_marginal_over_lamgrid / Listing 1)
    5. flowering_confidence      (Algorithm 5)
    6. join_with_fallback        (Algorithm 6)
    7. build_event_stream        (Algorithm 7)
    8. chronological_split       (Algorithm 8)

`run_pipeline()` orchestrates all eight steps and writes the five
processed files described in Table 1 of the paper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binom, poisson

from .config import PipelineConfig

log = logging.getLogger("pp_tgnn.pipeline")


# --------------------------------------------------------------------------
# Algorithm 1 — HarmonizeRawData
# --------------------------------------------------------------------------

def harmonize_raw_data(
    vis_path: Path, flw_path: Path, cfg: PipelineConfig = PipelineConfig()
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two raw source files, each parsed with its OWN date format.

    Pitfall documented in the paper (Section 2.1): SA02601 uses DD-MM-YYYY
    while SA02602 uses YYYY-MM-DD. Parsing both with a single shared format
    string silently produces incorrect or null dates for one of the two
    files. We parse them independently and assert that the parsed calendar
    year matches the YEAR column already present in the raw record.
    """
    log.info("Loading raw visitation file: %s", vis_path)
    vis = pd.read_csv(vis_path, low_memory=False)
    log.info("Loading raw flowering file: %s", flw_path)
    flw = pd.read_csv(flw_path, low_memory=False)

    vis["date"] = pd.to_datetime(vis[cfg.vis_date_col], format=cfg.vis_date_fmt, errors="raise")
    flw["date"] = pd.to_datetime(flw[cfg.flw_date_col], format=cfg.flw_date_fmt, errors="raise")

    _assert_year_consistency(vis, cfg.year_col, "visitation")
    _assert_year_consistency(flw, cfg.year_col, "flowering")

    log.info("Harmonized vis=%d rows, flw=%d rows", len(vis), len(flw))
    return vis, flw


def _assert_year_consistency(df: pd.DataFrame, year_col: str, name: str) -> None:
    mismatched = df["date"].dt.year != df[year_col]
    if mismatched.any():
        raise AssertionError(
            f"[harmonize_raw_data] Parsed-date/YEAR mismatch in {name} file "
            f"({mismatched.sum()} rows). This almost always means a "
            f"date-format mismatch (see Section 2.1 of the paper)."
        )


# --------------------------------------------------------------------------
# Algorithm 2 — PartitionInteractions
# --------------------------------------------------------------------------

def partition_interactions(
    vis: pd.DataFrame, cfg: PipelineConfig = PipelineConfig()
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the visitation table into positive interactions E+ and
    structural non-detections E0, using the PPI_STATUS flag."""
    assert vis[cfg.ppi_status_col].isin([0, 1]).all(), (
        "[partition_interactions] PPI_STATUS must be in {0, 1} for every row."
    )

    e_pos = vis[vis[cfg.ppi_status_col] == 1].copy()
    e_neg = vis[vis[cfg.ppi_status_col] == 0].copy()

    assert e_pos[cfg.plant_col].notna().all(), (
        "[partition_interactions] Found E+ rows with a null plant species code."
    )
    assert e_neg[cfg.plant_col].isna().all(), (
        "[partition_interactions] Found E0 rows with a non-null plant species code."
    )

    n_filled = e_pos[cfg.no_int_col].isna().sum()
    e_pos[cfg.no_int_col] = e_pos[cfg.no_int_col].fillna(1)

    log.info(
        "Partitioned E+=%d rows, E0=%d rows (%d NO_INT values filled with 1)",
        len(e_pos), len(e_neg), n_filled,
    )
    return e_pos, e_neg


# --------------------------------------------------------------------------
# Algorithm 3 — BuildReplicateGroups
# --------------------------------------------------------------------------

@dataclass
class ReplicateGroup:
    plot: object
    year: object
    plant: object
    poll: object
    y: np.ndarray   # replicate count vector y_g
    K: int          # number of replicate watches conducted


def build_replicate_groups(
    e_pos: pd.DataFrame, vis: pd.DataFrame, cfg: PipelineConfig = PipelineConfig()
) -> List[ReplicateGroup]:
    """Assemble the watch-level replicate count vector y_g for every
    (plot, year, plant, pollinator) group.

    The replicate set for a group is every watch ACTUALLY CONDUCTED at
    that plot in that year — a watch on which the pair went undetected
    still counts as a replicate observation of value zero.
    """
    key_cols = list(cfg.key_cols)
    watch_totals = e_pos.groupby(key_cols, dropna=False)[cfg.no_int_col].sum().reset_index()

    all_watches = (
        vis.groupby([cfg.plot_col, cfg.year_col])[cfg.watch_col].unique().to_dict()
    )

    lookup: Dict[Tuple, float] = {}
    for row in watch_totals.itertuples(index=False):
        k = (
            getattr(row, cfg.plot_col), getattr(row, cfg.year_col),
            getattr(row, cfg.plant_col), getattr(row, cfg.poll_col),
            getattr(row, cfg.watch_col),
        )
        lookup[k] = getattr(row, cfg.no_int_col)

    group_keys = watch_totals[[cfg.plot_col, cfg.year_col, cfg.plant_col, cfg.poll_col]].drop_duplicates()

    groups: List[ReplicateGroup] = []
    for plot, year, plant, poll in group_keys.itertuples(index=False):
        watches = all_watches.get((plot, year), np.array([]))
        y_g = np.array(
            [lookup.get((plot, year, plant, poll, w), 0.0) for w in watches], dtype=float
        )
        groups.append(ReplicateGroup(plot, year, plant, poll, y_g, len(watches)))

    log.info("Built %d replicate groups", len(groups))
    return groups


# --------------------------------------------------------------------------
# Listing 1 — Vectorized log marginal likelihood
# --------------------------------------------------------------------------

def log_marginal_over_lamgrid(
    y: np.ndarray, lam_grid: np.ndarray, p: float, n_trunc_extra: int = 30
) -> np.ndarray:
    """log P(y_g | lambda, p), evaluated simultaneously for every lambda
    on a grid via NumPy broadcasting (paper Listing 1)."""
    y = np.asarray(y)
    n_max = int(y.max()) + n_trunc_extra if y.size else n_trunc_extra
    n_start = int(y.max()) if y.size else 0
    n_range = np.arange(n_start, n_max + 1)  # (Nc,)

    log_lik_k = binom.logpmf(y[:, None], n_range[None, :], p)  # (K, Nc)
    log_lik_sum = log_lik_k.sum(axis=0)  # (Nc,)

    log_prior_N = poisson.logpmf(n_range[:, None], lam_grid[None, :])  # (Nc, L)

    log_joint = log_lik_sum[:, None] + log_prior_N  # (Nc, L)
    m = log_joint.max(axis=0, keepdims=True)
    log_ml = (
        m + np.log(np.sum(np.exp(log_joint - m), axis=0, keepdims=True))
    ).ravel()
    return log_ml  # (L,)


# --------------------------------------------------------------------------
# Algorithm 4 — FitDetectionProbability
# --------------------------------------------------------------------------

def fit_detection_probability(
    groups: List[ReplicateGroup], cfg: PipelineConfig = PipelineConfig()
) -> Tuple[float, Dict[Tuple, float]]:
    """Two-stage fit: (1) pool informative groups (K_g >= 2) to identify a
    single detection probability p_hat, (2) compute the posterior mean
    lambda_hat_g for every group given p_hat."""
    rng = np.random.default_rng(cfg.rng_seed)

    informative = [g for g in groups if g.K >= 2]
    log.info("Informative groups (K_g >= 2): %d / %d total", len(informative), len(groups))

    if len(informative) == 0:
        raise ValueError("No informative groups (K_g >= 2) found — cannot fit p_hat.")

    sample_idx = rng.choice(
        len(informative), size=min(cfg.max_informative_sample, len(informative)), replace=False
    )
    sampled = [informative[i] for i in sample_idx]

    totals = np.zeros(len(cfg.p_grid))
    for i, p in enumerate(cfg.p_grid):
        total = 0.0
        for g in sampled:
            ell = log_marginal_over_lamgrid(g.y, cfg.lam_grid, p, cfg.n_trunc_extra)
            total += ell.max()
        totals[i] = total

    p_hat = float(cfg.p_grid[int(np.argmax(totals))])
    log.info("Fitted pooled detection probability p_hat = %.4f", p_hat)

    lambda_hat: Dict[Tuple, float] = {}
    for g in groups:
        ell = log_marginal_over_lamgrid(g.y, cfg.lam_grid, p_hat, cfg.n_trunc_extra)
        w = np.exp(ell - ell.max())
        w = w / w.sum()
        lambda_hat[(g.plot, g.year, g.plant, g.poll)] = float(np.sum(w * cfg.lam_grid))

    return p_hat, lambda_hat


def posterior_confidence(lambda_hat: Dict[Tuple, float]) -> Dict[Tuple, float]:
    """c_post = minmax(log(1 + lambda_hat_g)), normalized globally (Eq. 4)."""
    keys = list(lambda_hat.keys())
    vals = np.array([lambda_hat[k] for k in keys], dtype=float)
    log_vals = np.log1p(vals)
    lo, hi = log_vals.min(), log_vals.max()
    norm = np.full_like(log_vals, 0.5) if (hi - lo) < 1e-9 else (log_vals - lo) / (hi - lo)
    return {k: float(v) for k, v in zip(keys, norm)}


# --------------------------------------------------------------------------
# Algorithm 5 — FloweringConfidence
# --------------------------------------------------------------------------

def flowering_confidence(flw: pd.DataFrame, cfg: PipelineConfig = PipelineConfig()) -> pd.DataFrame:
    """Log-transform flower counts and normalize within each plant species
    (Eq. 5). Degenerate species (no variation) get a neutral 0.5."""
    flw = flw.copy()
    flw[cfg.no_flws_col] = flw[cfg.no_flws_col].fillna(0)
    flw["log_flws"] = np.log1p(flw[cfg.no_flws_col])

    # Vectorized within-species min-max normalization via groupby-transform
    # (avoids a pandas apply-return-shape ambiguity that occurs when a
    # species group happens to be degenerate/singleton).
    grp = flw.groupby(cfg.plant_col)["log_flws"]
    lo = grp.transform("min")
    hi = grp.transform("max")
    span = hi - lo

    normalized = (flw["log_flws"] - lo) / span
    flw["confidence_flowering"] = normalized.where(span >= 1e-9, 0.5)

    log.info("Computed flowering confidence for %d records", len(flw))
    return flw


# --------------------------------------------------------------------------
# Algorithm 6 — JoinWithFallback
# --------------------------------------------------------------------------

def join_with_fallback(
    interactions: pd.DataFrame, flw_conf: pd.DataFrame, cfg: PipelineConfig = PipelineConfig()
) -> pd.DataFrame:
    """Left join on (plot, year, watch, plant). For unmatched rows, fall
    back to the most recent PRIOR flowering record for that species in
    that plot-year (last-observation-carried-forward)."""
    join_cols = [cfg.plot_col, cfg.year_col, cfg.watch_col, cfg.plant_col]
    flw_small = flw_conf[join_cols + ["confidence_flowering", "date"]].drop_duplicates(
        subset=join_cols, keep="last"
    )

    merged = interactions.merge(flw_small, on=join_cols, how="left", suffixes=("", "_flw"))
    merged["is_imputed_flowering"] = merged["confidence_flowering"].isna()

    n_missing = merged["is_imputed_flowering"].sum()
    log.info("Rows needing flowering fallback: %d (%.3f%%)", n_missing, 100 * n_missing / len(merged))

    if n_missing > 0:
        history = (
            flw_conf[[cfg.plot_col, cfg.year_col, cfg.plant_col, "date", "confidence_flowering"]]
            .sort_values("date")
        )
        history_groups = {
            k: v.reset_index(drop=True) for k, v in history.groupby([cfg.plot_col, cfg.year_col, cfg.plant_col])
        }

        idx_missing = merged.index[merged["is_imputed_flowering"]]
        for idx in idx_missing:
            row = merged.loc[idx]
            key = (row[cfg.plot_col], row[cfg.year_col], row[cfg.plant_col])
            cand = history_groups.get(key)
            if cand is not None:
                prior = cand[cand["date"] <= row["date"]]
                if len(prior) > 0:
                    merged.at[idx, "confidence_flowering"] = prior.iloc[-1]["confidence_flowering"]

    merged["confidence_flowering"] = merged["confidence_flowering"].fillna(0.5)
    return merged


# --------------------------------------------------------------------------
# Algorithm 7 — BuildEventStream
# --------------------------------------------------------------------------

def build_event_stream(
    e_pos: pd.DataFrame,
    posterior_conf: Dict[Tuple, float],
    flw_conf: pd.DataFrame,
    cfg: PipelineConfig = PipelineConfig(),
) -> pd.DataFrame:
    """Assemble the final weighted temporal event stream
    (tau, plant_id, pollinator_id, c)."""
    group_cols = [cfg.plot_col, cfg.year_col, cfg.watch_col, cfg.plant_col, cfg.poll_col, "date"]
    watch_level = e_pos.groupby(group_cols, dropna=False)[cfg.no_int_col].sum().reset_index()

    watch_level["confidence_posterior"] = watch_level.apply(
        lambda r: posterior_conf.get(
            (r[cfg.plot_col], r[cfg.year_col], r[cfg.plant_col], r[cfg.poll_col]), 0.5
        ),
        axis=1,
    )

    watch_level = join_with_fallback(watch_level, flw_conf, cfg)
    watch_level["c"] = watch_level["confidence_posterior"] * watch_level["confidence_flowering"]

    t0 = watch_level["date"].min()
    watch_level["tau"] = (watch_level["date"] - t0).dt.days
    watch_level = watch_level.sort_values("tau").reset_index(drop=True)

    out = watch_level.rename(columns={cfg.plant_col: "plant_id", cfg.poll_col: "pollinator_id"})
    cols = [
        "tau", "date", "plant_id", "pollinator_id", cfg.no_int_col,
        "confidence_posterior", "confidence_flowering", "c",
        "is_imputed_flowering", cfg.plot_col, cfg.year_col, cfg.watch_col,
    ]
    out = out[[c for c in cols if c in out.columns]]
    log.info(
        "Assembled event stream with %d rows, spanning tau in [0, %d]",
        len(out), out["tau"].max() if len(out) else 0,
    )
    return out


# --------------------------------------------------------------------------
# Algorithm 8 — ChronologicalSplit
# --------------------------------------------------------------------------

def chronological_split(
    events: pd.DataFrame, cfg: PipelineConfig = PipelineConfig()
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leakage-safe 80/10/10 split by cumulative event volume over tau,
    robust to arbitrary sampling gaps (e.g., the 2019-2020 hiatus)."""
    events = events.sort_values("tau").reset_index(drop=True)
    n = len(events)
    train_end = int(np.floor(n * cfg.train_frac))
    val_end = int(np.floor(n * (cfg.train_frac + cfg.val_frac)))

    train = events.iloc[:train_end].copy()
    val = events.iloc[train_end:val_end].copy()
    test = events.iloc[val_end:].copy()

    log.info("Chronological split -> train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return train, val, test


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_pipeline(
    vis_path: Path, flw_path: Path, outdir: Path, cfg: PipelineConfig = PipelineConfig()
) -> Dict[str, pd.DataFrame]:
    """Run the full pipeline (Algorithms 1-8) and write the five processed
    files. Returns a dict of the in-memory DataFrames for programmatic use
    (e.g. from a notebook), keyed by:
        'events', 'negatives', 'train', 'val', 'test', 'lambda_hat'
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    vis, flw = harmonize_raw_data(vis_path, flw_path, cfg)
    e_pos, e_neg = partition_interactions(vis, cfg)

    groups = build_replicate_groups(e_pos, vis, cfg)
    p_hat, lambda_hat = fit_detection_probability(groups, cfg)
    posterior_conf = posterior_confidence(lambda_hat)

    flw_conf = flowering_confidence(flw, cfg)

    events = build_event_stream(e_pos, posterior_conf, flw_conf, cfg)
    train, val, test = chronological_split(events, cfg)

    events.to_csv(outdir / "plant_pollinator_events.csv", index=False)

    neg_cols = [c for c in [cfg.plot_col, cfg.year_col, cfg.watch_col, "date"] if c in e_neg.columns]
    e_neg[neg_cols].to_csv(outdir / "plant_pollinator_negative_events.csv", index=False)

    train.to_csv(outdir / "events_train.csv", index=False)
    val.to_csv(outdir / "events_val.csv", index=False)
    test.to_csv(outdir / "events_test.csv", index=False)

    lam_df = pd.DataFrame(
        [{"plot": k[0], "year": k[1], "plant": k[2], "poll": k[3], "lambda_hat": v}
         for k, v in lambda_hat.items()]
    )
    lam_df.to_csv(outdir / "lambda_hat.csv", index=False)
    flw_conf.to_csv(outdir / "flw_conf.csv", index=False)

    n_pairs = events[["plant_id", "pollinator_id"]].drop_duplicates().shape[0]
    n_plants = events["plant_id"].nunique()
    n_polls = events["pollinator_id"].nunique()
    density = n_pairs / (n_plants * n_polls) if n_plants and n_polls else float("nan")

    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("Positive events: %d", len(events))
    log.info("Negative events: %d", len(e_neg))
    log.info("Distinct plant-pollinator pairs: %d (density %.2f%%)", n_pairs, 100 * density)
    log.info("Median edge weight c: %.3f", events["c"].median())
    log.info("Fraction imputed flowering: %.3f%%", 100 * events["is_imputed_flowering"].mean())
    log.info("Detection probability p_hat: %.3f", p_hat)
    log.info("=" * 60)

    return {
        "events": events,
        "negatives": e_neg,
        "train": train,
        "val": val,
        "test": test,
        "lambda_hat": lam_df,
        "flw_conf": flw_conf,
        "p_hat": p_hat,
    }
