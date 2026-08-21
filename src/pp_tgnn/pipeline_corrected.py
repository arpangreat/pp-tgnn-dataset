"""
pipeline_corrected.py

Drop-in replacement for `fit_detection_probability` (Algorithm 4) that
addresses three reviewer-facing issues found when running the pipeline on
the real SA02601_v6 / SA02602_v5 data:

  R1. The original fit used a fixed sample of `cfg.max_informative_sample`
      (250) groups out of 17,342 informative groups, with no reported
      uncertainty on p_hat.
      -> fit here uses ALL 17,342 informative groups for the point
      estimate, and reports a bootstrap 95% CI.

  R2. No convergence diagnostic was reported for the N-truncation window.
      -> check_truncation_convergence() flags any group whose posterior
      mass isn't negligible at the edge of [max(y_g), max(y_g)+30].

  R3. A single pooled p was assumed with no stratified robustness check.
      -> fit_stratified_detection_probability() repeats the fit within
      each meadow COMPLEX using the same cached profile-likelihood trick,
      so it costs almost nothing extra once the full fit has been run.

Performance note: naively refitting p_hat from scratch for every bootstrap
resample would cost ~100s PER RESAMPLE (17,342 groups x 19 p-values x a
vectorized-over-lambda marginal-likelihood call each) -- prohibitively slow
for a meaningful bootstrap. Instead we compute the per-group, per-p profiled
log-likelihood ONCE into a (n_groups x n_p) matrix; every bootstrap /
stratified resample after that is just a sum-and-argmax over rows of that
cached matrix, which is essentially free. This is mathematically identical
to refitting from scratch (profiling out lambda_g is per-group and does not
depend on which OTHER groups are in the sample), it is just not redundantly
recomputed.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .pipeline import ReplicateGroup, log_marginal_over_lamgrid

log = logging.getLogger("pp_tgnn.pipeline_corrected")


def _profile_ll_matrix(
    groups: List[ReplicateGroup], p_grid: np.ndarray, lam_grid: np.ndarray, n_trunc_extra: int
) -> np.ndarray:
    """Compute profile_ll[i, j] = max_lambda log P(y_{groups[i]} | lambda, p_grid[j])
    for every group and every candidate p, ONCE. Shape (n_groups, n_p)."""
    n_groups, n_p = len(groups), len(p_grid)
    profile_ll = np.empty((n_groups, n_p))
    for j, p in enumerate(p_grid):
        for i, g in enumerate(groups):
            ell = log_marginal_over_lamgrid(g.y, lam_grid, p, n_trunc_extra)
            profile_ll[i, j] = ell.max()
    return profile_ll


def fit_detection_probability_full(
    groups: List[ReplicateGroup],
    cfg: PipelineConfig = PipelineConfig(),
    n_bootstrap: int = 500,
    random_state: int = 0,
) -> dict:
    """Fixes R1. Fits p_hat on ALL informative groups (K_g >= 2), and
    reports a bootstrap 95% CI computed from a cached profile-likelihood
    matrix (see module docstring for why this is fast and still correct).
    """
    informative = [g for g in groups if g.K >= 2]
    log.info("R1 fix: fitting on all %d informative groups (was: %d sampled)",
              len(informative), cfg.max_informative_sample)

    t0 = time.time()
    profile_ll = _profile_ll_matrix(informative, cfg.p_grid, cfg.lam_grid, cfg.n_trunc_extra)
    log.info("Profile-likelihood matrix computed in %.1fs (%d groups x %d p-values)",
              time.time() - t0, *profile_ll.shape)

    totals = profile_ll.sum(axis=0)
    p_hat = float(cfg.p_grid[int(np.argmax(totals))])

    rng = np.random.default_rng(random_state)
    n = profile_ll.shape[0]
    boot_p_hat = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_totals = profile_ll[idx].sum(axis=0)
        boot_p_hat[b] = cfg.p_grid[int(np.argmax(boot_totals))]

    ci = (float(np.percentile(boot_p_hat, 2.5)), float(np.percentile(boot_p_hat, 97.5)))

    log.info("p_hat = %.4f, 95%% bootstrap CI = (%.4f, %.4f), n_bootstrap=%d",
              p_hat, ci[0], ci[1], n_bootstrap)

    return {
        "p_hat": p_hat,
        "p_hat_ci": ci,
        "p_hat_bootstrap": boot_p_hat,
        "n_informative": len(informative),
        "profile_ll": profile_ll,   # returned for reuse by stratified fit
        "informative_groups": informative,
    }


def fit_stratified_detection_probability(
    groups: List[ReplicateGroup],
    strata: List[str],
    cfg: PipelineConfig = PipelineConfig(),
    min_groups_per_stratum: int = 20,
    n_bootstrap: int = 200,
    random_state: int = 0,
    profile_ll: Optional[np.ndarray] = None,
    informative_groups: Optional[List[ReplicateGroup]] = None,
) -> pd.DataFrame:
    """Fixes R3. Repeats the pooled fit within each stratum (e.g. meadow
    complex) using the SAME cached profile-likelihood matrix passed in
    from fit_detection_probability_full (pass its 'profile_ll' and
    'informative_groups' outputs to avoid recomputing anything).

    `strata` is a list of stratum labels, one per group in `informative_groups`
    (same order) -- e.g. each group's meadow COMPLEX.
    """
    if profile_ll is None or informative_groups is None:
        informative = [g for g in groups if g.K >= 2]
        profile_ll = _profile_ll_matrix(informative, cfg.p_grid, cfg.lam_grid, cfg.n_trunc_extra)
        informative_groups = informative

    assert len(strata) == len(informative_groups), (
        f"strata length {len(strata)} != informative_groups length {len(informative_groups)}"
    )
    strata_arr = np.asarray(strata)
    rng = np.random.default_rng(random_state)

    rows = []
    for stratum_val in sorted(set(strata)):
        mask = strata_arr == stratum_val
        n_stratum = int(mask.sum())
        if n_stratum < min_groups_per_stratum:
            rows.append({
                "stratum": stratum_val, "n_informative": n_stratum,
                "p_hat": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "skipped_too_few_groups": True,
            })
            continue

        sub = profile_ll[mask]
        totals = sub.sum(axis=0)
        p_hat = float(cfg.p_grid[int(np.argmax(totals))])

        boot_p_hat = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n_stratum, size=n_stratum)
            boot_p_hat[b] = cfg.p_grid[int(np.argmax(sub[idx].sum(axis=0)))]
        ci = (float(np.percentile(boot_p_hat, 2.5)), float(np.percentile(boot_p_hat, 97.5)))

        rows.append({
            "stratum": stratum_val, "n_informative": n_stratum,
            "p_hat": p_hat, "ci_lo": ci[0], "ci_hi": ci[1],
            "skipped_too_few_groups": False,
        })

    df = pd.DataFrame(rows)
    log.info("Stratified p_hat:\n%s", df.to_string(index=False))
    return df


def check_truncation_convergence(
    groups: List[ReplicateGroup],
    p_hat: float,
    cfg: PipelineConfig = PipelineConfig(),
    tail_mass_threshold: float = 1e-4,
) -> dict:
    """Fixes R2. For every group, evaluate what fraction of the N-posterior
    mass (at that group's own posterior-mean lambda) falls on the LAST
    value of the truncation window [max(y_g), max(y_g)+n_trunc_extra].
    If that tail mass exceeds `tail_mass_threshold`, the window is too
    short for that group and lambda_hat may be biased downward.
    """
    from scipy.stats import binom, poisson

    flagged = []
    worst = 0.0
    for idx, g in enumerate(groups):
        y = np.asarray(g.y)
        if y.size == 0:
            continue
        n_max = int(y.max()) + cfg.n_trunc_extra
        n_range = np.arange(int(y.max()), n_max + 1)

        log_lik_sum = binom.logpmf(y[:, None], n_range[None, :], p_hat).sum(axis=0)
        ell = log_marginal_over_lamgrid(y, cfg.lam_grid, p_hat, cfg.n_trunc_extra)
        w = np.exp(ell - ell.max())
        w /= w.sum()
        lam_point = float(np.sum(w * cfg.lam_grid))

        log_post_N = log_lik_sum + poisson.logpmf(n_range, lam_point)
        post_N = np.exp(log_post_N - log_post_N.max())
        post_N /= post_N.sum()

        tail_mass = float(post_N[-1])
        worst = max(worst, tail_mass)
        if tail_mass > tail_mass_threshold:
            flagged.append({
                "plot": g.plot, "year": g.year, "plant": g.plant, "poll": g.poll,
                "max_y": int(y.max()), "tail_mass": tail_mass,
            })

    result = {
        "n_groups_checked": len(groups),
        "n_flagged": len(flagged),
        "max_tail_mass": worst,
        "flagged": pd.DataFrame(flagged) if flagged else pd.DataFrame(
            columns=["plot", "year", "plant", "poll", "max_y", "tail_mass"]
        ),
    }
    log.info("Truncation check: %d/%d groups flagged (tail mass > %.0e); worst = %.2e",
              result["n_flagged"], result["n_groups_checked"], tail_mass_threshold, worst)
    return result


def _posterior_mean_lambda(y: np.ndarray, lam_grid: np.ndarray, p: float, n_trunc_extra: int) -> float:
    ell = log_marginal_over_lamgrid(y, lam_grid, p, n_trunc_extra)
    w = np.exp(ell - ell.max())
    w /= w.sum()
    return float(np.sum(w * lam_grid))


def _tail_mass_for_window(y: np.ndarray, lam_point: float, p: float, n_trunc_extra: int) -> float:
    from scipy.stats import binom, poisson
    n_max = int(y.max()) + n_trunc_extra
    n_range = np.arange(int(y.max()), n_max + 1)
    log_lik_sum = binom.logpmf(y[:, None], n_range[None, :], p).sum(axis=0)
    log_post_N = log_lik_sum + poisson.logpmf(n_range, lam_point)
    post_N = np.exp(log_post_N - log_post_N.max())
    post_N /= post_N.sum()
    return float(post_N[-1])


def recompute_lambda_hat_adaptive(
    groups: List[ReplicateGroup],
    p_hat: float,
    cfg: PipelineConfig = PipelineConfig(),
    tail_mass_threshold: float = 1e-6,
    max_trunc_extra: int = 20000,
    max_lam_ceiling: float = 1_000_000.0,
    growth_factor: float = 3.0,
) -> Tuple[Dict[Tuple, float], pd.DataFrame]:
    """Fixes R2 for real, on BOTH axes that can silently truncate the
    N-mixture posterior:

      (a) the N-truncation window [max(y_g), max(y_g)+n_trunc_extra] can be
          too narrow for high-count groups (originally found: 2,319/17,472
          groups had non-negligible tail mass with the fixed extra=30);

      (b) MORE IMPORTANTLY: cfg.lam_grid itself is capped at 80 by default.
          For very high-count groups (e.g. a single-watch count of 1,770),
          the true posterior over lambda wants mass well above 80, but
          there is nothing on the grid past 80 to put it on -- so the
          posterior mean gets silently PINNED at the grid ceiling. We
          found 507/17,472 groups (2.9%) with lambda_hat within 1% of the
          80.0 ceiling, ~100% of posterior mass in the single top grid
          cell for the worst cases. This artificially ties together the
          confidence scores of otherwise very different high-abundance
          interactions -- exactly the saturation problem the paper's
          Section 2.3 says log(1+lambda) was chosen to avoid, recurring
          at the top of the range instead of at presence/absence.

    This function iteratively (a) widens the N-truncation window and
    (b) widens the lambda grid ceiling, per group, until BOTH tail masses
    (at the N-window edge and at the lambda-grid ceiling) are below
    `tail_mass_threshold`, or a hard cap is hit.

    Returns
    -------
    lambda_hat : dict (plot, year, plant, poll) -> corrected posterior mean
    audit_df   : per-group record of what changed and why, for every group
                 that needed either correction.
    """
    lambda_hat: Dict[Tuple, float] = {}
    audit_rows = []
    base_lam_max = float(cfg.lam_grid.max())
    n_lam_points = len(cfg.lam_grid)

    for g in groups:
        y = np.asarray(g.y)
        if y.size == 0:
            lambda_hat[(g.plot, g.year, g.plant, g.poll)] = 0.0
            continue

        n_trunc = cfg.n_trunc_extra
        lam_grid = cfg.lam_grid
        lam_point = _posterior_mean_lambda(y, lam_grid, p_hat, n_trunc)
        lam_original = lam_point
        n_expansions_trunc = 0
        n_expansions_lam = 0

        for _ in range(40):  # hard safety cap on iterations
            tail_trunc = _tail_mass_for_window(y, lam_point, p_hat, n_trunc)
            ell = log_marginal_over_lamgrid(y, lam_grid, p_hat, n_trunc)
            w = np.exp(ell - ell.max())
            w /= w.sum()
            tail_lam_top = float(w[-1])

            grew = False
            if tail_trunc > tail_mass_threshold and n_trunc < max_trunc_extra:
                n_trunc = int(n_trunc * growth_factor)
                n_expansions_trunc += 1
                grew = True
            if tail_lam_top > tail_mass_threshold and lam_grid.max() < max_lam_ceiling:
                new_max = lam_grid.max() * growth_factor
                lam_grid = np.geomspace(cfg.lam_grid.min(), new_max, n_lam_points)
                n_expansions_lam += 1
                grew = True

            if not grew:
                break
            lam_point = _posterior_mean_lambda(y, lam_grid, p_hat, n_trunc)

        lambda_hat[(g.plot, g.year, g.plant, g.poll)] = lam_point

        if n_expansions_trunc > 0 or n_expansions_lam > 0:
            audit_rows.append({
                "plot": g.plot, "year": g.year, "plant": g.plant, "poll": g.poll,
                "max_y": int(y.max()),
                "lambda_hat_original": lam_original,
                "lambda_hat_corrected": lam_point,
                "pct_change": 100 * (lam_point - lam_original) / lam_original if lam_original > 0 else np.nan,
                "final_n_trunc_extra": n_trunc,
                "final_lam_ceiling": float(lam_grid.max()),
                "n_expansions_trunc": n_expansions_trunc,
                "n_expansions_lam_grid": n_expansions_lam,
                "was_pinned_at_original_ceiling": lam_original >= 0.99 * base_lam_max,
            })

    audit_df = pd.DataFrame(audit_rows)
    if len(audit_df):
        n_pinned = int(audit_df["was_pinned_at_original_ceiling"].sum())
        log.info(
            "R2 fix: %d/%d groups needed correction (%d were pinned at the "
            "original lambda-grid ceiling of %.0f). lambda_hat pct change "
            "for corrected groups: median %.1f%%, max %.1f%%",
            len(audit_df), len(groups), n_pinned, base_lam_max,
            audit_df["pct_change"].median(), audit_df["pct_change"].max(),
        )
    return lambda_hat, audit_df
