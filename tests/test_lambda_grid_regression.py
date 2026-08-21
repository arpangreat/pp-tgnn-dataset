"""Regression test for the lambda-grid-ceiling bug found during the data-paper
audit: config.PipelineConfig.lam_grid must extend well beyond the highest
observed single-watch count divided by the fitted detection probability, or
posterior-mean lambda_hat for high-abundance groups gets silently pinned at
the grid's upper limit instead of reflecting their true value.

See docs/data_dictionary.md "Known structural pitfalls" #3 for the full
writeup: with the original default ceiling of 80, 382/17,472 real-data groups
(2.2%) were pinned, understating their true lambda_hat by up to ~39x.
"""

import numpy as np
import pytest

from pp_tgnn.config import PipelineConfig
from pp_tgnn.pipeline import log_marginal_over_lamgrid


def _posterior_mean_lambda(y, lam_grid, p, n_trunc_extra):
    ell = log_marginal_over_lamgrid(y, lam_grid, p, n_trunc_extra)
    w = np.exp(ell - ell.max())
    w /= w.sum()
    return float(np.sum(w * lam_grid))


def test_default_lam_grid_does_not_pin_high_count_group():
    """A single-watch count as high as observed in the real raw data (up to
    1,770) should NOT produce a posterior mean sitting at the grid ceiling.
    This is the exact failure mode: if this test starts failing after future
    edits to config.py, the lam_grid ceiling has regressed to being too
    narrow for the data this pipeline is meant to process."""
    cfg = PipelineConfig()
    y_high_count = np.array([0.0, 0.0, 27.0, 1770.0, 0.0])  # mirrors a real flagged group

    lam_hat = _posterior_mean_lambda(y_high_count, cfg.lam_grid, p=0.15, n_trunc_extra=cfg.n_trunc_extra)

    assert lam_hat < 0.99 * cfg.lam_grid.max(), (
        f"lambda_hat ({lam_hat:.1f}) is pinned at (or within 1% of) the lam_grid "
        f"ceiling ({cfg.lam_grid.max():.1f}) -- the grid is too narrow for this "
        f"count. Widen PipelineConfig.lam_grid's upper bound."
    )


def test_lam_grid_ceiling_covers_worst_real_case_with_margin():
    """Sanity check on the config default itself, calibrated against the
    actual worst real-data case found during the audit (plot FM209, 2011,
    GILICAPI-EPICPUNC, single-watch count of 1,770): posterior mean
    lambda_hat for that group is ~1,870, with ~0 posterior weight left on
    the top grid cell. NOTE: a naive count/p_hat heuristic (1770/0.15 ~=
    11,800) overestimates the required ceiling -- the Bayesian posterior
    mean is pulled below the raw MLE-style ratio by the Poisson prior, so
    that heuristic is NOT a reliable substitute for checking the actual
    posterior weight at the grid edge (which is what
    test_default_lam_grid_does_not_pin_high_count_group does directly)."""
    cfg = PipelineConfig()
    known_worst_case_posterior_mean = 1870.6
    margin = 1.5  # require at least 50% headroom above the known worst case

    assert cfg.lam_grid.max() >= margin * known_worst_case_posterior_mean, (
        f"lam_grid ceiling ({cfg.lam_grid.max():.0f}) has less than {margin}x "
        f"headroom above the worst real-data posterior mean found during audit "
        f"({known_worst_case_posterior_mean:.0f}). Re-verify with "
        f"test_default_lam_grid_does_not_pin_high_count_group before shrinking."
    )


@pytest.mark.parametrize("max_count", [50, 200, 1000, 1770])
def test_posterior_mean_scales_with_count_not_capped(max_count):
    """As the observed count increases, the posterior mean lambda_hat should
    increase roughly proportionally -- not flatten out because it hit a
    grid ceiling. This directly targets the "different high-abundance pairs
    all get squashed to the same confidence" failure mode."""
    cfg = PipelineConfig()
    y = np.array([0.0, 0.0, max_count, 0.0])
    lam_hat = _posterior_mean_lambda(y, cfg.lam_grid, p=0.15, n_trunc_extra=cfg.n_trunc_extra)
    # loose sanity bound: posterior mean should be within an order of
    # magnitude of count/p_hat, not stuck near a fixed ceiling regardless
    # of how large max_count is
    expected_order_of_magnitude = max_count / 0.15
    assert lam_hat > 0.1 * expected_order_of_magnitude
