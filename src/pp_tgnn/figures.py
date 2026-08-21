"""Reproduce Figure 1 of the paper: (a) events per year, (b) raw N-mixture
presence probability vs. normalized log-lambda confidence."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALL_YEARS = list(range(2011, 2022))
NO_SURVEY_YEARS = {2019, 2020}


def _plot_events_per_year(ax: plt.Axes, events: pd.DataFrame) -> None:
    ev_years = pd.to_datetime(events["date"]).dt.year
    counts = ev_years.value_counts().reindex(ALL_YEARS, fill_value=0)

    colors = ["#d9d9d9" if y in NO_SURVEY_YEARS else "#1f77b4" for y in ALL_YEARS]
    bars = ax.bar(range(len(ALL_YEARS)), counts.values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, y in zip(bars, ALL_YEARS):
        if y in NO_SURVEY_YEARS:
            bar.set_hatch("//")

    ax.set_xticks(range(len(ALL_YEARS)))
    ax.set_xticklabels(ALL_YEARS, rotation=45, ha="right")
    ax.set_ylabel("Interaction events")
    ax.set_title("(a) Events per year\n(grey = no survey conducted)")


def _plot_confidence_comparison(
    ax: plt.Axes, lambda_hat: np.ndarray, confidence_posterior: np.ndarray
) -> None:
    presence_prob = 1 - np.exp(-lambda_hat)
    bins = np.linspace(0, 1, 26)

    ax.hist(presence_prob, bins=bins, alpha=0.65, color="#ff7f0e",
            label=r"$P(N>0)=1-\exp(-\lambda)$" + "\n(saturates, unusable)")
    ax.hist(confidence_posterior, bins=bins, alpha=0.65, color="#1f77b4",
            label=r"normalized $\log(1+\lambda)$" + "\n(used as edge weight)")

    ax.set_xlabel("Confidence value")
    ax.set_ylabel("Number of (plot,year,plant,pollinator) groups")
    ax.set_title("(b) Why we use normalized log-λ,\nnot raw presence probability")
    ax.legend(fontsize=8, loc="upper left")


def make_figure1(
    events: pd.DataFrame,
    lambda_hat_df: pd.DataFrame,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """Reproduce Figure 1. `lambda_hat_df` must have a `lambda_hat` column
    (as written by pp_tgnn.pipeline.run_pipeline to lambda_hat.csv)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    _plot_events_per_year(axes[0], events)
    _plot_confidence_comparison(
        axes[1], lambda_hat_df["lambda_hat"].to_numpy(), events["confidence_posterior"].to_numpy()
    )
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig
