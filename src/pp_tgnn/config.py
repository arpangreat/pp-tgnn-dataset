"""Configuration for the preprocessing pipeline.

Column names default to the actual H.J. Andrews LTER raw schema
(SA02601_v6.csv / SA02602_v5.csv). Override via `PipelineConfig(...)`
if you're adapting the pipeline to a structurally similar dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class PipelineConfig:
    # --- raw column names ---
    vis_date_col: str = "SAMPLEDATE"
    vis_date_fmt: str = "%d-%m-%Y"          # SA02601: DD-MM-YYYY
    flw_date_col: str = "SAMPLEDATE"
    flw_date_fmt: str = "%Y-%m-%d"          # SA02602: YYYY-MM-DD

    plant_col: str = "PLTSP_CODE"
    poll_col: str = "VISSP_CODE"            # pollinator species code
    plot_col: str = "PLOT_ID"
    year_col: str = "YEAR"
    watch_col: str = "WATCH"
    no_int_col: str = "NO_INT"
    no_flws_col: str = "NO_FLWS"
    ppi_status_col: str = "PPI_STATUS"

    # --- N-mixture fitting grids (Algorithm 4) ---
    p_grid: np.ndarray = field(default_factory=lambda: np.linspace(0.05, 0.95, 19))
    # Widened from 80 -> 5000 (and 150 -> 300 grid points). The original
    # ceiling silently capped lambda_hat for high-abundance groups: the raw
    # data contains single-watch counts up to 1,770, and a candidate-lambda
    # grid that doesn't reach far enough will pin the posterior mean at
    # whatever the grid's top value happens to be, rather than the true
    # value. This affected 382/17,472 groups (2.2%) at the original ceiling.
    # See docs/data_dictionary.md "Known structural pitfalls" #3.
    lam_grid: np.ndarray = field(default_factory=lambda: np.geomspace(0.01, 5000, 300))
    n_trunc_extra: int = 100  # widened from 30 alongside the lambda-grid fix above
    max_informative_sample: int = 250
    rng_seed: int = 42

    # --- chronological split fractions (Algorithm 8) ---
    train_frac: float = 0.8
    val_frac: float = 0.1

    @property
    def key_cols(self) -> Tuple[str, str, str, str, str]:
        return (self.plot_col, self.year_col, self.plant_col, self.poll_col, self.watch_col)
