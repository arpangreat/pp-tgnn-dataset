"""Shared pytest fixtures: small synthetic raw data mirroring the real
SA02601 / SA02602 schema, so the test suite runs without the actual LTER
files (which are not distributed with this repo)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_raw_dir(tmp_path_factory) -> Path:
    rng = np.random.default_rng(0)

    n_plots = 6
    years = [2011, 2012, 2013, 2014]
    watches_per_year = 4
    n_plants = 8
    n_polls = 15

    plots = [f"P{i}" for i in range(n_plots)]
    plants = [f"PL{i}" for i in range(n_plants)]
    polls = [f"PO{i}" for i in range(n_polls)]

    vis_rows, flw_rows = [], []

    for plot in plots:
        for year in years:
            base_date = pd.Timestamp(year=year, month=6, day=1)
            for watch in range(1, watches_per_year + 1):
                date = base_date + pd.Timedelta(days=10 * watch)
                date_vis = date.strftime("%d-%m-%Y")   # DD-MM-YYYY, matches SA02601
                date_flw = date.strftime("%Y-%m-%d")   # YYYY-MM-DD, matches SA02602

                for plant in plants:
                    if rng.random() < 0.9:
                        flw_rows.append({
                            "SAMPLEDATE": date_flw, "YEAR": year, "PLOT_ID": plot,
                            "WATCH": watch, "PLTSP_CODE": plant,
                            "NO_STALK": rng.integers(0, 20),
                            "NO_FLWS": int(rng.lognormal(mean=1.0, sigma=1.2)),
                        })

                if rng.random() < 0.8:
                    for _ in range(rng.integers(1, 6)):
                        vis_rows.append({
                            "SAMPLEDATE": date_vis, "YEAR": year, "PLOT_ID": plot,
                            "WATCH": watch, "PLTSP_CODE": rng.choice(plants),
                            "VISSP_CODE": rng.choice(polls),
                            "NO_INT": rng.integers(1, 4), "PPI_STATUS": 1,
                        })
                else:
                    vis_rows.append({
                        "SAMPLEDATE": date_vis, "YEAR": year, "PLOT_ID": plot,
                        "WATCH": watch, "PLTSP_CODE": np.nan, "VISSP_CODE": np.nan,
                        "NO_INT": np.nan, "PPI_STATUS": 0,
                    })

    out_dir = tmp_path_factory.mktemp("synthetic_raw")
    vis_path = out_dir / "SA02601_v6_synthetic.csv"
    flw_path = out_dir / "SA02602_v5_synthetic.csv"
    pd.DataFrame(vis_rows).to_csv(vis_path, index=False)
    pd.DataFrame(flw_rows).to_csv(flw_path, index=False)
    return out_dir


@pytest.fixture(scope="session")
def synthetic_vis_path(synthetic_raw_dir) -> Path:
    return synthetic_raw_dir / "SA02601_v6_synthetic.csv"


@pytest.fixture(scope="session")
def synthetic_flw_path(synthetic_raw_dir) -> Path:
    return synthetic_raw_dir / "SA02602_v5_synthetic.csv"
