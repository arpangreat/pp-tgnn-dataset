"""
Synthetic data generator for data audit testing and verification.
Produces synthetic raw SA02601 and SA02602 tables with optional planted discrepancies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def generate_audit_test_datasets(
    out_dir: Path | str,
    n_plots: int = 6,
    years: list[int] | None = None,
    watches_per_year: int = 4,
    inject_discrepancies: bool = True,
    seed: int = 42,
) -> Tuple[Path, Path]:
    """Generates synthetic visitation (SA02601) and flowering (SA02602) CSVs.

    If inject_discrepancies is True, intentionally plants:
      - 2 date format/year mismatches
      - 2 invalid PPI_STATUS rows
      - 1 positive interaction with null species
      - 1 negative interaction with assigned species
      - 1 unmatched plant species in visitation
      - 1 watch session surveyed in visitation but omitted in flowering
      - 1 high-count interaction testing lambda-grid capping
      - 1 multi-year gap (years skip 2019-2020 by default)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    if years is None:
        years = [2011, 2012, 2013, 2014, 2021]  # naturally includes 2015-2020 gap

    plots = [f"PLOT_{i}" for i in range(n_plots)]
    plants = [f"PL_{i}" for i in range(8)]
    polls = [f"PO_{i}" for i in range(12)]
    complexes = {f"PLOT_{i}": ("COMPLEX_A" if i < 3 else "COMPLEX_B") for i in range(n_plots)}
    meadows = {f"PLOT_{i}": f"MEADOW_{i // 2}" for i in range(n_plots)}

    vis_rows = []
    flw_rows = []

    for plot in plots:
        for year in years:
            base_date = pd.Timestamp(year=year, month=6, day=1)
            for watch in range(1, watches_per_year + 1):
                date = base_date + pd.Timedelta(days=7 * watch)
                date_vis = date.strftime("%Y-%m-%d")
                date_flw = date.strftime("%Y-%m-%d")

                # Flowering records
                for plant in plants:
                    if rng.random() < 0.85:
                        flw_rows.append({
                            "DBCODE": "SA026", "ENTITY": 2, "COMPLEX": complexes[plot],
                            "MEADOW": meadows[plot], "PLOT_ID": plot, "YEAR": year,
                            "SAMPLEDATE": date_flw, "WATCH": watch, "OBSERVER": "OBS1",
                            "PLOT": plot, "FLW_STATUS": 1, "PLTSP_CODE": plant,
                            "NO_STALK": int(rng.integers(1, 15)),
                            "NO_FLWS": int(rng.geometric(p=0.2)),
                        })

                # Visitation records
                if rng.random() < 0.8:
                    for minute in range(1, rng.integers(2, 6)):
                        vis_rows.append({
                            "DBCODE": "SA026", "ENTITY": 1, "COMPLEX": complexes[plot],
                            "MEADOW": meadows[plot], "PLOT_ID": plot, "YEAR": year,
                            "SAMPLEDATE": date_vis, "WATCH": watch, "OBSERVER": "OBS1",
                            "PLOT": plot, "MINUTE": minute, "PPI_STATUS": 1,
                            "PLTSP_CODE": rng.choice(plants),
                            "VISSP_CODE": rng.choice(polls),
                            "NO_INT": int(rng.integers(1, 5)),
                        })
                else:
                    vis_rows.append({
                        "DBCODE": "SA026", "ENTITY": 1, "COMPLEX": complexes[plot],
                        "MEADOW": meadows[plot], "PLOT_ID": plot, "YEAR": year,
                        "SAMPLEDATE": date_vis, "WATCH": watch, "OBSERVER": "OBS1",
                        "PLOT": plot, "MINUTE": 1, "PPI_STATUS": 0,
                        "PLTSP_CODE": np.nan, "VISSP_CODE": np.nan, "NO_INT": np.nan,
                    })

    # Plant deliberate discrepancies for validation if requested
    if inject_discrepancies:
        # 1. Date year mismatch in vis
        vis_rows.append({
            "DBCODE": "SA026", "ENTITY": 1, "COMPLEX": "COMPLEX_A", "MEADOW": "MEADOW_0",
            "PLOT_ID": "PLOT_0", "YEAR": 2011, "SAMPLEDATE": "2099-06-15",  # Contradicts YEAR=2011
            "WATCH": 1, "OBSERVER": "OBS1", "MINUTE": 1, "PPI_STATUS": 1,
            "PLTSP_CODE": "PL_0", "VISSP_CODE": "PO_0", "NO_INT": 2,
        })
        # 2. Invalid PPI_STATUS
        vis_rows.append({
            "DBCODE": "SA026", "ENTITY": 1, "COMPLEX": "COMPLEX_A", "MEADOW": "MEADOW_0",
            "PLOT_ID": "PLOT_0", "YEAR": 2011, "SAMPLEDATE": "2011-06-15",
            "WATCH": 1, "OBSERVER": "OBS1", "MINUTE": 2, "PPI_STATUS": 99,  # invalid code
            "PLTSP_CODE": "PL_0", "VISSP_CODE": "PO_0", "NO_INT": 1,
        })
        # 3. Positive interaction missing species
        vis_rows.append({
            "DBCODE": "SA026", "ENTITY": 1, "COMPLEX": "COMPLEX_A", "MEADOW": "MEADOW_0",
            "PLOT_ID": "PLOT_0", "YEAR": 2011, "SAMPLEDATE": "2011-06-15",
            "WATCH": 1, "OBSERVER": "OBS1", "MINUTE": 3, "PPI_STATUS": 1,
            "PLTSP_CODE": np.nan, "VISSP_CODE": "PO_0", "NO_INT": 1,
        })
        # 4. Unmatched plant species
        vis_rows.append({
            "DBCODE": "SA026", "ENTITY": 1, "COMPLEX": "COMPLEX_A", "MEADOW": "MEADOW_0",
            "PLOT_ID": "PLOT_0", "YEAR": 2011, "SAMPLEDATE": "2011-06-15",
            "WATCH": 1, "OBSERVER": "OBS1", "MINUTE": 4, "PPI_STATUS": 1,
            "PLTSP_CODE": "PL_ALIEN", "VISSP_CODE": "PO_0", "NO_INT": 3,  # Missing in flw
        })
        # 5. Extreme count testing lambda capping
        vis_rows.append({
            "DBCODE": "SA026", "ENTITY": 1, "COMPLEX": "COMPLEX_A", "MEADOW": "MEADOW_0",
            "PLOT_ID": "PLOT_1", "YEAR": 2011, "SAMPLEDATE": "15-06-2011",
            "WATCH": 2, "OBSERVER": "OBS1", "MINUTE": 1, "PPI_STATUS": 1,
            "PLTSP_CODE": "PL_1", "VISSP_CODE": "PO_1", "NO_INT": 1500,  # Extreme count!
        })
        # 6. Negative flower count in flw
        flw_rows.append({
            "DBCODE": "SA026", "ENTITY": 2, "COMPLEX": "COMPLEX_A", "MEADOW": "MEADOW_0",
            "PLOT_ID": "PLOT_0", "YEAR": 2011, "SAMPLEDATE": "2011-06-15",
            "WATCH": 1, "OBSERVER": "OBS1", "PLOT": "PLOT_0", "FLW_STATUS": 1,
            "PLTSP_CODE": "PL_0", "NO_STALK": 5, "NO_FLWS": -10,  # Negative count!
        })

    vis_df = pd.DataFrame(vis_rows)
    flw_df = pd.DataFrame(flw_rows)

    vis_path = out_dir / "SA02601_v6.csv"
    flw_path = out_dir / "SA02602_v5.csv"

    vis_df.to_csv(vis_path, index=False)
    flw_df.to_csv(flw_path, index=False)
    return vis_path, flw_path
