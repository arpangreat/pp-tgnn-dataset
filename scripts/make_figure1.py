#!/usr/bin/env python
"""CLI entry point: reproduce Figure 1 from the processed dataset.

Usage:
    python scripts/make_figure1.py \\
        --processed_dir data/processed \\
        --out figures/figure1.png
"""

import argparse
from pathlib import Path

import pandas as pd

from pp_tgnn.figures import make_figure1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("figures/figure1.png"))
    args = parser.parse_args()

    events = pd.read_csv(args.processed_dir / "plant_pollinator_events.csv")
    lambda_hat_df = pd.read_csv(args.processed_dir / "lambda_hat.csv")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    make_figure1(events, lambda_hat_df, out_path=args.out)
    print(f"Saved figure to {args.out.resolve()}")


if __name__ == "__main__":
    main()
