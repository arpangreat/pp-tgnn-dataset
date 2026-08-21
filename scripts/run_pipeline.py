#!/usr/bin/env python
"""CLI entry point: run the full preprocessing pipeline (Algorithms 1-8).

Usage:
    python scripts/run_pipeline.py \\
        --vis data/raw/SA02601_v6.csv \\
        --flw data/raw/SA02602_v5.csv \\
        --outdir data/processed
"""

import argparse
import logging
from pathlib import Path

from pp_tgnn.config import PipelineConfig
from pp_tgnn.pipeline import run_pipeline


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vis", type=Path, required=True, help="Path to SA02601_v6.csv")
    parser.add_argument("--flw", type=Path, required=True, help="Path to SA02602_v5.csv")
    parser.add_argument("--outdir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    run_pipeline(args.vis, args.flw, args.outdir, cfg=PipelineConfig())


if __name__ == "__main__":
    main()
