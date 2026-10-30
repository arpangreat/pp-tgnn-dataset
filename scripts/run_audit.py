#!/usr/bin/env python
"""CLI entry point: Audit and log every data discrepancy in SA02601 and SA02602 datasets.

Usage:
    # Run on real raw datasets
    python scripts/run_audit.py \\
        --vis data/raw/SA02601_v6.csv \\
        --flw data/raw/SA02602_v5.csv \\
        --outdir data_discrepancy_audit

    # Run on synthetic dataset with intentional discrepancies planted
    python scripts/run_audit.py --synthetic --outdir data_discrepancy_audit
"""

import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_audit.audit import main

if __name__ == "__main__":
    main()
