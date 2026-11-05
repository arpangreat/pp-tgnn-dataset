"""
Corrected TGNN pipeline components:
1. Leakage-free calendar-date splitting
2. Hiatus-aware memory decay
3. Grounded 3-tier negative candidate exporter
4. Confidence-weighted training loop
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import torch

from .audit_negatives import materialize_grounded_structural_negatives
from .audit_splits import generate_leakage_free_splits
from pp_tgnn.train import build_node_vocab

log = logging.getLogger("tgnn_discrepancy.corrected")


def build_and_export_corrected_dataset(
    events_path: Path | str,
    flw_conf_path: Path | str,
    negative_events_path: Path | str,
    out_dir: Path | str,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Dict[str, Any]:
    """Generates strictly leakage-free splits and grounded negative benchmark files."""
    events_path = Path(events_path)
    flw_conf_path = Path(flw_conf_path)
    negative_events_path = Path(negative_events_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading processed files from %s...", events_path.parent)
    events = pd.read_csv(events_path)
    flw_conf = pd.read_csv(flw_conf_path)
    negative_events = pd.read_csv(negative_events_path)

    log.info("Generating leakage-free chronological splits...")
    train_clean, val_clean, test_clean, split_info = generate_leakage_free_splits(
        events, train_frac=train_frac, val_frac=val_frac
    )

    train_out = out_dir / "events_train_clean.csv"
    val_out = out_dir / "events_val_clean.csv"
    test_out = out_dir / "events_test_clean.csv"

    train_clean.to_csv(train_out, index=False)
    val_clean.to_csv(val_out, index=False)
    test_clean.to_csv(test_out, index=False)
    log.info("Exported clean splits: train=%d, val=%d, test=%d", len(train_clean), len(val_clean), len(test_clean))

    log.info("Materializing grounded structural negative candidates...")
    grounded_negs = materialize_grounded_structural_negatives(
        negative_events, flw_conf, events, seed=42
    )
    negs_out = out_dir / "grounded_structural_negatives.csv"
    grounded_negs.to_csv(negs_out, index=False)
    log.info("Exported %d grounded negative candidate edges to %s", len(grounded_negs), negs_out)

    return {
        "split_info": split_info,
        "train_file": str(train_out),
        "val_file": str(val_out),
        "test_file": str(test_out),
        "grounded_negatives_file": str(negs_out),
        "n_grounded_negatives": len(grounded_negs),
    }
