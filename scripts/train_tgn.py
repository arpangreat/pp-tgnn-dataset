#!/usr/bin/env python
"""CLI entry point: train and evaluate TGNNLite for dynamic link prediction.

Usage:
    python scripts/train_tgn.py \\
        --processed_dir data/processed \\
        --epochs 5 \\
        --out_dir data/processed
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pp_tgnn.model import TGNNLite
from pp_tgnn.train import build_node_vocab, encode_events, run_epoch, mrr_eval

log = logging.getLogger("train_tgn")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--mem_dim", type=int, default=64)
    parser.add_argument("--time_dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mrr_neg", type=int, default=50)
    parser.add_argument("--mrr_samples", type=int, default=300)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    events = pd.read_csv(args.processed_dir / "plant_pollinator_events.csv")
    train = pd.read_csv(args.processed_dir / "events_train.csv")
    val = pd.read_csv(args.processed_dir / "events_val.csv")
    test = pd.read_csv(args.processed_dir / "events_test.csv")

    plant2idx, poll2idx, n_plants, n_polls = build_node_vocab(events)
    train = encode_events(train, plant2idx, poll2idx)
    val = encode_events(val, plant2idx, poll2idx)
    test = encode_events(test, plant2idx, poll2idx)
    n_nodes = n_plants + n_polls
    log.info("n_plants=%d n_polls=%d n_nodes=%d", n_plants, n_polls, n_nodes)

    model = TGNNLite(n_nodes, mem_dim=args.mem_dim, time_dim=args.time_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.reset_memory()
        loss, ap, auc = run_epoch(model, train, n_plants, n_polls, device, opt=opt, batch_size=args.batch_size)
        log.info("epoch %d: train loss=%.4f AP=%.4f AUC=%.4f", epoch, loss, ap, auc)

    # Warm up memory to end-of-train state, then evaluate chronologically forward
    model.reset_memory()
    with torch.no_grad():
        run_epoch(model, train, n_plants, n_polls, device, opt=None, batch_size=args.batch_size)

    val_loss, val_ap, val_auc = run_epoch(model, val, n_plants, n_polls, device, opt=None, batch_size=args.batch_size)
    log.info("val:  loss=%.4f AP=%.4f AUC=%.4f", val_loss, val_ap, val_auc)

    test_loss, test_ap, test_auc = run_epoch(model, test, n_plants, n_polls, device, opt=None, batch_size=args.batch_size)
    log.info("test: loss=%.4f AP=%.4f AUC=%.4f", test_loss, test_ap, test_auc)

    test_mrr = mrr_eval(model, test, n_plants, n_polls, device, n_neg=args.mrr_neg, n_samples=args.mrr_samples, seed=args.seed)
    chance_mrr = float(np.mean(1.0 / np.arange(1, args.mrr_neg + 2)))
    log.info("test MRR (%d random negatives): %.4f  (chance-level: %.4f)", args.mrr_neg, test_mrr, chance_mrr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out_dir / "tgnn_lite_state_dict.pt")
    log.info("Saved model weights to %s", (args.out_dir / "tgnn_lite_state_dict.pt").resolve())


if __name__ == "__main__":
    main()
