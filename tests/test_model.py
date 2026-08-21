"""Tests for pp_tgnn.model and pp_tgnn.train — sanity-check the TGNNLite
model's forward pass, memory update, and one training epoch on tiny
synthetic event data (no real dataset needed)."""

import numpy as np
import pandas as pd
import torch

from pp_tgnn.model import TGNNLite, TimeEncode
from pp_tgnn.train import build_node_vocab, encode_events, mrr_eval, run_epoch


def _toy_events(n_plants=5, n_polls=8, n_events=60, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    plants = [f"PL{i}" for i in range(n_plants)]
    polls = [f"PO{i}" for i in range(n_polls)]
    df = pd.DataFrame({
        "plant_id": rng.choice(plants, n_events),
        "pollinator_id": rng.choice(polls, n_events),
        "tau": np.sort(rng.integers(0, 500, n_events)),
        "c": rng.uniform(0.01, 1.0, n_events),
    })
    df["date"] = pd.Timestamp("2011-01-01") + pd.to_timedelta(df["tau"], unit="D")
    return df


def test_time_encode_output_shape():
    enc = TimeEncode(dim=16)
    t = torch.tensor([0.0, 1.0, 100.0])
    out = enc(t)
    assert out.shape == (3, 16)


def test_tgnnlite_score_and_update_shapes():
    model = TGNNLite(n_nodes=20, mem_dim=8, time_dim=4)
    u = torch.tensor([0, 1, 2])
    v = torch.tensor([10, 11, 12])
    t = torch.tensor([0.0, 1.0, 2.0])
    c = torch.tensor([0.5, 0.3, 0.9])

    scores = model.score(u, v, t)
    assert scores.shape == (3,)

    model.update(u, v, t, c)
    assert not torch.allclose(model.memory[0], torch.zeros(8))  # memory changed


def test_tgnnlite_memory_reset():
    model = TGNNLite(n_nodes=10, mem_dim=8, time_dim=4)
    u, v = torch.tensor([0]), torch.tensor([5])
    t, c = torch.tensor([1.0]), torch.tensor([0.5])
    model.update(u, v, t, c)
    assert not torch.allclose(model.memory, torch.zeros_like(model.memory))
    model.reset_memory()
    assert torch.allclose(model.memory, torch.zeros_like(model.memory))
    assert torch.allclose(model.last_update, torch.zeros_like(model.last_update))


def test_run_epoch_trains_without_error():
    events = _toy_events()
    plant2idx, poll2idx, n_plants, n_polls = build_node_vocab(events)
    events = encode_events(events, plant2idx, poll2idx)

    device = torch.device("cpu")
    model = TGNNLite(n_nodes=n_plants + n_polls, mem_dim=8, time_dim=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    model.reset_memory()
    loss, ap, auc = run_epoch(model, events, n_plants, n_polls, device, opt=opt, batch_size=10)

    assert np.isfinite(loss)
    assert 0.0 <= ap <= 1.0
    assert 0.0 <= auc <= 1.0


def test_mrr_eval_in_valid_range():
    events = _toy_events(n_events=40)
    plant2idx, poll2idx, n_plants, n_polls = build_node_vocab(events)
    events = encode_events(events, plant2idx, poll2idx)

    device = torch.device("cpu")
    model = TGNNLite(n_nodes=n_plants + n_polls, mem_dim=8, time_dim=4).to(device)
    model.reset_memory()
    with torch.no_grad():
        run_epoch(model, events, n_plants, n_polls, device, opt=None, batch_size=10)

    mrr = mrr_eval(model, events, n_plants, n_polls, device, n_neg=5, n_samples=10)
    assert 0.0 < mrr <= 1.0
