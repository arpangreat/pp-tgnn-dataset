"""
TGNNLite — a compact, dependency-light memory-based Temporal Graph Network
(JODIE/TGN-style) for dynamic plant-pollinator link prediction, operating
directly on the (tau, plant_id, pollinator_id, c) event stream produced by
`pp_tgnn.pipeline.run_pipeline`.

For a production-grade TGN with full raw-message stores and multi-head
temporal attention over sampled neighborhoods, see the official TGN codebase
(github.com/twitter-research/tgn) or PyTorch Geometric's
`torch_geometric.nn.models.TGNMemory` — this module's event columns map
directly: tau -> t, plant_id/pollinator_id -> src/dst, c -> edge feature.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TimeEncode(nn.Module):
    """Functional time encoding (Time2Vec-style): cos(w*t + b)."""

    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Linear(1, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1).float()
        return torch.cos(self.w(t))


class TGNNLite(nn.Module):
    """JODIE/TGN-lite: per-node GRU memory updated at each interaction,
    time-aware MLP link predictor.

    Args:
        n_nodes: total number of nodes (plants + pollinators, shared index space).
        mem_dim: memory vector dimension per node.
        time_dim: dimension of the functional time encoding.
    """

    def __init__(self, n_nodes: int, mem_dim: int = 64, time_dim: int = 16):
        super().__init__()
        self.mem_dim = mem_dim
        self.time_enc = TimeEncode(time_dim)
        self.gru = nn.GRUCell(mem_dim + time_dim + 1, mem_dim)  # +1 for edge weight c
        self.predictor = nn.Sequential(
            nn.Linear(2 * mem_dim + time_dim, mem_dim),
            nn.ReLU(),
            nn.Linear(mem_dim, 1),
        )
        self.register_buffer("memory", torch.zeros(n_nodes, mem_dim))
        self.register_buffer("last_update", torch.zeros(n_nodes))

    def reset_memory(self) -> None:
        self.memory.zero_()
        self.last_update.zero_()

    def score(self, u: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Score candidate edges (u, v) at time t. Returns raw logits."""
        dt_u = (t - self.last_update[u]).clamp(min=0)
        te = self.time_enc(dt_u)
        z = torch.cat([self.memory[u], self.memory[v], te], dim=-1)
        return self.predictor(z).squeeze(-1)

    def update(self, u: torch.Tensor, v: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> None:
        """Advance memory state given a batch of true interactions (u, v, t, c).
        Must be called in strictly non-decreasing time order to avoid leakage.
        """
        dt_u = (t - self.last_update[u]).clamp(min=0)
        dt_v = (t - self.last_update[v]).clamp(min=0)
        te_u = self.time_enc(dt_u)
        te_v = self.time_enc(dt_v)

        msg_u = torch.cat([self.memory[v], te_u, c.unsqueeze(-1)], dim=-1)
        msg_v = torch.cat([self.memory[u], te_v, c.unsqueeze(-1)], dim=-1)

        new_mem_u = self.gru(msg_u, self.memory[u])
        new_mem_v = self.gru(msg_v, self.memory[v])

        self.memory[u] = new_mem_u.detach()
        self.memory[v] = new_mem_v.detach()
        self.last_update[u] = t.detach()
        self.last_update[v] = t.detach()
