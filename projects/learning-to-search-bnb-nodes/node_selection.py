from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class NodeState:
    depth: int
    lp_bound: float
    incumbent: float
    fractional_count: int
    fixed_count: int


def node_features(state: NodeState) -> np.ndarray:
    incumbent = state.incumbent if np.isfinite(state.incumbent) else 0.0
    gap = state.lp_bound - incumbent
    return np.array(
        [
            float(state.depth),
            float(state.lp_bound),
            float(incumbent),
            float(gap),
            float(state.fractional_count),
            float(state.fixed_count),
        ],
        dtype=np.float32,
    )


class NodePriorityNet(nn.Module):
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def fit_priority_model(
    features: np.ndarray,
    expert_priority: np.ndarray,
    *,
    epochs: int = 100,
    learning_rate: float = 1e-2,
    hidden_dim: int = 32,
    seed: int = 0,
) -> NodePriorityNet:
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("invalid training configuration")
    x = torch.tensor(np.asarray(features), dtype=torch.float32)
    y = torch.tensor(np.asarray(expert_priority), dtype=torch.float32)
    if x.ndim != 2 or x.shape[1] != 6 or y.shape != (x.shape[0],):
        raise ValueError("expected features [n,6] and priorities [n]")
    torch.manual_seed(seed)
    model = NodePriorityNet(hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    return model.eval()


def rank_nodes(model: NodePriorityNet, states: list[NodeState]) -> np.ndarray:
    if not states:
        return np.empty(0, dtype=int)
    x = torch.tensor(np.stack([node_features(s) for s in states]), dtype=torch.float32)
    with torch.no_grad():
        scores = model(x).cpu().numpy()
    return np.argsort(scores)
