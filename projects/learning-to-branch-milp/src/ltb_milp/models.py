"""Neural branching score model."""

from __future__ import annotations

from torch import nn


class BranchingMLP(nn.Module):
    """Predict a strong-branching score from candidate-level features."""

    def __init__(self, n_features: int = 6, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)
