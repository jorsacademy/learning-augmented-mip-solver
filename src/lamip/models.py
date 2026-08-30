from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PresolveSelector(MLP):
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__(8, 4, hidden_dim)


class CutScorer(MLP):
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__(6, 1, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).squeeze(-1)


class BranchScorer(MLP):
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__(6, 1, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).squeeze(-1)


class PrimalScorer(MLP):
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__(5, 1, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).squeeze(-1)


@dataclass
class LearnedComponents:
    presolve: PresolveSelector | None = None
    cuts: CutScorer | None = None
    branching: BranchScorer | None = None
    primal: PrimalScorer | None = None

    def eval(self) -> "LearnedComponents":
        for model in (self.presolve, self.cuts, self.branching, self.primal):
            if model is not None:
                model.eval()
        return self
