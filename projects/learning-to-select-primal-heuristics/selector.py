from __future__ import annotations

import numpy as np
import torch
from torch import nn

from lamip.problem import BinaryPackingMIP, generate_binary_packing

HEURISTICS = ("profit", "density", "sparsity")


def _variable_scores(problem: BinaryPackingMIP, kind: str) -> np.ndarray:
    if kind == "profit":
        return problem.profits.copy()
    mean_use = np.maximum(problem.A.mean(axis=0), 1e-6)
    if kind == "density":
        return problem.profits / mean_use
    if kind == "sparsity":
        nonzero = np.maximum(np.count_nonzero(problem.A, axis=0), 1)
        return problem.profits / nonzero
    raise ValueError(f"unknown heuristic: {kind}")


def run_heuristic(problem: BinaryPackingMIP, kind: str) -> tuple[np.ndarray, float]:
    order = np.argsort(-_variable_scores(problem, kind))
    x = np.zeros(problem.n_vars)
    activity = np.zeros(problem.n_constraints)
    for j in order:
        if problem.profits[j] <= 0:
            continue
        candidate = activity + problem.A[:, j]
        if np.all(candidate <= problem.b + 1e-10):
            x[j] = 1.0
            activity = candidate
    return x, float(problem.profits @ x)


def instance_features(problem: BinaryPackingMIP) -> np.ndarray:
    capacity_ratio = problem.b / np.maximum(problem.A.sum(axis=1), 1e-9)
    return np.array(
        [
            problem.n_vars,
            problem.n_constraints,
            float(np.count_nonzero(problem.A) / problem.A.size),
            float(problem.profits.mean()),
            float(problem.profits.std()),
            float(problem.A.mean()),
            float(problem.A.std()),
            float(capacity_ratio.mean()),
        ],
        dtype=np.float32,
    )


class HeuristicSelector(nn.Module):
    def __init__(self, hidden_dim: int = 24) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(HEURISTICS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def collect_training_data(
    *,
    n_instances: int = 80,
    n_vars: int = 18,
    n_constraints: int = 6,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for offset in range(n_instances):
        problem = generate_binary_packing(n_vars, n_constraints, seed + offset)
        values = [run_heuristic(problem, h)[1] for h in HEURISTICS]
        X.append(instance_features(problem))
        y.append(int(np.argmax(values)))
    return np.stack(X), np.asarray(y, dtype=np.int64)


def train_selector(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 120,
    learning_rate: float = 1e-2,
    hidden_dim: int = 24,
    seed: int = 0,
) -> HeuristicSelector:
    x = torch.tensor(np.asarray(features), dtype=torch.float32)
    y = torch.tensor(np.asarray(labels), dtype=torch.long)
    if x.ndim != 2 or x.shape[1] != 8 or y.shape != (x.shape[0],):
        raise ValueError("expected features [n,8] and labels [n]")
    torch.manual_seed(seed)
    model = HeuristicSelector(hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    return model.eval()


def choose_heuristic(model: HeuristicSelector, problem: BinaryPackingMIP) -> str:
    x = torch.tensor(instance_features(problem)).unsqueeze(0)
    with torch.no_grad():
        idx = int(torch.argmax(model(x), dim=1).item())
    return HEURISTICS[idx]
