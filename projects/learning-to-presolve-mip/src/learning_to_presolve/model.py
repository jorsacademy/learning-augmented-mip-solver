from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .core import CONFIGS, PresolveConfig, generate_packing, instance_features, oracle_configuration

CONFIG_TO_INDEX = {config: index for index, config in enumerate(CONFIGS)}
N_FEATURES = 12


class PresolveSelector(nn.Module):
    """Small MLP that selects a safe presolve configuration from instance features."""

    def __init__(self, hidden_dim: int = 48) -> None:
        super().__init__()
        if hidden_dim < 4:
            raise ValueError("hidden_dim must be at least four")
        self.register_buffer("feature_mean", torch.zeros(N_FEATURES))
        self.register_buffer("feature_scale", torch.ones(N_FEATURES))
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(CONFIGS)),
        )

    def set_normalization(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != N_FEATURES:
            raise ValueError("features have wrong shape")
        mean = features.mean(dim=0)
        scale = features.std(dim=0, unbiased=False).clamp_min(1e-6)
        self.feature_mean.copy_(mean)
        self.feature_scale.copy_(scale)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_scale
        return self.net(normalized)


@dataclass(frozen=True)
class PresolveDataset:
    features: torch.Tensor
    labels: torch.Tensor
    oracle_scores: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.labels.numel())


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    mean_normalized_regret: float


def collect_dataset(
    n_instances: int = 160,
    n_vars: int = 32,
    n_constraints: int = 10,
    seed: int = 0,
) -> PresolveDataset:
    if n_instances < 1:
        raise ValueError("n_instances must be positive")
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    score_rows: list[np.ndarray] = []
    for offset in range(n_instances):
        problem = generate_packing(n_vars=n_vars, n_constraints=n_constraints, seed=seed + offset)
        oracle = oracle_configuration(problem)
        feature_rows.append(instance_features(problem))
        labels.append(CONFIG_TO_INDEX[oracle.selected])
        score_rows.append(np.asarray([item.work_proxy for item in oracle.evaluations], dtype=np.float32))
    return PresolveDataset(
        features=torch.tensor(np.stack(feature_rows), dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.long),
        oracle_scores=torch.tensor(np.stack(score_rows), dtype=torch.float32),
    )


def class_weights(dataset: PresolveDataset) -> torch.Tensor:
    counts = torch.bincount(dataset.labels, minlength=len(CONFIGS)).to(torch.float32)
    present = counts > 0
    weights = torch.ones_like(counts)
    if torch.any(present):
        weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return weights


def train_selector(
    model: PresolveSelector,
    dataset: PresolveDataset,
    epochs: int = 180,
    learning_rate: float = 2e-3,
) -> tuple[float, float]:
    if epochs < 1 or learning_rate <= 0.0:
        raise ValueError("invalid training configuration")
    model.set_normalization(dataset.features)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    weights = class_weights(dataset)
    initial_loss = float("nan")
    final_loss = float("nan")
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(dataset.features)
        loss = nn.functional.cross_entropy(logits, dataset.labels, weight=weights)
        if epoch == 0:
            initial_loss = float(loss.detach())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return initial_loss, final_loss


def predict_configuration(model: PresolveSelector, features: np.ndarray) -> PresolveConfig:
    features = np.asarray(features, dtype=np.float32)
    if features.shape != (N_FEATURES,):
        raise ValueError("features have wrong shape")
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(features, dtype=torch.float32).unsqueeze(0))
    return CONFIGS[int(torch.argmax(logits, dim=1).item())]


def evaluate_selector(model: PresolveSelector, dataset: PresolveDataset) -> ClassificationMetrics:
    model.eval()
    with torch.no_grad():
        logits = model(dataset.features)
        chosen = torch.argmax(logits, dim=1)
    accuracy = float((chosen == dataset.labels).to(torch.float32).mean())
    row_ids = torch.arange(dataset.size)
    chosen_score = dataset.oracle_scores[row_ids, chosen]
    oracle_score = dataset.oracle_scores.min(dim=1).values
    regret = (chosen_score - oracle_score) / torch.clamp(torch.abs(oracle_score), min=1.0)
    return ClassificationMetrics(
        accuracy=accuracy,
        mean_normalized_regret=float(regret.mean()),
    )
