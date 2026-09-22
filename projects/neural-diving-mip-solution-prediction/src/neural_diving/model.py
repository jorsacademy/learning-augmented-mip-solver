from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


class SolutionPredictor(nn.Module):
    """Small per-variable MLP predicting P(x_i = 1 | instance coefficients)."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 32) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


@dataclass(frozen=True)
class TrainingMetrics:
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    brier: float
    positive_rate: float


def _metrics(y_true: np.ndarray, probs: np.ndarray) -> TrainingMetrics:
    y_true = y_true.astype(int)
    pred = (probs >= 0.5).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return TrainingMetrics(
        accuracy=float(np.mean(pred == y_true)),
        balanced_accuracy=float(0.5 * (tpr + tnr)),
        precision=float(tp / max(tp + fp, 1)),
        recall=float(tpr),
        brier=float(np.mean((probs - y_true) ** 2)),
        positive_rate=float(np.mean(y_true)),
    )


def train_predictor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int = 0,
    epochs: int = 120,
    learning_rate: float = 2e-3,
    hidden_dim: int = 32,
) -> tuple[SolutionPredictor, TrainingMetrics]:
    """Train a deterministic CPU MLP using numerically stable BCEWithLogitsLoss."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    model = SolutionPredictor(input_dim=x_train.shape[1], hidden_dim=hidden_dim)

    x_t = torch.as_tensor(x_train, dtype=torch.float32)
    y_t = torch.as_tensor(y_train, dtype=torch.float32)
    positives = max(float(y_t.sum().item()), 1.0)
    negatives = max(float(y_t.numel() - y_t.sum().item()), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        optimizer.step()

    probs = predict_probabilities(model, x_val)
    return model, _metrics(y_val, probs)


def predict_probabilities(model: SolutionPredictor, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(features, dtype=torch.float32))
        return torch.sigmoid(logits).cpu().numpy().astype(float)


def save_predictor(model: SolutionPredictor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_predictor(path: str | Path) -> SolutionPredictor:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported predictor format")
    model = SolutionPredictor(
        input_dim=int(payload["input_dim"]), hidden_dim=int(payload["hidden_dim"])
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
