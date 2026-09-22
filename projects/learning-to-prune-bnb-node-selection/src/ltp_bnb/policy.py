from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-arr))


@dataclass
class BinaryLinearModel:
    weights: list[float]
    bias: float
    mean: list[float]
    scale: list[float]

    def predict_proba(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        mean = np.asarray(self.mean, dtype=float)
        scale = np.asarray(self.scale, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        z = (x - mean) / scale
        return float(_sigmoid(float(z @ weights + self.bias)))


@dataclass
class ImitationPolicy:
    """Separate learned scoring models for node selection and approximate pruning."""

    selection: BinaryLinearModel
    pruning: BinaryLinearModel

    def selection_score(self, features: np.ndarray) -> float:
        return self.selection.predict_proba(features)

    def expand_probability(self, features: np.ndarray) -> float:
        return self.pruning.predict_proba(features)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "selection": asdict(self.selection),
            "pruning": asdict(self.pruning),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ImitationPolicy:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != 1:
            raise ValueError("unsupported policy format")
        return cls(
            selection=BinaryLinearModel(**payload["selection"]),
            pruning=BinaryLinearModel(**payload["pruning"]),
        )


def fit_binary_linear_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 600,
    learning_rate: float = 0.08,
    l2: float = 1e-3,
    positive_weight: float | None = None,
) -> BinaryLinearModel:
    """Fit a deterministic class-balanced logistic model with NumPy."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("invalid training array shapes")
    if len(np.unique(y)) < 2:
        raise ValueError("training data must contain both classes")

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale

    n_pos = max(float(np.sum(y == 1.0)), 1.0)
    n_neg = max(float(np.sum(y == 0.0)), 1.0)
    if positive_weight is None:
        positive_weight = n_neg / n_pos
    sample_weight = np.where(y == 1.0, positive_weight, 1.0)
    sample_weight /= np.mean(sample_weight)

    weights = np.zeros(z.shape[1], dtype=float)
    bias = 0.0
    for _ in range(epochs):
        logits = z @ weights + bias
        pred = np.asarray(_sigmoid(logits), dtype=float)
        error = (pred - y) * sample_weight
        grad_w = (z.T @ error) / len(z) + l2 * weights
        grad_b = float(np.mean(error))
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return BinaryLinearModel(
        weights=weights.tolist(),
        bias=float(bias),
        mean=mean.tolist(),
        scale=scale.tolist(),
    )


def fit_imitation_policy(x: np.ndarray, y: np.ndarray) -> ImitationPolicy:
    """Fit separate policies from the same oracle-path labels with different class emphasis."""
    selection = fit_binary_linear_model(x, y, positive_weight=None)
    n_pos = max(float(np.sum(y == 1)), 1.0)
    n_neg = max(float(np.sum(y == 0)), 1.0)
    pruning = fit_binary_linear_model(x, y, positive_weight=2.0 * n_neg / n_pos)
    return ImitationPolicy(selection=selection, pruning=pruning)
