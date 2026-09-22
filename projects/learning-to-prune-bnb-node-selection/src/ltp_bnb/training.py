from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bnb import BranchAndBoundSolver
from .policy import ImitationPolicy, fit_imitation_policy
from .problem import KnapsackInstance, brute_force_optimum, generate_knapsack


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float


def make_instances(seed: int, count: int, n_items: int) -> list[KnapsackInstance]:
    return [generate_knapsack(seed + i, n_items=n_items) for i in range(count)]


def collect_oracle_dataset(instances: list[KnapsackInstance]) -> tuple[np.ndarray, np.ndarray]:
    """Collect B&B states and label whether each state lies on one known optimal path."""
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    for instance in instances:
        _, oracle_bits = brute_force_optimum(instance)
        result = BranchAndBoundSolver(selection_mode="best_bound").solve(
            instance,
            oracle_decisions=oracle_bits,
            collect_trace=True,
        )
        for record in result.trace:
            x_rows.append(record.features)
            y_rows.append(record.oracle_expand)
    if not x_rows:
        raise RuntimeError("no training states were collected")
    x = np.stack(x_rows)
    y = np.asarray(y_rows, dtype=int)
    if len(np.unique(y)) < 2:
        raise RuntimeError("oracle dataset did not contain both positive and negative states")
    return x, y


def classification_metrics(
    policy: ImitationPolicy, x: np.ndarray, y: np.ndarray
) -> ClassificationMetrics:
    probs = np.asarray([policy.expand_probability(row) for row in x])
    pred = (probs >= 0.5).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    accuracy = (tp + tn) / max(len(y), 1)
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return ClassificationMetrics(
        accuracy=float(accuracy),
        balanced_accuracy=float((tpr + tnr) / 2),
        precision=float(precision),
        recall=float(tpr),
    )


def train_policy(
    *,
    seed: int = 42,
    train_instances: int = 80,
    validation_instances: int = 20,
    n_items: int = 12,
) -> tuple[ImitationPolicy, ClassificationMetrics, dict[str, int]]:
    train = make_instances(seed, train_instances, n_items)
    validation = make_instances(seed + 100_000, validation_instances, n_items)
    train_x, train_y = collect_oracle_dataset(train)
    val_x, val_y = collect_oracle_dataset(validation)
    policy = fit_imitation_policy(train_x, train_y)
    metrics = classification_metrics(policy, val_x, val_y)
    counts = {
        "train_states": int(len(train_y)),
        "train_positive": int(np.sum(train_y == 1)),
        "train_negative": int(np.sum(train_y == 0)),
        "validation_states": int(len(val_y)),
        "validation_positive": int(np.sum(val_y == 1)),
        "validation_negative": int(np.sum(val_y == 0)),
    }
    return policy, metrics, counts
