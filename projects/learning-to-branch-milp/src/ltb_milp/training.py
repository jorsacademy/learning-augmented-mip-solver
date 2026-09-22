"""Training utilities for learned branching scores."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from ltb_milp.dataset import BranchingDataset


def _group_slices(group_sizes: tuple[int, ...]) -> list[slice]:
    slices: list[slice] = []
    start = 0
    for size in group_sizes:
        slices.append(slice(start, start + size))
        start += size
    return slices


def listwise_expert_loss(
    predicted: Tensor,
    targets: Tensor,
    group_sizes: tuple[int, ...],
) -> Tensor:
    """Cross-entropy loss for selecting the expert's best candidate per B&B node."""
    losses: list[Tensor] = []
    for group in _group_slices(group_sizes):
        logits = predicted[group]
        expert_index = torch.argmax(targets[group]).reshape(1)
        losses.append(nn.functional.cross_entropy(logits.reshape(1, -1), expert_index))
    return torch.stack(losses).mean()


def train_branching_model(
    model: nn.Module,
    dataset: BranchingDataset,
    *,
    loss: str = "listwise",
    epochs: int = 200,
    learning_rate: float = 1e-3,
) -> list[float]:
    """Fit strong-branching behavior with score regression or listwise ranking."""
    if loss not in {"mse", "listwise"}:
        raise ValueError("loss must be 'mse' or 'listwise'")
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")

    features = torch.as_tensor(dataset.features, dtype=torch.float32)
    targets = torch.as_tensor(dataset.targets, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[float] = []

    model.train()
    for _ in range(epochs):
        predicted = model(features)
        if loss == "mse":
            objective = nn.functional.mse_loss(predicted, targets)
        else:
            objective = listwise_expert_loss(predicted, targets, dataset.group_sizes)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        history.append(float(objective.detach()))
    return history


@torch.inference_mode()
def predict_scores(model: nn.Module, features: np.ndarray) -> np.ndarray:
    """Predict branching scores for a candidate feature matrix."""
    model.eval()
    tensor = torch.as_tensor(features, dtype=torch.float32)
    return model(tensor).cpu().numpy()


@torch.inference_mode()
def top1_expert_agreement(model: nn.Module, dataset: BranchingDataset) -> float:
    """Return the fraction of B&B nodes where model and strong branching pick the same candidate."""
    predicted = predict_scores(model, dataset.features)
    agreements = 0
    for group in _group_slices(dataset.group_sizes):
        predicted_choice = int(np.argmax(predicted[group]))
        expert_choice = int(np.argmax(dataset.targets[group]))
        agreements += int(predicted_choice == expert_choice)
    return agreements / len(dataset.group_sizes)
