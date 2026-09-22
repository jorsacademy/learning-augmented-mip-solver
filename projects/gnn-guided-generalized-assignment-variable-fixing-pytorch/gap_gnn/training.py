from __future__ import annotations

import copy
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

from .graph import batch_features
from .model import BipartiteGNN
from .planning import GAPMILP, generate_gap_instance


@dataclass(frozen=True)
class LabeledDataset:
    instances: tuple
    labels: np.ndarray      # [N,J] optimal machine index
    objectives: np.ndarray


def generate_labeled_dataset(
    n_instances: int,
    *,
    seed: int,
    n_machines: int = 6,
    n_jobs: int = 18,
) -> LabeledDataset:
    instances, labels, objectives = [], [], []
    for k in range(n_instances):
        instance = generate_gap_instance(
            seed=seed + 7919 * k,
            n_machines=n_machines,
            n_jobs=n_jobs,
        )
        solution = GAPMILP(instance).solve()
        if solution is None or solution.status != "OPTIMAL":
            raise RuntimeError("training label MILP did not solve to optimality")
        instances.append(instance)
        labels.append(solution.assignment)
        objectives.append(solution.objective)
    return LabeledDataset(
        instances=tuple(instances),
        labels=np.asarray(labels, dtype=np.int64),
        objectives=np.asarray(objectives, dtype=float),
    )


@dataclass(frozen=True)
class TrainingResult:
    model: BipartiteGNN
    best_validation_accuracy: float
    final_train_loss: float


def assignment_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == labels).float().mean().item())


def train_gnn(
    train: LabeledDataset,
    validation: LabeledDataset,
    *,
    seed: int = 42,
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    hidden_dim: int = 48,
    layers: int = 3,
) -> TrainingResult:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = BipartiteGNN(hidden_dim=hidden_dim, layers=layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    vm, vj, ve = batch_features(validation.instances)
    vlabels = torch.tensor(validation.labels, dtype=torch.long)

    best_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    final_loss = float("nan")

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train.instances))
        model.train()
        epoch_losses = []

        for start in range(0, len(order), batch_size):
            idx = order[start:start+batch_size]
            instances = [train.instances[int(i)] for i in idx]
            mf, jf, ef = batch_features(instances)
            labels = torch.tensor(train.labels[idx], dtype=torch.long)
            logits = model(mf, jf, ef)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        final_loss = float(np.mean(epoch_losses))
        model.eval()
        with torch.no_grad():
            vacc = assignment_accuracy(model(vm, vj, ve), vlabels)
        if vacc > best_acc:
            best_acc = vacc
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0:
            print(f"epoch={epoch:03d} train_loss={final_loss:.4f} validation_accuracy={vacc:.3f}")

    model.load_state_dict(best_state)
    return TrainingResult(
        model=model,
        best_validation_accuracy=float(best_acc),
        final_train_loss=final_loss,
    )
