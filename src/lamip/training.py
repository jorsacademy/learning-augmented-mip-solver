from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .cuts import cut_features, generate_cover_cuts
from .models import BranchScorer, CutScorer, LearnedComponents, PresolveSelector, PrimalScorer
from .presolve import PRESOLVE_CONFIGS, apply_presolve, instance_features
from .problem import BinaryPackingMIP, generate_binary_packing, solve_reference
from .solver import _branch_features, _primal_features, _solve_lp


@dataclass(frozen=True)
class TrainingData:
    presolve_x: torch.Tensor
    presolve_y: torch.Tensor
    cut_x: torch.Tensor
    cut_y: torch.Tensor
    branch_x: torch.Tensor
    branch_y: torch.Tensor
    primal_x: torch.Tensor
    primal_y: torch.Tensor


def _presolve_label(problem: BinaryPackingMIP) -> int:
    baseline = solve_reference(problem, presolve=False)
    candidates: list[tuple[tuple[int, int, int], int]] = []
    for index, config in enumerate(PRESOLVE_CONFIGS):
        reduced = apply_presolve(problem, config)
        solved = solve_reference(reduced.problem, presolve=False)
        objective = solved.objective + reduced.objective_offset
        if not np.isclose(objective, baseline.objective, atol=1e-7):
            continue
        residual_size = reduced.problem.n_vars + reduced.problem.n_constraints
        candidates.append(((solved.nodes, residual_size, reduced.checks), index))
    if not candidates:
        raise RuntimeError("no objective-preserving presolve configuration")
    return min(candidates, key=lambda item: item[0])[1]


def _cut_examples(problem: BinaryPackingMIP) -> tuple[list[np.ndarray], list[float]]:
    lp = _solve_lp(problem, [])
    if lp is None:
        return [], []
    candidates = generate_cover_cuts(problem, lp.x)
    features: list[np.ndarray] = []
    labels: list[float] = []
    for cut in candidates:
        child = _solve_lp(problem, [cut])
        if child is None:
            continue
        features.append(cut_features(problem, lp.x, cut))
        labels.append(max(0.0, lp.objective - child.objective))
    return features, labels


def _branch_examples(problem: BinaryPackingMIP) -> tuple[list[np.ndarray], list[float]]:
    lp = _solve_lp(problem, [])
    if lp is None:
        return [], []
    candidates = np.flatnonzero(np.minimum(lp.x, 1.0 - lp.x) > 1e-7)
    if candidates.size == 0:
        return [], []
    features = _branch_features(problem, lp.x, candidates)
    labels: list[float] = []
    fallback = abs(lp.objective) + 1.0
    for j in candidates:
        down = _solve_lp(problem, [], {int(j): (0.0, 0.0)})
        up = _solve_lp(problem, [], {int(j): (1.0, 1.0)})
        down_gain = fallback if down is None else max(0.0, lp.objective - down.objective)
        up_gain = fallback if up is None else max(0.0, lp.objective - up.objective)
        labels.append(min(down_gain, up_gain) + 1e-6 * max(down_gain, up_gain))
    return list(features), labels


def collect_training_data(
    n_instances: int = 40,
    n_vars: int = 18,
    n_constraints: int = 6,
    seed: int = 0,
) -> TrainingData:
    if n_instances < 1:
        raise ValueError("n_instances must be positive")
    presolve_x: list[np.ndarray] = []
    presolve_y: list[int] = []
    cut_x: list[np.ndarray] = []
    cut_y: list[float] = []
    branch_x: list[np.ndarray] = []
    branch_y: list[float] = []
    primal_x: list[np.ndarray] = []
    primal_y: list[np.ndarray] = []

    for offset in range(n_instances):
        original = generate_binary_packing(n_vars, n_constraints, seed + offset)
        presolve_x.append(instance_features(original))
        presolve_y.append(_presolve_label(original))

        residual = apply_presolve(original, "full").problem
        cut_features_rows, cut_labels = _cut_examples(residual)
        cut_x.extend(cut_features_rows)
        cut_y.extend(cut_labels)

        branch_features_rows, branch_labels = _branch_examples(residual)
        branch_x.extend(branch_features_rows)
        branch_y.extend(branch_labels)

        exact = solve_reference(residual, presolve=False)
        primal_x.extend(_primal_features(residual))
        primal_y.extend(exact.solution.astype(np.float32))

    if not cut_x or not branch_x:
        raise RuntimeError("training generator produced insufficient expert decisions")

    return TrainingData(
        presolve_x=torch.tensor(np.stack(presolve_x), dtype=torch.float32),
        presolve_y=torch.tensor(presolve_y, dtype=torch.long),
        cut_x=torch.tensor(np.stack(cut_x), dtype=torch.float32),
        cut_y=torch.tensor(cut_y, dtype=torch.float32),
        branch_x=torch.tensor(np.stack(branch_x), dtype=torch.float32),
        branch_y=torch.tensor(branch_y, dtype=torch.float32),
        primal_x=torch.tensor(np.stack(primal_x), dtype=torch.float32),
        primal_y=torch.tensor(primal_y, dtype=torch.float32),
    )


def _fit(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
    epochs: int,
    learning_rate: float,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    value = float("nan")
    for _ in range(epochs):
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
    return value


def train_components(
    data: TrainingData,
    epochs: int = 80,
    learning_rate: float = 1e-3,
    hidden_dim: int = 32,
    seed: int = 0,
) -> tuple[LearnedComponents, dict[str, float]]:
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("invalid training configuration")
    torch.manual_seed(seed)
    components = LearnedComponents(
        presolve=PresolveSelector(hidden_dim),
        cuts=CutScorer(hidden_dim),
        branching=BranchScorer(hidden_dim),
        primal=PrimalScorer(hidden_dim),
    )
    losses = {
        "presolve": _fit(
            components.presolve,
            data.presolve_x,
            data.presolve_y,
            nn.CrossEntropyLoss(),
            epochs,
            learning_rate,
        ),
        "cuts": _fit(
            components.cuts,
            data.cut_x,
            data.cut_y,
            nn.MSELoss(),
            epochs,
            learning_rate,
        ),
        "branching": _fit(
            components.branching,
            data.branch_x,
            data.branch_y,
            nn.MSELoss(),
            epochs,
            learning_rate,
        ),
        "primal": _fit(
            components.primal,
            data.primal_x,
            data.primal_y,
            nn.BCEWithLogitsLoss(),
            epochs,
            learning_rate,
        ),
    }
    return components.eval(), losses


def checkpoint_payload(components: LearnedComponents, hidden_dim: int = 32) -> dict[str, object]:
    if any(
        model is None
        for model in (components.presolve, components.cuts, components.branching, components.primal)
    ):
        raise ValueError("all learned components are required")
    return {
        "hidden_dim": hidden_dim,
        "presolve": components.presolve.state_dict(),
        "cuts": components.cuts.state_dict(),
        "branching": components.branching.state_dict(),
        "primal": components.primal.state_dict(),
    }


def load_components(path: str) -> LearnedComponents:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    hidden_dim = int(payload["hidden_dim"])
    components = LearnedComponents(
        presolve=PresolveSelector(hidden_dim),
        cuts=CutScorer(hidden_dim),
        branching=BranchScorer(hidden_dim),
        primal=PrimalScorer(hidden_dim),
    )
    components.presolve.load_state_dict(payload["presolve"])
    components.cuts.load_state_dict(payload["cuts"])
    components.branching.load_state_dict(payload["branching"])
    components.primal.load_state_dict(payload["primal"])
    return components.eval()
