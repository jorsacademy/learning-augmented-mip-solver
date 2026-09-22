"""Bipartite MILP graph states for graph-based branching policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ltb_milp.branching import fractional_candidates, strong_branch_scores
from ltb_milp.problem import BinaryPackingMILP


@dataclass(frozen=True)
class GraphInputs:
    """Normalized bipartite graph features for one fractional B&B node."""

    variable_features: Tensor
    constraint_features: Tensor
    edge_weights: Tensor
    candidate_indices: Tensor


@dataclass(frozen=True)
class GraphBranchingState:
    """Graph inputs paired with the strong-branching expert choice."""

    inputs: GraphInputs
    expert_choice: int


def graph_inputs_from_node(problem: BinaryPackingMILP, x: np.ndarray) -> GraphInputs:
    """Build normalized graph features without invoking an expert branching rule."""
    candidates = fractional_candidates(x)
    if candidates.size == 0:
        raise ValueError("no fractional branching candidates")

    objective_scale = max(float(np.max(np.abs(problem.c))), 1.0)
    rhs_scale = max(float(np.max(np.abs(problem.b))), 1.0)
    coefficient_scale = max(float(np.max(np.abs(problem.A))), 1.0)

    variable_features = np.column_stack(
        [
            x,
            np.minimum(x, 1.0 - x),
            problem.c / objective_scale,
            (problem.A > 0).mean(axis=0),
        ]
    ).astype(np.float32)

    activity = problem.A @ x
    constraint_features = np.column_stack(
        [
            problem.b / rhs_scale,
            activity / rhs_scale,
            (problem.b - activity) / rhs_scale,
            (problem.A > 0).mean(axis=1),
        ]
    ).astype(np.float32)

    return GraphInputs(
        variable_features=torch.from_numpy(variable_features),
        constraint_features=torch.from_numpy(constraint_features),
        edge_weights=torch.from_numpy((problem.A / coefficient_scale).astype(np.float32)),
        candidate_indices=torch.from_numpy(candidates.astype(np.int64)),
    )


def graph_state_from_node(
    problem: BinaryPackingMILP,
    x: np.ndarray,
    objective: float,
    bounds: dict[int, tuple[float, float]] | None = None,
) -> GraphBranchingState:
    """Build graph features and label the node with strong branching."""
    inputs = graph_inputs_from_node(problem, x)
    candidates = inputs.candidate_indices.numpy()
    scored_candidates, scores, _ = strong_branch_scores(problem, x, objective, bounds)
    if not np.array_equal(candidates, scored_candidates):
        raise RuntimeError("candidate ordering mismatch")
    return GraphBranchingState(inputs=inputs, expert_choice=int(np.argmax(scores)))
