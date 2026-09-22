"""Adapters that turn learned score models into branch-and-bound policies."""

from __future__ import annotations

import numpy as np
from torch import nn

from ltb_milp.branching import BranchDecision
from ltb_milp.features import candidate_features
from ltb_milp.problem import BinaryPackingMILP
from ltb_milp.training import predict_scores


def learned_branch_policy(model: nn.Module):
    """Create a branch policy compatible with ``solve_branch_and_bound``."""

    def policy(
        problem: BinaryPackingMILP,
        x: np.ndarray,
        objective: float,
        bounds: dict[int, tuple[float, float]],
    ) -> BranchDecision:
        del objective, bounds
        candidates, features = candidate_features(problem, x)
        if candidates.size == 0:
            raise ValueError("no fractional branching candidates")
        scores = predict_scores(model, features)
        return BranchDecision(int(candidates[np.argmax(scores)]))

    return policy
