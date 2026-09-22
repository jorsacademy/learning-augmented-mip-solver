"""Candidate-level features for learning branching scores."""

from __future__ import annotations

import numpy as np

from ltb_milp.branching import fractional_candidates
from ltb_milp.problem import BinaryPackingMILP


def candidate_features(
    problem: BinaryPackingMILP,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate indices and inexpensive normalized features."""
    candidates = fractional_candidates(x)
    if candidates.size == 0:
        return candidates, np.empty((0, 6), dtype=float)

    objective_scale = max(float(np.max(np.abs(problem.c))), 1.0)
    column_sums = problem.A.sum(axis=0)
    column_scale = max(float(np.max(column_sums)), 1.0)
    densities = (problem.A > 0).mean(axis=0)

    features = []
    for variable in candidates:
        value = float(x[variable])
        fractionality = min(value, 1.0 - value)
        coeffs = problem.A[:, variable]
        nonzero = coeffs[coeffs > 0]
        mean_coeff = float(nonzero.mean()) if nonzero.size else 0.0
        max_coeff = float(nonzero.max()) if nonzero.size else 0.0
        row_scale = max(float(problem.A.max()), 1.0)
        features.append(
            [
                value,
                fractionality,
                float(problem.c[variable] / objective_scale),
                float(column_sums[variable] / column_scale),
                float(densities[variable]),
                float((mean_coeff + max_coeff) / (2.0 * row_scale)),
            ]
        )
    return candidates, np.asarray(features, dtype=float)
