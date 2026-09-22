from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .instances import SetCoverInstance


@dataclass(frozen=True)
class ExactSolution:
    objective: float
    decision: np.ndarray


def exhaustive_set_cover(instance: SetCoverInstance, *, max_columns: int = 20) -> ExactSolution:
    """Independent exact oracle for tiny set-cover instances.

    Exhaustive subset enumeration is intentionally independent of HiGHS and is used only in tests
    and verification examples where n_cols is small.
    """
    n = instance.n_cols
    if n > max_columns:
        raise ValueError(f"exhaustive oracle limited to {max_columns} columns, got {n}")

    best_obj = float("inf")
    best_x: np.ndarray | None = None
    order = np.argsort(instance.costs)
    for cardinality in range(1, n + 1):
        for chosen_pos in combinations(range(n), cardinality):
            chosen = order[list(chosen_pos)]
            if float(instance.costs[chosen].sum()) >= best_obj:
                continue
            if np.all(instance.matrix[:, chosen].sum(axis=1) >= 1):
                x = np.zeros(n, dtype=int)
                x[chosen] = 1
                obj = float(instance.costs @ x)
                if obj < best_obj:
                    best_obj = obj
                    best_x = x
        # Do not stop after the first feasible cardinality: with nonuniform costs, a larger
        # subset can still be cheaper than a smaller one.

    if best_x is None:
        raise RuntimeError("generator invariant violated: instance is infeasible")
    return ExactSolution(best_obj, best_x)
