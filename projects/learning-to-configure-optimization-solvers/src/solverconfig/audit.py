from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .instances import SetCoverInstance


@dataclass(frozen=True)
class FeasibilityAudit:
    feasible: bool
    decision_length_ok: bool
    max_integrality_violation: float
    max_cover_violation: float
    objective_recomputed: float | None


def audit_solution(
    instance: SetCoverInstance,
    values: np.ndarray | list[float] | None,
    *,
    tolerance: float = 1e-6,
) -> FeasibilityAudit:
    if values is None:
        return FeasibilityAudit(False, False, float("inf"), float("inf"), None)
    x = np.asarray(values, dtype=float)
    length_ok = x.ndim == 1 and x.size == instance.n_cols
    if not length_ok:
        return FeasibilityAudit(False, False, float("inf"), float("inf"), None)

    rounded = np.rint(x)
    integrality = float(np.max(np.abs(x - rounded), initial=0.0))
    cover_slack = instance.matrix @ rounded - 1.0
    cover_violation = float(max(0.0, -np.min(cover_slack, initial=0.0)))
    bounds_ok = bool(np.all(rounded >= 0) and np.all(rounded <= 1))
    feasible = bounds_ok and integrality <= tolerance and cover_violation <= tolerance
    objective = float(instance.costs @ rounded) if feasible else None
    return FeasibilityAudit(feasible, True, integrality, cover_violation, objective)
