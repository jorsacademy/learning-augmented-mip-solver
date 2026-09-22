import numpy as np

from solverconfig.audit import audit_solution
from solverconfig.instances import SetCoverInstance


def instance() -> SetCoverInstance:
    return SetCoverInstance(
        "audit",
        "uniform",
        np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8),
        np.array([2.0, 3.0, 4.0]),
        1,
    )


def test_audit_recomputes_objective_and_constraints() -> None:
    audit = audit_solution(instance(), [1.0, 1.0, 0.0])
    assert audit.feasible
    assert audit.objective_recomputed == 5.0
    assert audit.max_cover_violation == 0.0


def test_audit_rejects_missing_decisions_and_infeasibility() -> None:
    assert not audit_solution(instance(), [1.0, 0.0]).decision_length_ok
    bad = audit_solution(instance(), [1.0, 0.0, 0.0])
    assert not bad.feasible
    assert bad.max_cover_violation == 1.0
