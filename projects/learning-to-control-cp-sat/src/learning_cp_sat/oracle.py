from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .audit import audit_assignment
from .instances import GraphColoringInstance


@dataclass(frozen=True)
class OracleResult:
    feasible: bool
    assignment: tuple[int, ...] | None
    complete_assignments_checked: int


def brute_force_oracle(instance: GraphColoringInstance) -> OracleResult:
    """Independent exhaustive oracle for very small instances.

    This deliberately does not reuse the project's propagation/search code.
    """
    checked = 0
    for assignment in product(range(instance.n_colors), repeat=instance.n_vertices):
        checked += 1
        if audit_assignment(instance, assignment).feasible:
            return OracleResult(True, tuple(assignment), checked)
    return OracleResult(False, None, checked)
