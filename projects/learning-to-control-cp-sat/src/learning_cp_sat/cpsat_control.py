from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .audit import audit_assignment
from .features import candidate_features
from .instances import GraphColoringInstance
from .propagation import initial_domains


@dataclass(frozen=True)
class CpSatResult:
    strategy: str
    status: str
    assignment: tuple[int, ...] | None
    branches: int
    conflicts: int
    wall_time_seconds: float


def _root_order(
    instance: GraphColoringInstance,
    strategy: str,
    model: Any | None,
) -> list[int]:
    vertices = list(range(instance.n_vertices))
    if strategy == "fixed":
        return vertices
    if strategy == "degree":
        return sorted(vertices, key=lambda v: (-instance.degrees[v], v))
    if strategy == "learned_root":
        if model is None:
            raise ValueError("learned_root requires a fitted model")
        domains = initial_domains(instance)
        x = np.vstack([candidate_features(instance, domains, v, 0) for v in vertices])
        scores = model.predict(x)
        return sorted(vertices, key=lambda v: (-float(scores[v]), -instance.degrees[v], v))
    raise ValueError(f"unknown fixed-search strategy: {strategy}")


def solve_with_cpsat(
    instance: GraphColoringInstance,
    *,
    strategy: str,
    seed: int,
    time_limit_seconds: float,
    model: Any | None = None,
) -> CpSatResult:
    """Run CP-SAT using only public Python API search-control surfaces."""
    from ortools.sat.python import cp_model

    cp = cp_model.CpModel()
    variables = [
        cp.new_int_var(0, instance.n_colors - 1, f"color_{v}") for v in range(instance.n_vertices)
    ]
    for u, v in instance.edges:
        cp.add(variables[u] != variables[v])

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = seed
    solver.parameters.cp_model_presolve = False
    solver.parameters.symmetry_level = 0

    if strategy == "min_domain":
        cp.add_decision_strategy(
            variables,
            cp_model.CHOOSE_MIN_DOMAIN_SIZE,
            cp_model.SELECT_MIN_VALUE,
        )
        solver.parameters.search_branching = cp_model.FIXED_SEARCH
    elif strategy != "default":
        order = _root_order(instance, strategy, model)
        cp.add_decision_strategy(
            [variables[v] for v in order],
            cp_model.CHOOSE_FIRST,
            cp_model.SELECT_MIN_VALUE,
        )
        solver.parameters.search_branching = cp_model.FIXED_SEARCH

    status_code = solver.solve(cp)
    status = solver.status_name(status_code)
    assignment: tuple[int, ...] | None = None
    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        assignment = tuple(int(solver.value(variable)) for variable in variables)
        audit = audit_assignment(instance, assignment)
        if not audit.feasible:
            raise AssertionError("CP-SAT returned an assignment that failed independent audit")

    return CpSatResult(
        strategy=strategy,
        status=status,
        assignment=assignment,
        branches=int(solver.num_branches),
        conflicts=int(solver.num_conflicts),
        wall_time_seconds=float(solver.wall_time),
    )
