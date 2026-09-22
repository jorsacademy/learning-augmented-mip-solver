from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .audit import audit_assignment
from .expert import choose_expert_variable
from .features import candidate_features
from .instances import GraphColoringInstance
from .propagation import assign_and_propagate, colors_in, domain_size, initial_domains


@dataclass(frozen=True)
class SearchResult:
    status: str
    assignment: tuple[int, ...] | None
    nodes: int
    backtracks: int
    propagations: int
    policy_evaluations: int
    expert_probes: int
    wall_time_seconds: float


def _select_variable(
    policy: str,
    instance: GraphColoringInstance,
    domains: list[int],
    depth: int,
    rng: random.Random,
    model: Any | None,
) -> tuple[int, int, int]:
    candidates = [v for v, mask in enumerate(domains) if domain_size(mask) > 1]
    if not candidates:
        raise ValueError("no branching candidate")
    degrees = instance.degrees

    if policy == "fixed":
        return min(candidates), 0, 0
    if policy == "degree":
        return max(candidates, key=lambda v: (degrees[v], -v)), 0, 0
    if policy == "mrv":
        return min(candidates, key=lambda v: (domain_size(domains[v]), -degrees[v], v)), 0, 0
    if policy == "dom_degree":

        def key(v: int) -> tuple[float, int, int, int]:
            remaining = sum(domain_size(domains[u]) > 1 for u in instance.adjacency[v])
            return (
                domain_size(domains[v]) / max(1, remaining),
                domain_size(domains[v]),
                -remaining,
                v,
            )

        return min(candidates, key=key), 0, 0
    if policy == "random":
        return rng.choice(candidates), 0, 0
    if policy == "expert":
        variable, _, probes = choose_expert_variable(instance, domains)
        return variable, 0, probes
    if policy == "learned":
        if model is None:
            raise ValueError("learned policy requires a fitted model")
        matrix = np.vstack([candidate_features(instance, domains, v, depth) for v in candidates])
        scores = model.predict(matrix)
        best_index = max(
            range(len(candidates)),
            key=lambda i: (float(scores[i]), degrees[candidates[i]], -candidates[i]),
        )
        return candidates[best_index], len(candidates), 0
    raise ValueError(f"unknown policy: {policy}")


def solve_coloring(
    instance: GraphColoringInstance,
    *,
    policy: str,
    seed: int = 0,
    model: Any | None = None,
    node_limit: int | None = None,
    time_limit_seconds: float | None = None,
) -> SearchResult:
    """Exact DFS with singleton propagation unless a configured limit is hit."""
    start = time.perf_counter()
    rng = random.Random(seed)
    nodes = 0
    backtracks = 0
    propagations = 0
    policy_evaluations = 0
    expert_probes = 0
    limit_hit = False
    solution: tuple[int, ...] | None = None

    def over_limit() -> bool:
        nonlocal limit_hit
        if node_limit is not None and nodes >= node_limit:
            limit_hit = True
            return True
        if time_limit_seconds is not None and time.perf_counter() - start >= time_limit_seconds:
            limit_hit = True
            return True
        return False

    def dfs(domains: list[int], depth: int) -> bool:
        nonlocal nodes, backtracks, propagations, policy_evaluations, expert_probes, solution
        if all(domain_size(mask) == 1 for mask in domains):
            candidate = tuple(mask.bit_length() - 1 for mask in domains)
            audit = audit_assignment(instance, candidate)
            if not audit.feasible:
                raise AssertionError("internal solver produced an infeasible assignment")
            solution = candidate
            return True
        if over_limit():
            return False

        variable, evals, probes = _select_variable(policy, instance, domains, depth, rng, model)
        policy_evaluations += evals
        expert_probes += probes
        nodes += 1

        for color in colors_in(domains[variable]):
            child, _, propagated = assign_and_propagate(instance, domains, variable, color)
            propagations += propagated
            if child is None:
                backtracks += 1
                continue
            if dfs(child, depth + 1):
                return True
            if limit_hit:
                return False
            backtracks += 1
        return False

    found = dfs(initial_domains(instance), 0)
    if found:
        status = "FEASIBLE"
    elif limit_hit:
        status = "UNKNOWN"
    else:
        status = "INFEASIBLE"

    return SearchResult(
        status=status,
        assignment=solution,
        nodes=nodes,
        backtracks=backtracks,
        propagations=propagations,
        policy_evaluations=policy_evaluations,
        expert_probes=expert_probes,
        wall_time_seconds=time.perf_counter() - start,
    )
