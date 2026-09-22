from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class GAPInstance:
    costs: np.ndarray          # [machine, job]
    resources: np.ndarray      # [machine, job]
    capacities: np.ndarray     # [machine]

    def __post_init__(self):
        c = np.asarray(self.costs, dtype=float)
        r = np.asarray(self.resources, dtype=float)
        cap = np.asarray(self.capacities, dtype=float)
        if c.ndim != 2 or r.shape != c.shape:
            raise ValueError("costs/resources must have identical [machine,job] shape")
        if cap.shape != (c.shape[0],):
            raise ValueError("capacities must have one entry per machine")
        if np.any(c < 0) or np.any(r <= 0) or np.any(cap <= 0):
            raise ValueError("invalid GAP data")
        if np.any(np.min(r, axis=0) > np.max(cap)):
            raise ValueError("at least one job cannot fit any machine")

    @property
    def n_machines(self) -> int:
        return int(self.costs.shape[0])

    @property
    def n_jobs(self) -> int:
        return int(self.costs.shape[1])

    @property
    def n_vars(self) -> int:
        return self.n_machines * self.n_jobs


@dataclass(frozen=True)
class GAPSolution:
    assignment: np.ndarray     # [job] -> machine
    vector: np.ndarray         # flattened x[machine,job]
    objective: float
    max_violation: float
    solve_seconds: float
    mip_node_count: int
    mip_gap: float
    status: str


def generate_gap_instance(
    *,
    seed: int,
    n_machines: int = 6,
    n_jobs: int = 18,
    capacity_slack: float = 0.18,
) -> GAPInstance:
    """
    Generate a feasible but nontrivial generalized-assignment instance.

    A hidden balanced assignment is used only to construct capacities, thereby
    guaranteeing at least one feasible solution. Costs are generated
    independently enough that the planted assignment is not an optimization
    label.
    """
    if n_machines < 2 or n_jobs < n_machines:
        raise ValueError("need at least two machines and at least as many jobs")
    rng = np.random.default_rng(seed)

    job_size = rng.uniform(7.0, 18.0, size=n_jobs)
    machine_efficiency = rng.uniform(0.78, 1.22, size=n_machines)
    resources = (
        machine_efficiency[:, None] * job_size[None, :]
        + rng.normal(0.0, 1.5, size=(n_machines, n_jobs))
    )
    resources = np.clip(resources, 2.5, None)

    machine_cost_level = rng.uniform(18.0, 34.0, size=n_machines)
    job_value = rng.uniform(7.0, 23.0, size=n_jobs)
    preference = rng.normal(0.0, 5.5, size=(n_machines, n_jobs))
    nonlinear = 4.0 * np.sin(
        np.arange(n_machines)[:, None] * 0.8
        + np.arange(n_jobs)[None, :] * 0.55
        + rng.normal(0.0, 0.25, size=(n_machines, n_jobs))
    )
    costs = (
        machine_cost_level[:, None]
        + 1.15 * job_value[None, :]
        + 0.65 * resources
        + preference
        + nonlinear
    )
    costs = np.clip(costs, 1.0, None)

    jobs = rng.permutation(n_jobs)
    planted = np.empty(n_jobs, dtype=int)
    for pos, j in enumerate(jobs):
        planted[j] = pos % n_machines

    load = np.zeros(n_machines)
    for j, m in enumerate(planted):
        load[m] += resources[m, j]

    capacities = load * (1.0 + capacity_slack) + rng.uniform(4.0, 10.0, size=n_machines)
    return GAPInstance(
        costs=costs.astype(np.float64),
        resources=resources.astype(np.float64),
        capacities=capacities.astype(np.float64),
    )


class GAPMILP:
    """Exact HiGHS MILP for the classical generalized assignment problem."""

    def __init__(self, instance: GAPInstance):
        self.instance = instance
        M, J = instance.n_machines, instance.n_jobs
        self.M, self.J = M, J
        self.n_vars = M * J

        rows, lower, upper = [], [], []

        for j in range(J):
            row = np.zeros(self.n_vars)
            for m in range(M):
                row[self._idx(m, j)] = 1.0
            rows.append(row)
            lower.append(1.0)
            upper.append(1.0)

        for m in range(M):
            row = np.zeros(self.n_vars)
            for j in range(J):
                row[self._idx(m, j)] = instance.resources[m, j]
            rows.append(row)
            lower.append(-np.inf)
            upper.append(instance.capacities[m])

        self.A = np.vstack(rows)
        self.lb = np.asarray(lower, dtype=float)
        self.ub = np.asarray(upper, dtype=float)
        self.constraint = LinearConstraint(self.A, self.lb, self.ub)
        self.integrality = np.ones(self.n_vars, dtype=int)

    def _idx(self, machine: int, job: int) -> int:
        return machine * self.J + job

    def solve(
        self,
        *,
        fixed_jobs: Dict[int, int] | None = None,
        time_limit: float = 30.0,
    ) -> GAPSolution | None:
        fixed_jobs = dict(fixed_jobs or {})
        lower = np.zeros(self.n_vars)
        upper = np.ones(self.n_vars)

        for j, chosen_machine in fixed_jobs.items():
            if not 0 <= j < self.J or not 0 <= chosen_machine < self.M:
                raise ValueError("invalid fixed assignment")
            for m in range(self.M):
                idx = self._idx(m, j)
                if m == chosen_machine:
                    lower[idx] = upper[idx] = 1.0
                else:
                    lower[idx] = upper[idx] = 0.0

        start = perf_counter()
        result = milp(
            c=self.instance.costs.reshape(-1),
            integrality=self.integrality,
            bounds=Bounds(lower, upper),
            constraints=self.constraint,
            options={"time_limit": float(time_limit), "mip_rel_gap": 0.0},
        )
        elapsed = perf_counter() - start

        if result.x is None:
            return None

        vector = np.asarray(result.x, dtype=float)
        x = vector.reshape(self.M, self.J)
        assignment = np.argmax(x, axis=0).astype(int)
        status = "OPTIMAL" if result.status == 0 else "LIMIT"
        return GAPSolution(
            assignment=assignment,
            vector=vector,
            objective=float(self.instance.costs.reshape(-1) @ vector),
            max_violation=self.max_constraint_violation(vector),
            solve_seconds=float(elapsed),
            mip_node_count=int(getattr(result, "mip_node_count", 0) or 0),
            mip_gap=float(getattr(result, "mip_gap", np.nan)),
            status=status,
        )

    def max_constraint_violation(self, vector: np.ndarray) -> float:
        z = np.asarray(vector, dtype=float)
        lhs = self.A @ z
        return float(max(
            np.max(np.maximum(self.lb - lhs, 0.0)),
            np.max(np.maximum(lhs - self.ub, 0.0)),
            np.max(np.maximum(-z, 0.0)),
            np.max(np.maximum(z - 1.0, 0.0)),
            np.max(np.abs(z - np.rint(z))),
        ))


def greedy_assignment(instance: GAPInstance, *, restarts: int = 32, seed: int = 0):
    """
    Multi-start capacity-aware greedy heuristic.

    Jobs are ordered by a mixture of resource difficulty and random jitter.
    This is a heuristic baseline only; it has no optimality claim.
    """
    rng = np.random.default_rng(seed)
    M, J = instance.n_machines, instance.n_jobs
    best_assignment, best_obj = None, float("inf")

    difficulty = np.min(
        instance.resources / instance.capacities[:, None],
        axis=0,
    )

    for restart in range(restarts):
        if restart == 0:
            order = np.argsort(-difficulty)
        else:
            order = np.argsort(-(difficulty + rng.normal(0.0, 0.08, size=J)))

        remaining = instance.capacities.copy()
        assignment = np.full(J, -1, dtype=int)
        feasible = True
        for j in order:
            candidates = [
                m for m in range(M)
                if instance.resources[m, j] <= remaining[m] + 1e-9
            ]
            if not candidates:
                feasible = False
                break
            scored = [
                (
                    instance.costs[m, j]
                    + 2.0 * instance.resources[m, j] / max(remaining[m], 1e-9),
                    m,
                )
                for m in candidates
            ]
            _, chosen = min(scored)
            assignment[j] = chosen
            remaining[chosen] -= instance.resources[chosen, j]

        if feasible:
            obj = float(sum(instance.costs[assignment[j], j] for j in range(J)))
            if obj < best_obj:
                best_obj = obj
                best_assignment = assignment.copy()

    return best_assignment, best_obj
