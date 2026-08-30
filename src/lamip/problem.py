from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class BinaryPackingMIP:
    profits: np.ndarray
    A: np.ndarray
    b: np.ndarray

    def __post_init__(self) -> None:
        if self.profits.ndim != 1 or self.A.ndim != 2 or self.b.ndim != 1:
            raise ValueError("invalid problem dimensions")
        if self.A.shape != (self.b.size, self.profits.size):
            raise ValueError("A shape must match b and profits")
        if self.profits.size < 2 or self.b.size < 1:
            raise ValueError("problem must contain variables and constraints")
        if np.any(self.A < 0) or np.any(self.b < 0):
            raise ValueError("packing coefficients and capacities must be nonnegative")

    @property
    def n_vars(self) -> int:
        return int(self.profits.size)

    @property
    def n_constraints(self) -> int:
        return int(self.b.size)


@dataclass(frozen=True)
class ExactResult:
    objective: float
    solution: np.ndarray
    nodes: int


def generate_binary_packing(
    n_vars: int = 18,
    n_constraints: int = 6,
    seed: int = 0,
) -> BinaryPackingMIP:
    if n_vars < 2 or n_constraints < 1:
        raise ValueError("invalid problem size")
    rng = np.random.default_rng(seed)
    density = rng.uniform(0.35, 0.75)
    mask = rng.random((n_constraints, n_vars)) < density
    A = (mask * rng.integers(1, 11, size=(n_constraints, n_vars))).astype(float)
    profits = rng.integers(1, 31, size=n_vars).astype(float)
    b = np.maximum(1.0, 0.42 * A.sum(axis=1))

    # Inject safe-presolve structure without making every instance identical.
    mode = seed % 4
    if mode in {1, 3}:
        profits[0] = -1.0
        row = mode % n_constraints
        A[row, 1] = b[row] + 2.0
    if mode in {2, 3}:
        A = np.vstack([A, 0.8 * A[0]])
        b = np.r_[b, b[0] + 1.0]

    return BinaryPackingMIP(profits=profits, A=A, b=b)


def solve_reference(problem: BinaryPackingMIP, presolve: bool = False) -> ExactResult:
    result = milp(
        c=-problem.profits,
        integrality=np.ones(problem.n_vars),
        bounds=Bounds(np.zeros(problem.n_vars), np.ones(problem.n_vars)),
        constraints=LinearConstraint(problem.A, -np.inf, problem.b),
        options={"presolve": presolve},
    )
    if not result.success or result.fun is None or result.x is None:
        raise RuntimeError(f"reference MILP solve failed: {result.message}")
    return ExactResult(
        objective=float(-result.fun),
        solution=np.rint(result.x).astype(float),
        nodes=int(getattr(result, "mip_node_count", 0) or 0),
    )
