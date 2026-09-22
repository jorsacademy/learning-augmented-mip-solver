"""Binary packing MILP generation and LP relaxation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class BinaryPackingMILP:
    """Binary packing problem max c^T x subject to A x <= b, x in {0,1}^n."""

    c: np.ndarray
    A: np.ndarray
    b: np.ndarray

    def __post_init__(self) -> None:
        if self.c.ndim != 1:
            raise ValueError("c must be one-dimensional")
        if self.A.ndim != 2 or self.A.shape[1] != self.c.size:
            raise ValueError("A must have shape [constraints, variables]")
        if self.b.shape != (self.A.shape[0],):
            raise ValueError("b must match the number of constraints")

    @property
    def n_vars(self) -> int:
        return int(self.c.size)

    @property
    def n_constraints(self) -> int:
        return int(self.b.size)


@dataclass(frozen=True)
class LPSolution:
    feasible: bool
    objective: float
    x: np.ndarray | None


def generate_binary_packing(
    n_vars: int,
    n_constraints: int,
    *,
    seed: int = 0,
    density: float = 0.7,
    tightness: float | None = None,
) -> BinaryPackingMILP:
    """Generate a reproducible random binary packing instance.

    ``tightness`` controls the RHS as a fraction of each constraint's total
    coefficient mass. Smaller values produce tighter packing constraints. When
    omitted, each row independently samples a fraction in [0.35, 0.65], matching
    the original training distribution.
    """
    if n_vars < 2 or n_constraints < 1:
        raise ValueError("n_vars must be >= 2 and n_constraints must be >= 1")
    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")
    if tightness is not None and not 0 < tightness < 1:
        raise ValueError("tightness must be in (0, 1)")

    rng = np.random.default_rng(seed)
    mask = rng.random((n_constraints, n_vars)) < density
    coefficients = rng.integers(1, 11, size=(n_constraints, n_vars)).astype(float)
    A = coefficients * mask
    for row in range(n_constraints):
        if not np.any(A[row]):
            A[row, rng.integers(0, n_vars)] = float(rng.integers(1, 11))
    if tightness is None:
        fractions = rng.uniform(0.35, 0.65, size=n_constraints)
    else:
        fractions = np.full(n_constraints, tightness, dtype=float)
    b = np.maximum(1.0, fractions * A.sum(axis=1))
    c = rng.integers(1, 21, size=n_vars).astype(float)
    return BinaryPackingMILP(c=c, A=A, b=b)


def solve_lp_relaxation(
    problem: BinaryPackingMILP,
    bounds: dict[int, tuple[float, float]] | None = None,
) -> LPSolution:
    """Solve the continuous relaxation, respecting optional variable bounds."""
    variable_bounds = [(0.0, 1.0) for _ in range(problem.n_vars)]
    if bounds:
        for index, (lower, upper) in bounds.items():
            if not 0 <= index < problem.n_vars:
                raise ValueError("bound index out of range")
            if lower > upper:
                return LPSolution(False, float("-inf"), None)
            variable_bounds[index] = (lower, upper)

    result = linprog(
        c=-problem.c,
        A_ub=problem.A,
        b_ub=problem.b,
        bounds=variable_bounds,
        method="highs",
    )
    if not result.success:
        return LPSolution(False, float("-inf"), None)
    return LPSolution(True, float(-result.fun), np.asarray(result.x, dtype=float))
