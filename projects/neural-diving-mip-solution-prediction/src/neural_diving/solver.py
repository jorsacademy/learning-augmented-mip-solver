from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .problem import BinaryPackingMIP


@dataclass(frozen=True)
class SolveResult:
    x: np.ndarray | None
    objective: float | None
    success: bool
    status: int
    message: str
    elapsed_seconds: float
    mip_node_count: int | None
    mip_gap: float | None
    fixed_count: int


def solve_mip(
    instance: BinaryPackingMIP,
    *,
    fixed: dict[int, int] | None = None,
    time_limit: float | None = None,
    mip_rel_gap: float = 0.0,
) -> SolveResult:
    """Solve the full or partially fixed binary MILP with SciPy/HiGHS."""
    n = instance.n_items
    lb = np.zeros(n, dtype=float)
    ub = np.ones(n, dtype=float)
    fixed = fixed or {}
    for index, value in fixed.items():
        if not 0 <= index < n or value not in (0, 1):
            raise ValueError("fixed assignments must map valid indices to 0/1")
        lb[index] = value
        ub[index] = value

    options: dict[str, float | bool] = {"mip_rel_gap": float(mip_rel_gap), "presolve": True}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)

    start = perf_counter()
    result = milp(
        c=-instance.c,
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(instance.A, -np.inf, instance.b),
        options=options,
    )
    elapsed = perf_counter() - start

    x = None if result.x is None else np.rint(np.asarray(result.x, dtype=float)).astype(int)
    objective = None if x is None else instance.objective(x)
    success = bool(result.success and x is not None and instance.feasible(x))
    return SolveResult(
        x=x,
        objective=objective,
        success=success,
        status=int(result.status),
        message=str(result.message),
        elapsed_seconds=float(elapsed),
        mip_node_count=(
            None if getattr(result, "mip_node_count", None) is None else int(result.mip_node_count)
        ),
        mip_gap=None if getattr(result, "mip_gap", None) is None else float(result.mip_gap),
        fixed_count=len(fixed),
    )


def brute_force_solve(instance: BinaryPackingMIP, *, max_items: int = 22) -> SolveResult:
    """Exact enumerative verifier for tests and tiny research checks."""
    if instance.n_items > max_items:
        raise ValueError(f"brute force is limited to {max_items} items")
    start = perf_counter()
    best_x: np.ndarray | None = None
    best_obj = -np.inf
    for bits in product((0, 1), repeat=instance.n_items):
        x = np.asarray(bits, dtype=int)
        if instance.feasible(x):
            obj = instance.objective(x)
            if obj > best_obj + 1e-12:
                best_obj = obj
                best_x = x.copy()
    elapsed = perf_counter() - start
    return SolveResult(
        x=best_x,
        objective=None if best_x is None else float(best_obj),
        success=best_x is not None,
        status=0 if best_x is not None else 2,
        message="brute-force optimum" if best_x is not None else "infeasible",
        elapsed_seconds=float(elapsed),
        mip_node_count=None,
        mip_gap=0.0 if best_x is not None else None,
        fixed_count=0,
    )
