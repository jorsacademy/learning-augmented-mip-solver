from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .problem import BinaryPackingMIP

PresolveConfig = Literal["none", "fixing", "redundancy", "full"]
PRESOLVE_CONFIGS: tuple[PresolveConfig, ...] = ("none", "fixing", "redundancy", "full")


@dataclass(frozen=True)
class PresolvedProblem:
    problem: BinaryPackingMIP
    kept_indices: np.ndarray
    fixed_values: dict[int, int]
    objective_offset: float
    checks: int


def _safe_fixings(problem: BinaryPackingMIP) -> tuple[dict[int, int], int]:
    fixed: dict[int, int] = {}
    checks = 0
    for j in range(problem.n_vars):
        checks += 1
        if problem.profits[j] <= 0:
            fixed[j] = 0
            continue
        column = problem.A[:, j]
        checks += problem.n_constraints
        if np.any(column > problem.b + 1e-12):
            fixed[j] = 0
            continue
        checks += 1
        if np.allclose(column, 0.0):
            fixed[j] = 1
    return fixed, checks


def _redundant_rows(A: np.ndarray, b: np.ndarray) -> tuple[set[int], int]:
    redundant: set[int] = set()
    checks = 0
    for i in range(b.size):
        for j in range(b.size):
            if i == j or i in redundant:
                continue
            checks += A.shape[1]
            if np.all(A[j] >= A[i] - 1e-12) and b[j] <= b[i] + 1e-12:
                redundant.add(i)
                break
    return redundant, checks


def apply_presolve(problem: BinaryPackingMIP, config: PresolveConfig) -> PresolvedProblem:
    if config not in PRESOLVE_CONFIGS:
        raise ValueError("unknown presolve configuration")

    profits = problem.profits.copy()
    A = problem.A.copy()
    b = problem.b.copy()
    kept = np.arange(problem.n_vars)
    fixed: dict[int, int] = {}
    objective_offset = 0.0
    checks = 0

    if config in {"fixing", "full"}:
        fixed, fixing_checks = _safe_fixings(problem)
        checks += fixing_checks
        for original, value in fixed.items():
            if value == 1:
                b -= A[:, original]
                objective_offset += float(profits[original])
        keep_mask = np.asarray([index not in fixed for index in kept], dtype=bool)
        profits = profits[keep_mask]
        A = A[:, keep_mask]
        kept = kept[keep_mask]

    if np.any(b < -1e-9):
        raise RuntimeError("safe presolve produced an infeasible residual model")

    if config in {"redundancy", "full"} and b.size > 1:
        redundant, row_checks = _redundant_rows(A, b)
        checks += row_checks
        if redundant:
            keep_rows = np.asarray([i not in redundant for i in range(b.size)], dtype=bool)
            A = A[keep_rows]
            b = b[keep_rows]

    residual = BinaryPackingMIP(profits=profits, A=A, b=b)
    return PresolvedProblem(
        problem=residual,
        kept_indices=kept,
        fixed_values=fixed,
        objective_offset=objective_offset,
        checks=checks,
    )


def instance_features(problem: BinaryPackingMIP) -> np.ndarray:
    row_sum = problem.A.sum(axis=1)
    density = float(np.count_nonzero(problem.A) / problem.A.size)
    tightness = float(np.mean(problem.b / np.maximum(row_sum, 1.0)))
    nonpositive = float(np.mean(problem.profits <= 0))
    infeasible_if_one = float(np.mean(np.any(problem.A > problem.b[:, None] + 1e-12, axis=0)))
    zero_columns = float(np.mean(np.all(np.isclose(problem.A, 0.0), axis=0)))
    redundant, _ = _redundant_rows(problem.A, problem.b)
    redundancy_fraction = len(redundant) / max(1, problem.n_constraints)
    return np.asarray(
        [
            np.log1p(problem.n_vars),
            np.log1p(problem.n_constraints),
            density,
            tightness,
            nonpositive,
            infeasible_if_one,
            zero_columns,
            redundancy_fraction,
        ],
        dtype=np.float32,
    )
