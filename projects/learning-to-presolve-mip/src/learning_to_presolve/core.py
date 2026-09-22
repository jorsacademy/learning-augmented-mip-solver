from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

PresolveConfig = Literal["none", "fixing", "rows", "full"]
CONFIGS: tuple[PresolveConfig, ...] = ("none", "fixing", "rows", "full")

_TOL = 1e-9


@dataclass(frozen=True)
class BinaryPackingMIP:
    """Binary packing MIP: maximize profits @ x subject to A @ x <= b."""

    profits: np.ndarray
    A: np.ndarray
    b: np.ndarray

    def __post_init__(self) -> None:
        profits = np.asarray(self.profits, dtype=float)
        A = np.asarray(self.A, dtype=float)
        b = np.asarray(self.b, dtype=float)
        if profits.ndim != 1 or A.ndim != 2 or b.ndim != 1:
            raise ValueError("profits and b must be vectors and A must be a matrix")
        if A.shape != (b.size, profits.size):
            raise ValueError("A shape must match b and profits")
        if profits.size < 1 or b.size < 1:
            raise ValueError("problem must contain at least one variable and one constraint")
        if not np.all(np.isfinite(profits)) or not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
            raise ValueError("problem data must be finite")
        if np.any(A < 0.0) or np.any(b < 0.0):
            raise ValueError("packing coefficients and right-hand sides must be nonnegative")
        object.__setattr__(self, "profits", profits)
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "b", b)

    @property
    def n_vars(self) -> int:
        return int(self.profits.size)

    @property
    def n_constraints(self) -> int:
        return int(self.b.size)

    @property
    def nnz(self) -> int:
        return int(np.count_nonzero(self.A))


@dataclass(frozen=True)
class PresolveReport:
    config: PresolveConfig
    fixed_zero: tuple[int, ...]
    fixed_one: tuple[int, ...]
    removed_rows: tuple[int, ...]
    checks: int

    @property
    def fixed_count(self) -> int:
        return len(self.fixed_zero) + len(self.fixed_one)


@dataclass(frozen=True)
class PresolvedMIP:
    profits: np.ndarray
    A: np.ndarray
    b: np.ndarray
    kept_indices: np.ndarray
    fixed_values: dict[int, int]
    objective_offset: float
    original_n_vars: int
    report: PresolveReport

    @property
    def residual_size(self) -> int:
        return int(self.profits.size + self.b.size)

    @property
    def residual_nnz(self) -> int:
        return int(np.count_nonzero(self.A))


@dataclass(frozen=True)
class SolveResult:
    objective: float
    x: np.ndarray
    nodes: int
    optimal: bool


@dataclass(frozen=True)
class ConfigEvaluation:
    config: PresolveConfig
    objective: float
    nodes: int
    residual_vars: int
    residual_rows: int
    residual_nnz: int
    checks: int
    work_proxy: float


@dataclass(frozen=True)
class OracleResult:
    selected: PresolveConfig
    evaluations: tuple[ConfigEvaluation, ...]

    def by_config(self, config: PresolveConfig) -> ConfigEvaluation:
        return self.evaluations[CONFIGS.index(config)]


def generate_packing(
    n_vars: int = 32,
    n_constraints: int = 10,
    seed: int = 0,
) -> BinaryPackingMIP:
    """Generate a reproducible family with heterogeneous safe-presolve opportunities."""

    if n_vars < 8 or n_constraints < 4:
        raise ValueError("use at least 8 variables and 4 constraints")
    rng = np.random.default_rng(seed)

    density = rng.uniform(0.25, 0.68)
    mask = rng.random((n_constraints, n_vars)) < density
    weights = rng.integers(1, 16, size=(n_constraints, n_vars)).astype(float)
    A = mask * weights

    for i in range(n_constraints):
        if not np.any(A[i] > 0.0):
            j = int(rng.integers(n_vars))
            A[i, j] = float(rng.integers(1, 16))

    profits = rng.integers(2, 35, size=n_vars).astype(float)
    row_sums = A.sum(axis=1)
    tightness = rng.uniform(0.35, 0.58, size=n_constraints)
    b = np.maximum(1.0, np.floor(tightness * row_sums))

    mode = seed % 4
    available = list(rng.permutation(n_vars))

    if mode in {1, 3}:
        count = max(2, n_vars // 10)
        for _ in range(min(count, len(available))):
            j = int(available.pop())
            kind = int(rng.integers(3))
            if kind == 0:
                profits[j] = -float(rng.integers(1, 10))
            elif kind == 1:
                i = int(rng.integers(n_constraints))
                A[i, j] = b[i] + float(rng.integers(1, 7))
            else:
                A[:, j] = 0.0
                profits[j] = float(rng.integers(4, 35))

    if mode in {2, 3}:
        pairs = max(1, n_constraints // 5)
        protected: set[int] = set()
        for _ in range(pairs):
            source_choices = [i for i in range(n_constraints) if i not in protected]
            if not source_choices:
                break
            source = int(rng.choice(source_choices))
            target_choices = [i for i in range(n_constraints) if i != source and i not in protected]
            if not target_choices:
                break
            target = int(rng.choice(target_choices))
            factor = float(rng.uniform(0.45, 0.9))
            A[target] = factor * A[source]
            b[target] = b[source] + float(rng.integers(1, 6))
            protected.update({source, target})

    return BinaryPackingMIP(profits=profits, A=A, b=b)


def _find_fixed_variables(problem: BinaryPackingMIP) -> tuple[dict[int, int], int]:
    """Return optimality-preserving binary fixings for nonnegative packing models."""

    fixed: dict[int, int] = {}
    checks = 0
    for j in range(problem.n_vars):
        checks += 1
        if problem.profits[j] <= 0.0:
            fixed[j] = 0
            continue

        column = problem.A[:, j]
        checks += problem.n_constraints
        if np.any(column > problem.b + _TOL):
            fixed[j] = 0
            continue

        checks += 1
        if np.all(np.abs(column) <= _TOL):
            fixed[j] = 1
    return fixed, checks


def _find_redundant_rows(A: np.ndarray, b: np.ndarray) -> tuple[set[int], int]:
    """Find rows implied componentwise by another packing row."""

    redundant: set[int] = set()
    checks = 0
    m, n = A.shape
    for weak in range(m):
        if weak in redundant:
            continue
        if np.all(np.abs(A[weak]) <= _TOL) and b[weak] >= -_TOL:
            redundant.add(weak)
            checks += n
            continue
        for strong in range(m):
            if strong == weak or strong in redundant:
                continue
            checks += n + 1
            if b[strong] <= b[weak] + _TOL and np.all(A[strong] >= A[weak] - _TOL):
                redundant.add(weak)
                break
    return redundant, checks


def apply_presolve(problem: BinaryPackingMIP, config: PresolveConfig) -> PresolvedMIP:
    """Apply only transformations whose validity is independent of the learned selector."""

    if config not in CONFIGS:
        raise ValueError(f"unknown presolve configuration: {config}")

    profits = problem.profits.copy()
    A = problem.A.copy()
    b = problem.b.copy()
    kept = np.arange(problem.n_vars, dtype=int)
    fixed: dict[int, int] = {}
    objective_offset = 0.0
    checks = 0
    fixed_zero: list[int] = []
    fixed_one: list[int] = []

    if config in {"fixing", "full"}:
        fixed, fixing_checks = _find_fixed_variables(problem)
        checks += fixing_checks
        if fixed:
            for original_index, value in sorted(fixed.items()):
                if value == 1:
                    b = b - A[:, original_index]
                    objective_offset += float(problem.profits[original_index])
                    fixed_one.append(original_index)
                else:
                    fixed_zero.append(original_index)
            keep_mask = np.array([idx not in fixed for idx in kept], dtype=bool)
            profits = profits[keep_mask]
            A = A[:, keep_mask]
            kept = kept[keep_mask]

    if np.any(b < -_TOL):
        raise RuntimeError("presolve produced an infeasible residual model")

    removed_rows: tuple[int, ...] = ()
    if config in {"rows", "full"}:
        redundant, row_checks = _find_redundant_rows(A, b)
        checks += row_checks
        if redundant:
            removed_rows = tuple(sorted(redundant))
            keep_rows = np.array([i not in redundant for i in range(b.size)], dtype=bool)
            A = A[keep_rows]
            b = b[keep_rows]

    report = PresolveReport(
        config=config,
        fixed_zero=tuple(fixed_zero),
        fixed_one=tuple(fixed_one),
        removed_rows=removed_rows,
        checks=checks,
    )
    return PresolvedMIP(
        profits=profits,
        A=A,
        b=b,
        kept_indices=kept,
        fixed_values=fixed,
        objective_offset=objective_offset,
        original_n_vars=problem.n_vars,
        report=report,
    )


def reconstruct_solution(presolved: PresolvedMIP, residual_x: np.ndarray) -> np.ndarray:
    residual_x = np.asarray(residual_x, dtype=float)
    if residual_x.shape != (presolved.profits.size,):
        raise ValueError("residual solution has wrong shape")
    x = np.zeros(presolved.original_n_vars, dtype=float)
    x[presolved.kept_indices] = residual_x
    for index, value in presolved.fixed_values.items():
        x[index] = float(value)
    return x


def solve_presolved(presolved: PresolvedMIP) -> SolveResult:
    """Solve the residual binary MIP with HiGHS' own presolve disabled."""

    if presolved.profits.size == 0:
        x = reconstruct_solution(presolved, np.zeros(0, dtype=float))
        return SolveResult(
            objective=float(presolved.objective_offset),
            x=x,
            nodes=0,
            optimal=True,
        )

    constraints: LinearConstraint | None = None
    if presolved.b.size:
        constraints = LinearConstraint(presolved.A, -np.inf, presolved.b)
    result = milp(
        c=-presolved.profits,
        integrality=np.ones(presolved.profits.size, dtype=int),
        bounds=Bounds(np.zeros(presolved.profits.size), np.ones(presolved.profits.size)),
        constraints=constraints,
        options={"presolve": False},
    )
    if not result.success or result.fun is None or result.x is None:
        raise RuntimeError(f"MILP solve failed: {result.message}")
    residual_x = np.rint(np.asarray(result.x, dtype=float))
    x = reconstruct_solution(presolved, residual_x)
    nodes = int(getattr(result, "mip_node_count", 0) or 0)
    return SolveResult(
        objective=float(-result.fun + presolved.objective_offset),
        x=x,
        nodes=nodes,
        optimal=True,
    )


def is_feasible(problem: BinaryPackingMIP, x: np.ndarray, tol: float = 1e-7) -> bool:
    x = np.asarray(x, dtype=float)
    if x.shape != (problem.n_vars,):
        return False
    if np.any(x < -tol) or np.any(x > 1.0 + tol):
        return False
    if np.any(np.abs(x - np.rint(x)) > tol):
        return False
    return bool(np.all(problem.A @ x <= problem.b + tol))


def validate_solution(problem: BinaryPackingMIP, result: SolveResult, tol: float = 1e-7) -> None:
    if not is_feasible(problem, result.x, tol=tol):
        raise AssertionError("reconstructed solution is infeasible in the original model")
    objective = float(problem.profits @ result.x)
    if not np.isclose(objective, result.objective, atol=tol, rtol=0.0):
        raise AssertionError("reported objective does not match reconstructed solution")


def instance_features(problem: BinaryPackingMIP) -> np.ndarray:
    """Compute cheap, transparent structural features before custom presolve."""

    row_sums = problem.A.sum(axis=1)
    col_nnz = np.count_nonzero(problem.A, axis=0)
    density = float(problem.nnz / problem.A.size)
    tightness = problem.b / np.maximum(row_sums, 1.0)
    nonpositive_profit = float(np.mean(problem.profits <= 0.0))
    individually_infeasible = float(
        np.mean(np.any(problem.A > problem.b[:, None] + _TOL, axis=0))
    )
    zero_columns = float(np.mean(col_nnz == 0))

    norms = np.linalg.norm(problem.A, axis=1)
    normalized = problem.A / np.maximum(norms[:, None], _TOL)
    similarity = normalized @ normalized.T
    if problem.n_constraints > 1:
        similarity = similarity - np.eye(problem.n_constraints)
        max_similarity = float(np.max(similarity))
    else:
        max_similarity = 0.0

    positive_coeffs = problem.A[problem.A > 0.0]
    coeff_cv = float(np.std(positive_coeffs) / max(np.mean(positive_coeffs), _TOL))
    positive_profits = problem.profits[problem.profits > 0.0]
    profit_cv = float(np.std(positive_profits) / max(np.mean(positive_profits), _TOL))

    return np.asarray(
        [
            np.log1p(problem.n_vars),
            np.log1p(problem.n_constraints),
            density,
            float(np.mean(tightness)),
            float(np.std(tightness)),
            nonpositive_profit,
            individually_infeasible,
            zero_columns,
            float(np.mean(col_nnz) / problem.n_constraints),
            max_similarity,
            coeff_cv,
            profit_cv,
        ],
        dtype=np.float32,
    )


def work_proxy(evaluation: ConfigEvaluation) -> float:
    """Return a deterministic proxy balancing residual B&B work and presolve effort."""

    residual_complexity = (
        evaluation.residual_nnz
        + 2.0 * evaluation.residual_vars
        + 2.0 * evaluation.residual_rows
    )
    return float((evaluation.nodes + 1.0) * residual_complexity + 0.025 * evaluation.checks)


def evaluate_configuration(
    problem: BinaryPackingMIP,
    config: PresolveConfig,
) -> ConfigEvaluation:
    reduced = apply_presolve(problem, config)
    solved = solve_presolved(reduced)
    validate_solution(problem, solved)
    draft = ConfigEvaluation(
        config=config,
        objective=solved.objective,
        nodes=solved.nodes,
        residual_vars=int(reduced.profits.size),
        residual_rows=int(reduced.b.size),
        residual_nnz=reduced.residual_nnz,
        checks=reduced.report.checks,
        work_proxy=0.0,
    )
    return ConfigEvaluation(
        config=draft.config,
        objective=draft.objective,
        nodes=draft.nodes,
        residual_vars=draft.residual_vars,
        residual_rows=draft.residual_rows,
        residual_nnz=draft.residual_nnz,
        checks=draft.checks,
        work_proxy=work_proxy(draft),
    )


def oracle_configuration(problem: BinaryPackingMIP) -> OracleResult:
    """Evaluate every safe configuration and choose the lowest deterministic work proxy."""

    evaluations = tuple(evaluate_configuration(problem, config) for config in CONFIGS)
    baseline_objective = evaluations[0].objective
    for evaluation in evaluations:
        if not np.isclose(evaluation.objective, baseline_objective, atol=1e-7, rtol=0.0):
            raise AssertionError(
                f"configuration {evaluation.config} changed the exact objective "
                f"from {baseline_objective} to {evaluation.objective}"
            )
    selected = min(evaluations, key=lambda item: (item.work_proxy, CONFIGS.index(item.config)))
    return OracleResult(selected=selected.config, evaluations=evaluations)
