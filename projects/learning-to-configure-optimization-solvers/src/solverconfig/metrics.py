from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .solver import SolveResult


@dataclass(frozen=True)
class MethodMetrics:
    method: str
    observations: int
    solved_rate: float
    feasible_rate: float
    runtime_mean: float
    runtime_std: float
    runtime_median: float
    runtime_p90: float
    node_count_mean: float | None
    gap_mean: float | None
    par10_mean: float
    par10_std: float
    normalized_par10_mean: float
    regret_mean: float
    regret_std: float
    regret_median: float
    regret_ci95_low: float
    regret_ci95_high: float


def par10(result: SolveResult, time_limit: float) -> float:
    return result.runtime if result.optimal else 10.0 * time_limit


def bootstrap_mean_ci(
    values: np.ndarray, *, seed: int = 0, draws: int = 1000
) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_method(
    *,
    method: str,
    selected: list[SolveResult],
    oracle_par10: np.ndarray,
    time_limit: float,
    ci_seed: int,
) -> MethodMetrics:
    if not selected:
        raise ValueError("cannot summarize an empty result list")
    par = np.array([par10(result, time_limit) for result in selected], dtype=float)
    runtime = np.array([result.runtime for result in selected], dtype=float)
    feasible = np.array([result.feasible for result in selected], dtype=float)
    solved = np.array([result.optimal for result in selected], dtype=float)
    regret = par - oracle_par10
    normalized = par / np.maximum(oracle_par10, 1e-9)
    nodes = np.array([r.node_count for r in selected if r.node_count is not None], dtype=float)
    gaps = np.array([r.mip_gap for r in selected if r.mip_gap is not None], dtype=float)
    ci_low, ci_high = bootstrap_mean_ci(regret, seed=ci_seed)
    return MethodMetrics(
        method=method,
        observations=len(selected),
        solved_rate=float(solved.mean()),
        feasible_rate=float(feasible.mean()),
        runtime_mean=float(runtime.mean()),
        runtime_std=float(runtime.std()),
        runtime_median=float(np.median(runtime)),
        runtime_p90=float(np.quantile(runtime, 0.90)),
        node_count_mean=float(nodes.mean()) if nodes.size else None,
        gap_mean=float(gaps.mean()) if gaps.size else None,
        par10_mean=float(par.mean()),
        par10_std=float(par.std()),
        normalized_par10_mean=float(normalized.mean()),
        regret_mean=float(regret.mean()),
        regret_std=float(regret.std()),
        regret_median=float(np.median(regret)),
        regret_ci95_low=ci_low,
        regret_ci95_high=ci_high,
    )
