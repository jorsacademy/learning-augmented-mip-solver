from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .problem import BinaryPackingMIP
from .solver import SolveResult, solve_mip


@dataclass(frozen=True)
class DiveResult:
    solve: SolveResult
    requested_fraction: float
    fraction_used: float
    fixed_count: int
    attempts: int
    fallback_used: bool
    candidate_count: int
    method: str
    elapsed_seconds: float


def partial_assignment(
    probabilities: np.ndarray,
    *,
    fraction: float,
    rng: np.random.Generator | None = None,
    stochastic: bool = False,
) -> dict[int, int]:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be a one-dimensional vector in [0,1]")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0,1]")
    k = int(np.floor(fraction * probabilities.size + 1e-12))
    if k <= 0:
        return {}
    confidence = np.abs(probabilities - 0.5)
    chosen = np.argsort(-confidence, kind="stable")[:k]
    if stochastic:
        rng = rng or np.random.default_rng(0)
        values = rng.binomial(1, probabilities[chosen])
    else:
        values = (probabilities[chosen] >= 0.5).astype(int)
    return {int(i): int(v) for i, v in zip(chosen, values, strict=True)}


def _best_success(results: list[SolveResult]) -> SolveResult | None:
    feasible = [result for result in results if result.success and result.objective is not None]
    if not feasible:
        return None
    return max(feasible, key=lambda r: float(r.objective))


def neural_dive(
    instance: BinaryPackingMIP,
    probabilities: np.ndarray,
    *,
    fractions: tuple[float, ...] = (0.70, 0.50, 0.30),
    stochastic_samples: int = 3,
    seed: int = 0,
    time_limit: float | None = 10.0,
) -> DiveResult:
    """Fix high-confidence predictions and let HiGHS complete each sub-MIP.

    At each fixation fraction we try one deterministic mode assignment plus optional stochastic
    assignments. If all are infeasible, the routine backs off to a smaller fixation fraction.
    A final full-MIP fallback guarantees that the caller receives a solver result.
    """
    if stochastic_samples < 0:
        raise ValueError("stochastic_samples must be nonnegative")
    start = perf_counter()
    rng = np.random.default_rng(seed)
    attempts = 0
    requested = float(fractions[0]) if fractions else 0.0

    for fraction in fractions:
        candidates: list[SolveResult] = []
        deterministic = partial_assignment(probabilities, fraction=fraction, stochastic=False)
        attempts += 1
        candidates.append(solve_mip(instance, fixed=deterministic, time_limit=time_limit))
        for _ in range(stochastic_samples):
            fixed = partial_assignment(
                probabilities, fraction=fraction, stochastic=True, rng=rng
            )
            attempts += 1
            candidates.append(solve_mip(instance, fixed=fixed, time_limit=time_limit))
        best = _best_success(candidates)
        if best is not None:
            return DiveResult(
                solve=best,
                requested_fraction=requested,
                fraction_used=float(fraction),
                fixed_count=best.fixed_count,
                attempts=attempts,
                fallback_used=float(fraction) != requested,
                candidate_count=len(candidates),
                method="neural_dive",
                elapsed_seconds=float(perf_counter() - start),
            )

    attempts += 1
    full = solve_mip(instance, time_limit=time_limit)
    return DiveResult(
        solve=full,
        requested_fraction=requested,
        fraction_used=0.0,
        fixed_count=0,
        attempts=attempts,
        fallback_used=True,
        candidate_count=1,
        method="neural_dive",
        elapsed_seconds=float(perf_counter() - start),
    )


def random_dive(
    instance: BinaryPackingMIP,
    *,
    fraction: float = 0.70,
    seed: int = 0,
    samples: int = 4,
    time_limit: float | None = 10.0,
) -> DiveResult:
    """Random partial-fixing baseline with the same sub-MIP completion mechanism."""
    rng = np.random.default_rng(seed)
    k = int(np.floor(fraction * instance.n_items))
    attempts = 0
    results: list[SolveResult] = []
    start = perf_counter()
    for _ in range(samples):
        indices = rng.choice(instance.n_items, size=k, replace=False) if k else np.array([], int)
        fixed = {int(i): int(rng.integers(0, 2)) for i in indices}
        attempts += 1
        results.append(solve_mip(instance, fixed=fixed, time_limit=time_limit))
    best = _best_success(results)
    if best is None:
        attempts += 1
        best = solve_mip(instance, time_limit=time_limit)
        used_fraction = 0.0
        fallback = True
    else:
        used_fraction = fraction
        fallback = False
    return DiveResult(
        solve=best,
        requested_fraction=fraction,
        fraction_used=float(used_fraction),
        fixed_count=best.fixed_count,
        attempts=attempts,
        fallback_used=fallback,
        candidate_count=samples,
        method="random_dive",
        elapsed_seconds=float(perf_counter() - start),
    )
