"""Branching rules and strong-branching scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltb_milp.problem import BinaryPackingMILP, solve_lp_relaxation


@dataclass(frozen=True)
class BranchDecision:
    variable: int
    extra_lp_solves: int = 0


@dataclass
class PseudocostState:
    """Running down/up objective degradation estimates for each variable."""

    down_sum: np.ndarray
    down_count: np.ndarray
    up_sum: np.ndarray
    up_count: np.ndarray

    @classmethod
    def zeros(cls, n_vars: int) -> PseudocostState:
        if n_vars <= 0:
            raise ValueError("n_vars must be positive")
        return cls(
            down_sum=np.zeros(n_vars, dtype=float),
            down_count=np.zeros(n_vars, dtype=int),
            up_sum=np.zeros(n_vars, dtype=float),
            up_count=np.zeros(n_vars, dtype=int),
        )

    def update(self, variable: int, direction: str, gain: float, distance: float) -> None:
        """Update a unit pseudocost from one observed child LP bound degradation."""
        if direction not in {"down", "up"}:
            raise ValueError("direction must be 'down' or 'up'")
        if distance <= 0:
            return
        unit_cost = max(0.0, gain) / distance
        if direction == "down":
            self.down_sum[variable] += unit_cost
            self.down_count[variable] += 1
        else:
            self.up_sum[variable] += unit_cost
            self.up_count[variable] += 1

    def means(self, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return down/up mean pseudocosts, using global observed means as fallback."""
        observed_down = self.down_sum[self.down_count > 0] / self.down_count[self.down_count > 0]
        observed_up = self.up_sum[self.up_count > 0] / self.up_count[self.up_count > 0]
        fallback_down = float(observed_down.mean()) if observed_down.size else 1.0
        fallback_up = float(observed_up.mean()) if observed_up.size else 1.0

        down = np.full(candidates.size, fallback_down, dtype=float)
        up = np.full(candidates.size, fallback_up, dtype=float)
        for pos, variable in enumerate(candidates):
            if self.down_count[variable] > 0:
                down[pos] = self.down_sum[variable] / self.down_count[variable]
            if self.up_count[variable] > 0:
                up[pos] = self.up_sum[variable] / self.up_count[variable]
        return down, up


def fractional_candidates(x: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """Return indices of variables that are fractional in an LP solution."""
    return np.flatnonzero((x > tol) & (x < 1.0 - tol))


def _combine_gains(down_gain: np.ndarray, up_gain: np.ndarray) -> np.ndarray:
    return np.minimum(down_gain, up_gain) + 1e-6 * np.maximum(down_gain, up_gain)


def most_fractional(x: np.ndarray) -> BranchDecision:
    """Choose the candidate closest to 0.5."""
    candidates = fractional_candidates(x)
    if candidates.size == 0:
        raise ValueError("no fractional branching candidates")
    distances = np.abs(x[candidates] - 0.5)
    return BranchDecision(int(candidates[np.argmin(distances)]))


def pseudocost_branch(x: np.ndarray, state: PseudocostState) -> BranchDecision:
    """Choose the candidate with the largest estimated two-sided bound degradation."""
    candidates = fractional_candidates(x)
    if candidates.size == 0:
        raise ValueError("no fractional branching candidates")
    down_means, up_means = state.means(candidates)
    down_gain = down_means * x[candidates]
    up_gain = up_means * (1.0 - x[candidates])
    scores = _combine_gains(down_gain, up_gain)
    return BranchDecision(int(candidates[np.argmax(scores)]))


def reliability_branch(
    problem: BinaryPackingMILP,
    x: np.ndarray,
    parent_objective: float,
    bounds: dict[int, tuple[float, float]],
    state: PseudocostState,
    *,
    reliability_threshold: int = 2,
) -> BranchDecision:
    """Use pseudocosts when reliable and strong probing for under-observed candidates."""
    if reliability_threshold <= 0:
        raise ValueError("reliability_threshold must be positive")
    candidates = fractional_candidates(x)
    if candidates.size == 0:
        raise ValueError("no fractional branching candidates")

    down_means, up_means = state.means(candidates)
    down_gain = down_means * x[candidates]
    up_gain = up_means * (1.0 - x[candidates])
    extra_lp_solves = 0
    fallback_gain = abs(parent_objective) + 1.0

    for pos, variable in enumerate(candidates):
        reliable = (
            state.down_count[variable] >= reliability_threshold
            and state.up_count[variable] >= reliability_threshold
        )
        if reliable:
            continue

        observed_gains: list[float] = []
        for direction, branch_value, distance in (
            ("down", 0.0, float(x[variable])),
            ("up", 1.0, float(1.0 - x[variable])),
        ):
            child_bounds = dict(bounds)
            child_bounds[int(variable)] = (branch_value, branch_value)
            child = solve_lp_relaxation(problem, child_bounds)
            extra_lp_solves += 1
            gain = (
                fallback_gain
                if not child.feasible
                else max(0.0, parent_objective - child.objective)
            )
            state.update(int(variable), direction, gain, distance)
            observed_gains.append(gain)
        down_gain[pos], up_gain[pos] = observed_gains

    scores = _combine_gains(down_gain, up_gain)
    return BranchDecision(
        int(candidates[np.argmax(scores)]),
        extra_lp_solves=extra_lp_solves,
    )


def strong_branch_scores(
    problem: BinaryPackingMILP,
    x: np.ndarray,
    parent_objective: float,
    bounds: dict[int, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Score all fractional candidates by the minimum child bound improvement."""
    candidates = fractional_candidates(x)
    if candidates.size == 0:
        raise ValueError("no fractional branching candidates")

    scores = np.zeros(candidates.size, dtype=float)
    base_bounds = dict(bounds or {})
    solves = 0
    fallback_gain = abs(parent_objective) + 1.0

    for pos, variable in enumerate(candidates):
        child_objectives: list[float | None] = []
        for value in (0.0, 1.0):
            child_bounds = dict(base_bounds)
            child_bounds[int(variable)] = (value, value)
            child = solve_lp_relaxation(problem, child_bounds)
            solves += 1
            child_objectives.append(child.objective if child.feasible else None)

        gains = [
            fallback_gain if objective is None else max(0.0, parent_objective - objective)
            for objective in child_objectives
        ]
        scores[pos] = min(gains) + 1e-6 * max(gains)

    return candidates, scores, solves


def strong_branch(
    problem: BinaryPackingMILP,
    x: np.ndarray,
    parent_objective: float,
    bounds: dict[int, tuple[float, float]] | None = None,
) -> BranchDecision:
    candidates, scores, solves = strong_branch_scores(problem, x, parent_objective, bounds)
    return BranchDecision(int(candidates[np.argmax(scores)]), extra_lp_solves=solves)
