from __future__ import annotations

import itertools

import numpy as np
import pytest

from ltb_milp.branching import (
    PseudocostState,
    fractional_candidates,
    pseudocost_branch,
    reliability_branch,
    strong_branch_scores,
)
from ltb_milp.problem import BinaryPackingMILP, generate_binary_packing, solve_lp_relaxation
from ltb_milp.solver import solve_branch_and_bound


def brute_force_optimum(problem: BinaryPackingMILP) -> float:
    best = float("-inf")
    for bits in itertools.product((0.0, 1.0), repeat=problem.n_vars):
        x = np.asarray(bits)
        if np.all(problem.A @ x <= problem.b + 1e-9):
            best = max(best, float(problem.c @ x))
    return best


def test_lp_relaxation_upper_bounds_integer_optimum() -> None:
    problem = generate_binary_packing(7, 3, seed=4)
    lp = solve_lp_relaxation(problem)
    assert lp.feasible
    assert lp.objective + 1e-8 >= brute_force_optimum(problem)


def test_branch_and_bound_matches_brute_force() -> None:
    problem = generate_binary_packing(8, 3, seed=12)
    expected = brute_force_optimum(problem)
    for policy in ("most_fractional", "pseudocost", "reliability", "strong"):
        result = solve_branch_and_bound(problem, policy=policy)
        assert result.optimal
        assert result.objective == expected
        assert result.solution is not None
        assert np.all(problem.A @ result.solution <= problem.b + 1e-8)


def test_pseudocost_state_updates_and_selects_candidate() -> None:
    state = PseudocostState.zeros(3)
    state.update(0, "down", gain=2.0, distance=0.5)
    state.update(0, "up", gain=3.0, distance=0.5)
    state.update(1, "down", gain=0.5, distance=0.5)
    state.update(1, "up", gain=0.5, distance=0.5)
    decision = pseudocost_branch(np.array([0.5, 0.5, 1.0]), state)
    assert decision.variable == 0


def test_reliability_branch_probes_unreliable_candidates() -> None:
    problem = generate_binary_packing(10, 4, seed=2)
    lp = solve_lp_relaxation(problem)
    assert lp.feasible and lp.x is not None
    candidates = fractional_candidates(lp.x)
    if candidates.size:
        state = PseudocostState.zeros(problem.n_vars)
        decision = reliability_branch(
            problem,
            lp.x,
            lp.objective,
            {},
            state,
            reliability_threshold=2,
        )
        assert decision.variable in candidates
        assert decision.extra_lp_solves == 2 * candidates.size
        assert np.all(state.down_count[candidates] == 1)
        assert np.all(state.up_count[candidates] == 1)


def test_strong_branch_scores_cover_all_fractional_candidates() -> None:
    problem = generate_binary_packing(10, 4, seed=2)
    lp = solve_lp_relaxation(problem)
    assert lp.feasible and lp.x is not None
    candidates = fractional_candidates(lp.x)
    if candidates.size:
        scored, scores, solves = strong_branch_scores(problem, lp.x, lp.objective)
        assert np.array_equal(scored, candidates)
        assert scores.shape == candidates.shape
        assert solves == 2 * candidates.size
        assert np.all(scores >= 0)


def test_tightness_controls_rhs_scale() -> None:
    tight = generate_binary_packing(10, 4, seed=9, tightness=0.3)
    loose = generate_binary_packing(10, 4, seed=9, tightness=0.7)
    assert np.array_equal(tight.A, loose.A)
    assert np.array_equal(tight.c, loose.c)
    assert np.all(tight.b < loose.b)


def test_tightness_validation() -> None:
    with pytest.raises(ValueError, match="tightness"):
        generate_binary_packing(8, 3, tightness=1.0)
