from itertools import product

import numpy as np

from lamip import (
    BranchScorer,
    CutScorer,
    LearnedComponents,
    PresolveSelector,
    PrimalScorer,
    SolverConfig,
    apply_presolve,
    generate_binary_packing,
    generate_cover_cuts,
    solve,
    solve_reference,
)
from lamip.solver import _solve_lp


def test_safe_presolve_preserves_optimum() -> None:
    for seed in range(4):
        problem = generate_binary_packing(12, 4, seed)
        expected = solve_reference(problem, presolve=False).objective
        for config in ("none", "fixing", "redundancy", "full"):
            reduced = apply_presolve(problem, config)
            actual = solve_reference(reduced.problem, presolve=False).objective
            actual += reduced.objective_offset
            assert np.isclose(actual, expected, atol=1e-7)


def test_generated_cover_cuts_are_valid() -> None:
    problem = generate_binary_packing(8, 3, seed=11)
    lp = _solve_lp(problem, [])
    assert lp is not None
    cuts = generate_cover_cuts(problem, lp.x)
    feasible = []
    for values in product((0.0, 1.0), repeat=problem.n_vars):
        x = np.asarray(values)
        if np.all(problem.A @ x <= problem.b + 1e-9):
            feasible.append(x)
    for cut in cuts:
        for x in feasible:
            assert x[list(cut.indices)].sum() <= cut.rhs + 1e-9


def test_classical_and_random_learned_pipelines_remain_exact() -> None:
    problem = generate_binary_packing(12, 4, seed=22)
    expected = solve_reference(problem, presolve=False).objective
    classical = solve(problem, SolverConfig())
    assert classical.optimal
    assert np.isclose(classical.objective, expected, atol=1e-7)

    learned = LearnedComponents(
        presolve=PresolveSelector(16),
        cuts=CutScorer(16),
        branching=BranchScorer(16),
        primal=PrimalScorer(16),
    )
    augmented = solve(
        problem,
        SolverConfig(
            presolve="learned",
            cuts="learned",
            branching="learned",
            primal="learned",
            max_nodes=5000,
        ),
        learned,
    )
    assert augmented.optimal
    assert np.isclose(augmented.objective, expected, atol=1e-7)
