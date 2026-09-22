import numpy as np

from neural_diving.problem import generate_instance
from neural_diving.solver import brute_force_solve, solve_mip


def test_highs_matches_bruteforce_on_tiny_instance():
    instance = generate_instance(11, n_items=12, n_constraints=3)
    highs = solve_mip(instance, mip_rel_gap=0.0)
    brute = brute_force_solve(instance)
    assert highs.success and brute.success
    assert abs(highs.objective - brute.objective) < 1e-7


def test_fixed_submip_honors_assignment():
    instance = generate_instance(12, n_items=14, n_constraints=3)
    solved = solve_mip(instance, fixed={0: 0, 1: 0})
    assert solved.success
    assert solved.x is not None
    assert np.array_equal(solved.x[:2], np.array([0, 0]))
