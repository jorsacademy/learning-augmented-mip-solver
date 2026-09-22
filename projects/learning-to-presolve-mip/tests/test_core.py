import numpy as np

from learning_to_presolve import (
    CONFIGS,
    BinaryPackingMIP,
    apply_presolve,
    generate_packing,
    instance_features,
    is_feasible,
    oracle_configuration,
    solve_presolved,
    validate_solution,
)


def test_generation_is_reproducible() -> None:
    first = generate_packing(20, 7, seed=11)
    second = generate_packing(20, 7, seed=11)
    np.testing.assert_allclose(first.profits, second.profits)
    np.testing.assert_allclose(first.A, second.A)
    np.testing.assert_allclose(first.b, second.b)


def test_known_safe_fixings_and_solution_reconstruction() -> None:
    problem = BinaryPackingMIP(
        profits=np.array([-2.0, 8.0, 5.0, 4.0]),
        A=np.array([[1.0, 6.0, 0.0, 2.0], [0.0, 0.0, 0.0, 3.0]]),
        b=np.array([5.0, 4.0]),
    )
    reduced = apply_presolve(problem, "fixing")
    assert reduced.fixed_values == {0: 0, 1: 0, 2: 1}
    result = solve_presolved(reduced)
    validate_solution(problem, result)
    assert is_feasible(problem, result.x)
    np.testing.assert_allclose(result.x, [0.0, 0.0, 1.0, 1.0])
    assert np.isclose(result.objective, 9.0)


def test_componentwise_dominated_row_is_removed() -> None:
    problem = BinaryPackingMIP(
        profits=np.array([3.0, 4.0, 5.0]),
        A=np.array([[4.0, 3.0, 2.0], [2.0, 1.0, 1.0], [1.0, 4.0, 3.0]]),
        b=np.array([5.0, 7.0, 6.0]),
    )
    reduced = apply_presolve(problem, "rows")
    assert 1 in reduced.report.removed_rows
    assert reduced.b.size == 2


def test_all_safe_configurations_preserve_optimal_value_and_feasibility() -> None:
    for seed in range(12):
        problem = generate_packing(18, 6, seed=seed)
        baseline = solve_presolved(apply_presolve(problem, "none"))
        for config in CONFIGS:
            result = solve_presolved(apply_presolve(problem, config))
            validate_solution(problem, result)
            assert np.isclose(result.objective, baseline.objective, atol=1e-7)


def test_features_are_finite_and_have_expected_shape() -> None:
    features = instance_features(generate_packing(20, 7, seed=3))
    assert features.shape == (12,)
    assert np.all(np.isfinite(features))


def test_oracle_returns_objective_equivalent_configuration() -> None:
    problem = generate_packing(20, 7, seed=9)
    oracle = oracle_configuration(problem)
    assert oracle.selected in CONFIGS
    objectives = [item.objective for item in oracle.evaluations]
    assert max(objectives) - min(objectives) <= 1e-7
