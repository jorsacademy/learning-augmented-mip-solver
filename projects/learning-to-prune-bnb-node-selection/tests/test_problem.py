import numpy as np

from ltp_bnb.problem import KnapsackInstance, brute_force_optimum, generate_knapsack


def test_generator_is_deterministic_and_sorted() -> None:
    a = generate_knapsack(123, n_items=10)
    b = generate_knapsack(123, n_items=10)
    assert np.allclose(a.weights, b.weights)
    assert np.allclose(a.values, b.values)
    ratios = a.values / a.weights
    assert np.all(ratios[:-1] >= ratios[1:] - 1e-12)


def test_fractional_bound_dominates_integer_optimum() -> None:
    instance = generate_knapsack(5, n_items=12)
    optimum, _ = brute_force_optimum(instance)
    assert instance.fractional_bound(0, 0.0, 0.0) + 1e-9 >= optimum


def test_invalid_unsorted_instance_is_rejected() -> None:
    try:
        KnapsackInstance(
            weights=np.array([10.0, 1.0]), values=np.array([1.0, 10.0]), capacity=10
        )
    except ValueError as exc:
        assert "sorted" in str(exc)
    else:
        raise AssertionError("expected ValueError")
