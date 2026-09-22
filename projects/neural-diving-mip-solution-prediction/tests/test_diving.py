import numpy as np

from neural_diving.diving import neural_dive, partial_assignment, random_dive
from neural_diving.problem import generate_instance


def test_partial_assignment_fixes_high_confidence_variables():
    probs = np.array([0.51, 0.99, 0.02, 0.55, 0.80])
    fixed = partial_assignment(probs, fraction=0.4)
    assert set(fixed) == {1, 2}
    assert fixed[1] == 1 and fixed[2] == 0


def test_neural_dive_returns_feasible_solution_with_backoff():
    instance = generate_instance(21, n_items=16, n_constraints=4)
    # Deliberately overconfident all-ones predictions are usually infeasible at high fixation.
    probs = np.full(instance.n_items, 0.99)
    result = neural_dive(instance, probs, fractions=(0.9, 0.5, 0.2), stochastic_samples=0)
    assert result.solve.success
    assert result.solve.x is not None
    assert instance.feasible(result.solve.x)
    assert result.fraction_used <= 0.9


def test_random_dive_has_full_mip_fallback():
    instance = generate_instance(22, n_items=15, n_constraints=3)
    result = random_dive(instance, fraction=0.9, seed=1, samples=2)
    assert result.solve.success
    assert result.solve.x is not None
    assert instance.feasible(result.solve.x)
