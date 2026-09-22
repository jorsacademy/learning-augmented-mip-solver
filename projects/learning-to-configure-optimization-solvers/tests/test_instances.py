import numpy as np

from solverconfig.instances import generate_set_cover


def test_generation_is_deterministic_and_feasible() -> None:
    kwargs = dict(
        instance_id="x",
        regime="near_duplicate",
        n_rows=20,
        n_cols=30,
        density=0.08,
        seed=123,
    )
    first = generate_set_cover(**kwargs)
    second = generate_set_cover(**kwargs)
    assert np.array_equal(first.matrix, second.matrix)
    assert np.array_equal(first.costs, second.costs)
    assert np.all(first.matrix.sum(axis=1) >= 1)
