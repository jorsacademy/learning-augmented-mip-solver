import numpy as np

from solverconfig.selector import PortfolioCostSelector


def test_selector_fits_cost_surface_without_test_labels() -> None:
    x = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    costs = np.column_stack([x[:, 0], 5.0 - x[:, 0]])
    selector = PortfolioCostSelector(
        ("a", "b"), n_estimators=50, min_samples_leaf=1, random_state=3
    )
    selector.fit(x, costs)
    indices, elapsed = selector.predict_indices(np.array([[0.1], [4.9]]))
    assert indices.tolist() == [0, 1]
    assert elapsed >= 0
