import numpy as np

from learning_cp_sat.expert import collect_expert_samples
from learning_cp_sat.instances import generate_split
from learning_cp_sat.learning import (
    fit_random_forest,
    fit_ridge,
    samples_to_xy,
    top1_imitation_accuracy,
)
from learning_cp_sat.search import solve_coloring


def _samples():
    instances = generate_split(
        count=3,
        n_vertices=8,
        n_colors=3,
        edge_probability=0.45,
        base_seed=500,
        infeasible_fraction=1 / 3,
    )
    rows = []
    for i, instance in enumerate(instances):
        rows.extend(
            collect_expert_samples(
                instance,
                max_states=12,
                node_limit=100,
                state_prefix=f"train-{i}",
            )
        )
    return instances, rows


def test_learning_data_shapes_and_finite_targets() -> None:
    _, samples = _samples()
    x, y = samples_to_xy(samples)
    assert x.ndim == 2
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == 8
    assert np.isfinite(x).all()
    assert np.isfinite(y).all()


def test_learned_policy_remains_exact() -> None:
    instances, samples = _samples()
    model = fit_random_forest(samples, seed=19)
    result = solve_coloring(instances[0], policy="learned", model=model)
    reference = solve_coloring(instances[0], policy="mrv")
    assert (result.status == "FEASIBLE") == (reference.status == "FEASIBLE")


def test_imitation_accuracy_is_a_bounded_diagnostic() -> None:
    _, samples = _samples()
    ridge = fit_ridge(samples)
    accuracy = top1_imitation_accuracy(ridge, samples)
    assert 0.0 <= accuracy <= 1.0
