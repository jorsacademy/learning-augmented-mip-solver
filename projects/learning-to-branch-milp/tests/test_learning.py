from __future__ import annotations

import numpy as np

from ltb_milp.dataset import (
    collect_root_strong_branching_dataset,
    collect_tree_strong_branching_dataset,
)
from ltb_milp.models import BranchingMLP
from ltb_milp.training import (
    predict_scores,
    top1_expert_agreement,
    train_branching_model,
)


def _assert_valid_dataset(dataset) -> None:
    assert dataset.features.ndim == 2
    assert dataset.features.shape[1] == 6
    assert dataset.targets.shape == (dataset.features.shape[0],)
    assert sum(dataset.group_sizes) == dataset.features.shape[0]
    assert len(dataset.group_sizes) > 0
    assert all(size > 0 for size in dataset.group_sizes)
    assert np.all(np.isfinite(dataset.features))
    assert np.all(np.isfinite(dataset.targets))


def test_root_dataset_and_model_training_smoke() -> None:
    dataset = collect_root_strong_branching_dataset(20, 10, 4, seed=0)
    _assert_valid_dataset(dataset)

    model = BranchingMLP()
    history = train_branching_model(
        model,
        dataset,
        loss="listwise",
        epochs=5,
        learning_rate=1e-3,
    )
    scores = predict_scores(model, dataset.features[:4])
    agreement = top1_expert_agreement(model, dataset)
    assert len(history) == 5
    assert scores.shape == (4,)
    assert np.all(np.isfinite(scores))
    assert 0.0 <= agreement <= 1.0


def test_mse_training_remains_available_as_baseline() -> None:
    dataset = collect_root_strong_branching_dataset(10, 10, 4, seed=7)
    model = BranchingMLP()
    history = train_branching_model(model, dataset, loss="mse", epochs=3)
    assert len(history) == 3
    assert np.all(np.isfinite(history))


def test_tree_dataset_collects_fractional_search_nodes() -> None:
    dataset = collect_tree_strong_branching_dataset(
        8,
        10,
        4,
        seed=3,
        max_nodes_per_instance=24,
    )
    _assert_valid_dataset(dataset)
    assert dataset.features.shape[0] > 0


def test_tree_dataset_validates_node_budget() -> None:
    try:
        collect_tree_strong_branching_dataset(1, 8, 3, max_nodes_per_instance=0)
    except ValueError as exc:
        assert "max_nodes_per_instance" in str(exc)
    else:
        raise AssertionError("expected ValueError for zero node budget")
