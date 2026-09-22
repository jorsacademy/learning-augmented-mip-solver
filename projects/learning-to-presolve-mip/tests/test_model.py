import numpy as np
import torch

from learning_to_presolve import (
    PresolveSelector,
    collect_dataset,
    evaluate_selector,
    generate_packing,
    instance_features,
    load_checkpoint,
    predict_configuration,
    save_checkpoint,
    train_selector,
)


def test_dataset_shapes_and_scores() -> None:
    dataset = collect_dataset(n_instances=8, n_vars=16, n_constraints=5, seed=100)
    assert dataset.features.shape == (8, 12)
    assert dataset.labels.shape == (8,)
    assert dataset.oracle_scores.shape == (8, 4)
    assert torch.all(torch.isfinite(dataset.oracle_scores))


def test_training_is_finite_and_selector_is_valid() -> None:
    torch.manual_seed(0)
    dataset = collect_dataset(n_instances=12, n_vars=16, n_constraints=5, seed=200)
    model = PresolveSelector(hidden_dim=16)
    initial, final = train_selector(model, dataset, epochs=4, learning_rate=1e-3)
    assert np.isfinite(initial)
    assert np.isfinite(final)
    metrics = evaluate_selector(model, dataset)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.mean_normalized_regret >= -1e-7
    prediction = predict_configuration(model, instance_features(generate_packing(16, 5, seed=999)))
    assert prediction in {"none", "fixing", "rows", "full"}


def test_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(0)
    dataset = collect_dataset(n_instances=8, n_vars=16, n_constraints=5, seed=300)
    model = PresolveSelector(hidden_dim=12)
    train_selector(model, dataset, epochs=2)
    path = tmp_path / "selector.pt"
    save_checkpoint(model, path)
    loaded = load_checkpoint(path)
    sample = dataset.features[0]
    with torch.no_grad():
        torch.testing.assert_close(model(sample), loaded(sample))
