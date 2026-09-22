import torch

from learning_to_presolve import (
    PresolveSelector,
    benchmark_selector,
    collect_dataset,
    generate_packing,
    heuristic_configuration,
    instance_features,
    train_selector,
)


def test_heuristic_returns_valid_configuration() -> None:
    features = instance_features(generate_packing(16, 5, seed=4))
    assert heuristic_configuration(features) in {"none", "fixing", "rows", "full"}


def test_benchmark_contains_all_baselines() -> None:
    torch.manual_seed(0)
    training = collect_dataset(n_instances=8, n_vars=16, n_constraints=5, seed=500)
    model = PresolveSelector(hidden_dim=12)
    train_selector(model, training, epochs=2)
    result = benchmark_selector(model, n_instances=4, n_vars=16, n_constraints=5, seed_start=900)
    policies = {summary.policy for summary in result.summaries}
    assert policies == {"none", "fixing", "rows", "full", "heuristic", "learned", "oracle"}
    assert result.by_policy("oracle").mean_normalized_regret == 0.0
    assert abs(result.by_policy("oracle").oracle_match_rate - 1.0) < 1e-12
    assert sum(result.label_counts.values()) == 4
