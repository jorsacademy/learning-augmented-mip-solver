import torch

from lamip.training import collect_training_data, train_components


def test_oracle_data_and_training_smoke() -> None:
    data = collect_training_data(
        n_instances=8,
        n_vars=12,
        n_constraints=4,
        seed=100,
    )
    assert data.presolve_x.shape[1] == 8
    assert data.cut_x.shape[1] == 6
    assert data.branch_x.shape[1] == 6
    assert data.primal_x.shape[1] == 5
    assert torch.isfinite(data.cut_y).all()
    assert torch.isfinite(data.branch_y).all()

    components, losses = train_components(
        data,
        epochs=2,
        learning_rate=1e-3,
        hidden_dim=16,
        seed=0,
    )
    assert components.presolve is not None
    assert components.cuts is not None
    assert components.branching is not None
    assert components.primal is not None
    assert all(value >= 0.0 for value in losses.values())
