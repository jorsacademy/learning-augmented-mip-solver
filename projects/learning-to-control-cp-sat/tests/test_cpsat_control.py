import pytest

pytest.importorskip("ortools")

from learning_cp_sat.cpsat_control import solve_with_cpsat
from learning_cp_sat.expert import collect_expert_samples
from learning_cp_sat.instances import generate_instance
from learning_cp_sat.learning import fit_ridge


def test_cpsat_default_and_fixed_search_agree_on_feasibility() -> None:
    instance = generate_instance(
        n_vertices=9,
        n_colors=3,
        edge_probability=0.45,
        seed=21,
        infeasible=True,
    )
    default = solve_with_cpsat(instance, strategy="default", seed=1, time_limit_seconds=5)
    fixed = solve_with_cpsat(instance, strategy="degree", seed=1, time_limit_seconds=5)
    min_domain = solve_with_cpsat(instance, strategy="min_domain", seed=1, time_limit_seconds=5)
    assert default.status == "INFEASIBLE"
    assert fixed.status == "INFEASIBLE"
    assert min_domain.status == "INFEASIBLE"


def test_cpsat_learned_root_uses_public_fixed_strategy() -> None:
    train = generate_instance(
        n_vertices=8,
        n_colors=3,
        edge_probability=0.4,
        seed=31,
        infeasible=False,
    )
    samples = collect_expert_samples(train, max_states=10, node_limit=100, state_prefix="cp")
    model = fit_ridge(samples)
    result = solve_with_cpsat(
        train,
        strategy="learned_root",
        model=model,
        seed=2,
        time_limit_seconds=5,
    )
    assert result.status in {"OPTIMAL", "FEASIBLE"}
    assert result.assignment is not None
