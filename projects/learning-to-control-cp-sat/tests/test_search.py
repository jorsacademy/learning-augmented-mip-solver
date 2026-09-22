import pytest

from learning_cp_sat.audit import audit_assignment
from learning_cp_sat.instances import generate_instance
from learning_cp_sat.oracle import brute_force_oracle
from learning_cp_sat.search import solve_coloring


@pytest.mark.parametrize("policy", ["fixed", "degree", "mrv", "dom_degree", "random", "expert"])
def test_exact_search_matches_independent_oracle(policy: str) -> None:
    instance = generate_instance(
        n_vertices=7,
        n_colors=3,
        edge_probability=0.45,
        seed=121,
        infeasible=True,
    )
    oracle = brute_force_oracle(instance)
    result = solve_coloring(instance, policy=policy, seed=13)
    assert (result.status == "FEASIBLE") == oracle.feasible
    assert result.status != "UNKNOWN"


def test_feasible_solution_passes_independent_audit() -> None:
    instance = generate_instance(
        n_vertices=10,
        n_colors=3,
        edge_probability=0.55,
        seed=77,
        infeasible=False,
    )
    result = solve_coloring(instance, policy="mrv")
    assert result.status == "FEASIBLE"
    assert result.assignment is not None
    assert audit_assignment(instance, result.assignment).feasible


def test_node_limit_reports_unknown_not_infeasible() -> None:
    instance = generate_instance(
        n_vertices=12,
        n_colors=3,
        edge_probability=0.5,
        seed=88,
        infeasible=True,
    )
    result = solve_coloring(instance, policy="fixed", node_limit=0)
    assert result.status == "UNKNOWN"


def test_random_policy_is_seed_deterministic() -> None:
    instance = generate_instance(
        n_vertices=11,
        n_colors=3,
        edge_probability=0.4,
        seed=123,
        infeasible=False,
    )
    a = solve_coloring(instance, policy="random", seed=42)
    b = solve_coloring(instance, policy="random", seed=42)
    assert (a.status, a.assignment, a.nodes, a.backtracks) == (
        b.status,
        b.assignment,
        b.nodes,
        b.backtracks,
    )
