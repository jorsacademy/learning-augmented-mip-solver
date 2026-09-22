from learning_cp_sat.audit import audit_assignment
from learning_cp_sat.instances import GraphColoringInstance, generate_instance
from learning_cp_sat.oracle import brute_force_oracle


def test_oracle_finds_feasible_triangle_with_three_colors() -> None:
    instance = GraphColoringInstance(
        n_vertices=3,
        n_colors=3,
        edges=((0, 1), (0, 2), (1, 2)),
        seed=0,
        expected_feasible=True,
    )
    result = brute_force_oracle(instance)
    assert result.feasible
    assert result.assignment is not None
    assert audit_assignment(instance, result.assignment).feasible


def test_oracle_proves_kplus1_clique_infeasible() -> None:
    instance = generate_instance(
        n_vertices=6,
        n_colors=3,
        edge_probability=0.1,
        seed=7,
        infeasible=True,
    )
    result = brute_force_oracle(instance)
    assert not result.feasible
    assert result.assignment is None


def test_audit_catches_missing_and_violated_edges() -> None:
    instance = GraphColoringInstance(
        n_vertices=2,
        n_colors=2,
        edges=((0, 1),),
        seed=0,
        expected_feasible=True,
    )
    assert not audit_assignment(instance, [0]).feasible
    violation = audit_assignment(instance, [1, 1])
    assert not violation.feasible
    assert violation.violated_edges == ((0, 1),)
