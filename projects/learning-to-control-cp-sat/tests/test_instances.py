from learning_cp_sat.instances import generate_instance, generate_split


def test_generation_is_deterministic_and_certificates_match() -> None:
    a = generate_instance(n_vertices=8, n_colors=3, edge_probability=0.5, seed=17, infeasible=False)
    b = generate_instance(n_vertices=8, n_colors=3, edge_probability=0.5, seed=17, infeasible=False)
    assert a == b
    assert a.expected_feasible

    bad = generate_instance(
        n_vertices=8, n_colors=3, edge_probability=0.2, seed=17, infeasible=True
    )
    clique = {(u, v) for u in range(4) for v in range(u + 1, 4)}
    assert clique.issubset(set(bad.edges))
    assert not bad.expected_feasible


def test_split_seed_reproducibility() -> None:
    kwargs = dict(
        count=6,
        n_vertices=9,
        n_colors=3,
        edge_probability=0.4,
        base_seed=99,
        infeasible_fraction=0.5,
    )
    assert generate_split(**kwargs) == generate_split(**kwargs)
