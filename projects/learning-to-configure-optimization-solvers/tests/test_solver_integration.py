import math

from solverconfig.configurations import by_name
from solverconfig.instances import generate_set_cover
from solverconfig.oracle import exhaustive_set_cover
from solverconfig.solver import solve_with_highs


def test_highs_optimum_matches_independent_oracle() -> None:
    inst = generate_set_cover(
        instance_id="oracle-check",
        regime="uniform",
        n_rows=7,
        n_cols=10,
        density=0.35,
        seed=42,
    )
    exact = exhaustive_set_cover(inst)
    result = solve_with_highs(inst, by_name("default"), time_limit=5.0, random_seed=7)
    assert result.optimal
    assert result.feasible
    assert result.objective_recomputed is not None
    assert math.isclose(result.objective_recomputed, exact.objective, rel_tol=1e-8, abs_tol=1e-8)
    assert math.isclose(result.objective or 0.0, exact.objective, rel_tol=1e-8, abs_tol=1e-8)
