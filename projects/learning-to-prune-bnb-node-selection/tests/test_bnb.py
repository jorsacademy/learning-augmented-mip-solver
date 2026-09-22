import numpy as np

from ltp_bnb.bnb import BranchAndBoundSolver
from ltp_bnb.policy import BinaryLinearModel, ImitationPolicy
from ltp_bnb.problem import brute_force_optimum, generate_knapsack


def _flat_policy(n_features: int = 10) -> ImitationPolicy:
    model = BinaryLinearModel(
        weights=[0.0] * n_features,
        bias=0.0,
        mean=[0.0] * n_features,
        scale=[1.0] * n_features,
    )
    return ImitationPolicy(selection=model, pruning=model)


def test_exact_selection_modes_match_bruteforce() -> None:
    policy = _flat_policy()
    for seed in range(8):
        instance = generate_knapsack(seed, n_items=11)
        optimum, _ = brute_force_optimum(instance)
        for solver in (
            BranchAndBoundSolver(selection_mode="best_bound"),
            BranchAndBoundSolver(selection_mode="dfs"),
            BranchAndBoundSolver(selection_mode="learned", policy=policy),
        ):
            result = solver.solve(instance)
            assert np.isclose(result.objective, optimum)
            assert result.optimality_guaranteed


def test_learned_pruning_is_explicitly_marked_approximate() -> None:
    instance = generate_knapsack(99, n_items=11)
    policy = _flat_policy()
    result = BranchAndBoundSolver(
        selection_mode="learned",
        policy=policy,
        allow_learned_pruning=True,
        prune_threshold=0.9,
    ).solve(instance)
    assert not result.optimality_guaranteed
    assert result.learned_prunes >= 0
