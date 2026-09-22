import numpy as np
import torch

from ltb_milp.gnn import (
    BipartiteBranchingGNN,
    gnn_branch_policy,
    gnn_top1_expert_agreement,
    train_gnn_branching_model,
)
from ltb_milp.graph import graph_inputs_from_node, graph_state_from_node
from ltb_milp.graph_dataset import collect_tree_graph_dataset
from ltb_milp.problem import generate_binary_packing, solve_lp_relaxation
from ltb_milp.solver import solve_branch_and_bound


def _fractional_problem(seed: int = 0):
    for offset in range(100):
        problem = generate_binary_packing(10, 4, seed=seed + offset)
        lp = solve_lp_relaxation(problem)
        if lp.feasible and lp.x is not None:
            fractional = (lp.x > 1e-7) & (lp.x < 1.0 - 1e-7)
            if np.any(fractional):
                return problem, lp
    raise RuntimeError("failed to generate fractional root node")


def test_graph_inputs_and_gnn_candidate_scores() -> None:
    problem, lp = _fractional_problem(3)
    inputs = graph_inputs_from_node(problem, lp.x)
    model = BipartiteBranchingGNN(hidden_dim=16)
    scores = model(inputs)

    assert inputs.variable_features.shape == (problem.n_vars, 4)
    assert inputs.constraint_features.shape == (problem.n_constraints, 4)
    assert inputs.edge_weights.shape == problem.A.shape
    assert scores.shape == (len(inputs.candidate_indices),)
    assert torch.isfinite(scores).all()


def test_graph_state_labels_expert_candidate() -> None:
    problem, lp = _fractional_problem(11)
    state = graph_state_from_node(problem, lp.x, lp.objective)
    assert 0 <= state.expert_choice < len(state.inputs.candidate_indices)


def test_gnn_training_and_branch_policy_smoke() -> None:
    states = collect_tree_graph_dataset(5, 9, 4, seed=20, max_nodes_per_instance=10)
    model = BipartiteBranchingGNN(hidden_dim=16)
    history = train_gnn_branching_model(model, states, epochs=2, learning_rate=1e-3)
    agreement = gnn_top1_expert_agreement(model, states)

    assert len(history) == 2
    assert 0.0 <= agreement <= 1.0

    problem = generate_binary_packing(9, 4, seed=500)
    baseline = solve_branch_and_bound(problem, policy="most_fractional")
    learned = solve_branch_and_bound(problem, policy=gnn_branch_policy(model))
    assert baseline.optimal and learned.optimal
    assert abs(baseline.objective - learned.objective) <= 1e-6
