"""Graph-state dataset generation from strong-branching search trees."""

from __future__ import annotations

import numpy as np

from ltb_milp.graph import GraphBranchingState, graph_state_from_node
from ltb_milp.problem import generate_binary_packing, solve_lp_relaxation


def collect_tree_graph_dataset(
    n_instances: int,
    n_vars: int,
    n_constraints: int,
    *,
    seed: int = 0,
    max_nodes_per_instance: int = 32,
    integrality_tol: float = 1e-7,
) -> list[GraphBranchingState]:
    """Collect labeled graph states along strong-branching expert trajectories."""
    if n_instances <= 0:
        raise ValueError("n_instances must be positive")
    if max_nodes_per_instance <= 0:
        raise ValueError("max_nodes_per_instance must be positive")

    states: list[GraphBranchingState] = []
    for offset in range(n_instances):
        problem = generate_binary_packing(n_vars, n_constraints, seed=seed + offset)
        incumbent = float("-inf")
        stack: list[dict[int, tuple[float, float]]] = [{}]
        nodes = 0

        while stack and nodes < max_nodes_per_instance:
            bounds = stack.pop()
            lp = solve_lp_relaxation(problem, bounds)
            nodes += 1
            if not lp.feasible or lp.x is None or lp.objective <= incumbent + 1e-9:
                continue

            rounded = np.rint(lp.x)
            if np.all(np.abs(lp.x - rounded) <= integrality_tol):
                incumbent = max(incumbent, float(problem.c @ rounded))
                continue

            state = graph_state_from_node(problem, lp.x, lp.objective, bounds)
            states.append(state)
            variable = int(state.inputs.candidate_indices[state.expert_choice])

            down = dict(bounds)
            up = dict(bounds)
            down[variable] = (0.0, 0.0)
            up[variable] = (1.0, 1.0)
            if lp.x[variable] >= 0.5:
                stack.extend([down, up])
            else:
                stack.extend([up, down])

    if not states:
        raise RuntimeError("no fractional graph states were generated")
    return states
