"""Branch-and-bound solver used to benchmark branching policies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ltb_milp.branching import (
    BranchDecision,
    PseudocostState,
    most_fractional,
    pseudocost_branch,
    reliability_branch,
    strong_branch,
)
from ltb_milp.problem import BinaryPackingMILP, solve_lp_relaxation

BranchPolicy = Callable[
    [BinaryPackingMILP, np.ndarray, float, dict[int, tuple[float, float]]], BranchDecision
]


@dataclass(frozen=True)
class SolveResult:
    objective: float
    solution: np.ndarray | None
    nodes_processed: int
    lp_solves: int
    optimal: bool


@dataclass(frozen=True)
class _Node:
    bounds: dict[int, tuple[float, float]]
    parent_objective: float | None = None
    branch_variable: int | None = None
    branch_direction: str | None = None
    branch_distance: float | None = None


def _most_fractional_policy(
    problem: BinaryPackingMILP,
    x: np.ndarray,
    objective: float,
    bounds: dict[int, tuple[float, float]],
) -> BranchDecision:
    del problem, objective, bounds
    return most_fractional(x)


def _strong_policy(
    problem: BinaryPackingMILP,
    x: np.ndarray,
    objective: float,
    bounds: dict[int, tuple[float, float]],
) -> BranchDecision:
    return strong_branch(problem, x, objective, bounds)


def solve_branch_and_bound(
    problem: BinaryPackingMILP,
    *,
    policy: BranchPolicy | str = "most_fractional",
    max_nodes: int = 10_000,
    integrality_tol: float = 1e-7,
    reliability_threshold: int = 2,
) -> SolveResult:
    """Solve a small binary MILP with depth-first branch-and-bound."""
    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive")
    if reliability_threshold <= 0:
        raise ValueError("reliability_threshold must be positive")

    pseudocost_state: PseudocostState | None = None
    stateful_policy: str | None = None
    if isinstance(policy, str):
        if policy == "most_fractional":
            branch_policy = _most_fractional_policy
        elif policy == "strong":
            branch_policy = _strong_policy
        elif policy in {"pseudocost", "reliability"}:
            pseudocost_state = PseudocostState.zeros(problem.n_vars)
            stateful_policy = policy
            branch_policy = None
        else:
            raise ValueError("unknown branching policy")
    else:
        branch_policy = policy

    incumbent = float("-inf")
    incumbent_x: np.ndarray | None = None
    stack: list[_Node] = [_Node({})]
    nodes = 0
    lp_solves = 0

    while stack and nodes < max_nodes:
        node = stack.pop()
        lp = solve_lp_relaxation(problem, node.bounds)
        nodes += 1
        lp_solves += 1

        if (
            pseudocost_state is not None
            and node.parent_objective is not None
            and node.branch_variable is not None
            and node.branch_direction is not None
            and node.branch_distance is not None
        ):
            gain = (
                abs(node.parent_objective) + 1.0
                if not lp.feasible
                else max(0.0, node.parent_objective - lp.objective)
            )
            pseudocost_state.update(
                node.branch_variable,
                node.branch_direction,
                gain,
                node.branch_distance,
            )

        if not lp.feasible or lp.objective <= incumbent + 1e-9:
            continue
        assert lp.x is not None
        rounded = np.rint(lp.x)
        if np.all(np.abs(lp.x - rounded) <= integrality_tol):
            value = float(problem.c @ rounded)
            if value > incumbent:
                incumbent = value
                incumbent_x = rounded.astype(float)
            continue

        if pseudocost_state is not None:
            if stateful_policy == "reliability":
                decision = reliability_branch(
                    problem,
                    lp.x,
                    lp.objective,
                    node.bounds,
                    pseudocost_state,
                    reliability_threshold=reliability_threshold,
                )
            else:
                decision = pseudocost_branch(lp.x, pseudocost_state)
        else:
            assert branch_policy is not None
            decision = branch_policy(problem, lp.x, lp.objective, node.bounds)
        lp_solves += decision.extra_lp_solves
        variable = decision.variable
        value = float(lp.x[variable])
        down_bounds = dict(node.bounds)
        up_bounds = dict(node.bounds)
        down_bounds[variable] = (0.0, 0.0)
        up_bounds[variable] = (1.0, 1.0)
        down = _Node(
            down_bounds,
            parent_objective=lp.objective,
            branch_variable=variable,
            branch_direction="down",
            branch_distance=value,
        )
        up = _Node(
            up_bounds,
            parent_objective=lp.objective,
            branch_variable=variable,
            branch_direction="up",
            branch_distance=1.0 - value,
        )
        if value >= 0.5:
            stack.extend([down, up])
        else:
            stack.extend([up, down])

    return SolveResult(
        objective=incumbent,
        solution=incumbent_x,
        nodes_processed=nodes,
        lp_solves=lp_solves,
        optimal=not stack,
    )
