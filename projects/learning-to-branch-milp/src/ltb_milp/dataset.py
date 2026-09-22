"""Imitation dataset generation from strong-branching scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltb_milp.branching import strong_branch_scores
from ltb_milp.features import candidate_features
from ltb_milp.problem import BinaryPackingMILP, generate_binary_packing, solve_lp_relaxation


@dataclass(frozen=True)
class BranchingDataset:
    features: np.ndarray
    targets: np.ndarray
    group_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("features must be two-dimensional")
        if self.targets.shape != (self.features.shape[0],):
            raise ValueError("targets must contain one value per feature row")
        if not self.group_sizes or any(size <= 0 for size in self.group_sizes):
            raise ValueError("group_sizes must contain positive candidate counts")
        if sum(self.group_sizes) != self.features.shape[0]:
            raise ValueError("group_sizes must partition all candidate rows")


def _append_node_examples(
    problem: BinaryPackingMILP,
    x: np.ndarray,
    objective: float,
    bounds: dict[int, tuple[float, float]],
    feature_rows: list[np.ndarray],
    targets: list[np.ndarray],
    group_sizes: list[int],
) -> int:
    candidates, features = candidate_features(problem, x)
    if candidates.size == 0:
        return -1
    score_candidates, scores, _ = strong_branch_scores(problem, x, objective, bounds)
    if not np.array_equal(candidates, score_candidates):
        raise RuntimeError("candidate ordering mismatch")
    scale = max(float(np.max(np.abs(scores))), 1e-8)
    feature_rows.append(features)
    targets.append(scores / scale)
    group_sizes.append(int(candidates.size))
    return int(candidates[np.argmax(scores)])


def _finalize_dataset(
    feature_rows: list[np.ndarray],
    targets: list[np.ndarray],
    group_sizes: list[int],
    *,
    context: str,
) -> BranchingDataset:
    if not feature_rows:
        raise RuntimeError(f"no fractional {context} nodes were generated")
    return BranchingDataset(
        np.vstack(feature_rows),
        np.concatenate(targets),
        tuple(group_sizes),
    )


def collect_root_strong_branching_dataset(
    n_instances: int,
    n_vars: int,
    n_constraints: int,
    *,
    seed: int = 0,
) -> BranchingDataset:
    """Collect candidate features and normalized strong-branching scores at root nodes."""
    if n_instances <= 0:
        raise ValueError("n_instances must be positive")

    feature_rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    group_sizes: list[int] = []
    for offset in range(n_instances):
        problem = generate_binary_packing(n_vars, n_constraints, seed=seed + offset)
        lp = solve_lp_relaxation(problem)
        if not lp.feasible or lp.x is None:
            continue
        _append_node_examples(
            problem,
            lp.x,
            lp.objective,
            {},
            feature_rows,
            targets,
            group_sizes,
        )

    return _finalize_dataset(feature_rows, targets, group_sizes, context="root")


def collect_tree_strong_branching_dataset(
    n_instances: int,
    n_vars: int,
    n_constraints: int,
    *,
    seed: int = 0,
    max_nodes_per_instance: int = 64,
    integrality_tol: float = 1e-7,
) -> BranchingDataset:
    """Collect expert examples from multiple nodes of strong-branching search trees.

    Each training instance is explored depth-first using strong branching as the
    expert policy. Fractional nodes contribute one row per branching candidate.
    Incumbent-based LP-bound pruning mirrors the benchmark branch-and-bound solver.
    """
    if n_instances <= 0:
        raise ValueError("n_instances must be positive")
    if max_nodes_per_instance <= 0:
        raise ValueError("max_nodes_per_instance must be positive")

    feature_rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    group_sizes: list[int] = []

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

            variable = _append_node_examples(
                problem,
                lp.x,
                lp.objective,
                bounds,
                feature_rows,
                targets,
                group_sizes,
            )
            if variable < 0:
                continue

            down = dict(bounds)
            up = dict(bounds)
            down[variable] = (0.0, 0.0)
            up[variable] = (1.0, 1.0)
            if lp.x[variable] >= 0.5:
                stack.extend([down, up])
            else:
                stack.extend([up, down])

    return _finalize_dataset(feature_rows, targets, group_sizes, context="tree")
