from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .bnb import BranchAndBoundSolver
from .policy import ImitationPolicy
from .problem import brute_force_optimum, generate_knapsack


@dataclass(frozen=True)
class MethodResult:
    method: str
    objective: float
    optimum: float
    relative_gap: float
    nodes_popped: int
    nodes_expanded: int
    learned_prunes: int
    elapsed_seconds: float
    optimality_guaranteed: bool


def _gap(optimum: float, objective: float) -> float:
    return float(max(optimum - objective, 0.0) / max(abs(optimum), 1e-12))


def benchmark_policy(
    policy: ImitationPolicy,
    *,
    seed: int = 7_000,
    instances: int = 30,
    n_items: int = 14,
    prune_threshold: float = 0.15,
) -> dict[str, object]:
    rows: list[MethodResult] = []
    for i in range(instances):
        instance = generate_knapsack(seed + i, n_items=n_items)
        optimum, _ = brute_force_optimum(instance)
        configurations = [
            ("best_bound_exact", BranchAndBoundSolver(selection_mode="best_bound")),
            ("dfs_exact", BranchAndBoundSolver(selection_mode="dfs")),
            (
                "learned_selection_exact",
                BranchAndBoundSolver(selection_mode="learned", policy=policy),
            ),
            (
                "learned_selection_plus_pruning_approx",
                BranchAndBoundSolver(
                    selection_mode="learned",
                    policy=policy,
                    allow_learned_pruning=True,
                    prune_threshold=prune_threshold,
                ),
            ),
        ]
        for name, solver in configurations:
            result = solver.solve(instance)
            gap = _gap(optimum, result.objective)
            if result.optimality_guaranteed and gap > 1e-9:
                raise AssertionError(f"exact method {name} failed on seed {seed + i}: gap={gap}")
            rows.append(
                MethodResult(
                    method=name,
                    objective=result.objective,
                    optimum=optimum,
                    relative_gap=gap,
                    nodes_popped=result.nodes_popped,
                    nodes_expanded=result.nodes_expanded,
                    learned_prunes=result.learned_prunes,
                    elapsed_seconds=result.elapsed_seconds,
                    optimality_guaranteed=result.optimality_guaranteed,
                )
            )

    methods = sorted({row.method for row in rows})
    summary: dict[str, dict[str, float | bool]] = {}
    for method in methods:
        selected = [row for row in rows if row.method == method]
        summary[method] = {
            "mean_relative_gap": float(np.mean([row.relative_gap for row in selected])),
            "max_relative_gap": float(np.max([row.relative_gap for row in selected])),
            "mean_nodes_popped": float(np.mean([row.nodes_popped for row in selected])),
            "mean_nodes_expanded": float(np.mean([row.nodes_expanded for row in selected])),
            "mean_elapsed_seconds": float(np.mean([row.elapsed_seconds for row in selected])),
            "optimality_guaranteed": bool(selected[0].optimality_guaranteed),
        }
    return {
        "seed": seed,
        "instances": instances,
        "n_items": n_items,
        "prune_threshold": prune_threshold,
        "summary": summary,
        "rows": [asdict(row) for row in rows],
    }
