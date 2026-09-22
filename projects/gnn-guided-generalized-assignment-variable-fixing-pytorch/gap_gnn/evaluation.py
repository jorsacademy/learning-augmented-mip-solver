from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .inference import (
    exact_repair_with_relaxation,
    oracle_fixing_for_same_jobs,
    propose_confident_fixings,
)
from .planning import GAPMILP, greedy_assignment


@dataclass(frozen=True)
class InstanceMetrics:
    accuracy: float
    mean_confidence: float
    initially_fixed: int
    finally_fixed: int
    wrong_fixes: int
    relaxed_fixes: int
    repair_gap_pct: float
    greedy_gap_pct: float
    vanilla_nodes: int
    repair_nodes: int
    oracle_fix_nodes: int
    vanilla_seconds: float
    repair_seconds: float
    oracle_fix_seconds: float


@dataclass(frozen=True)
class EvaluationResult:
    metrics: tuple

    def mean(self, attr: str) -> float:
        return float(np.mean([getattr(x, attr) for x in self.metrics]))


def evaluate_model_on_instances(
    model,
    instances,
    *,
    confidence_threshold: float = 0.92,
    margin_threshold: float = 0.15,
) -> EvaluationResult:
    rows = []
    for k, instance in enumerate(instances):
        vanilla = GAPMILP(instance).solve()
        if vanilla is None or vanilla.status != "OPTIMAL":
            raise RuntimeError("vanilla benchmark MILP failed")

        proposal = propose_confident_fixings(
            model,
            instance,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
        )
        accuracy = float(np.mean(proposal.predicted_assignment == vanilla.assignment))
        confidence = float(np.mean(np.max(proposal.probabilities, axis=0)))

        repaired = exact_repair_with_relaxation(
            instance,
            proposal,
            optimal_assignment=vanilla.assignment,
        )
        oracle = oracle_fixing_for_same_jobs(
            instance,
            proposal,
            vanilla.assignment,
        )
        greedy_assignment_vector, greedy_obj = greedy_assignment(
            instance,
            restarts=48,
            seed=9000 + k,
        )
        greedy_gap = (
            100.0 * (greedy_obj - vanilla.objective) / max(abs(vanilla.objective), 1e-9)
            if greedy_assignment_vector is not None
            else float("inf")
        )
        repair_gap = 100.0 * (
            repaired.solution.objective - vanilla.objective
        ) / max(abs(vanilla.objective), 1e-9)

        rows.append(InstanceMetrics(
            accuracy=accuracy,
            mean_confidence=confidence,
            initially_fixed=repaired.initially_fixed_jobs,
            finally_fixed=repaired.finally_fixed_jobs,
            wrong_fixes=repaired.wrong_initial_fixes,
            relaxed_fixes=repaired.relaxed_jobs,
            repair_gap_pct=float(repair_gap),
            greedy_gap_pct=float(greedy_gap),
            vanilla_nodes=vanilla.mip_node_count,
            repair_nodes=repaired.solution.mip_node_count,
            oracle_fix_nodes=oracle.mip_node_count,
            vanilla_seconds=vanilla.solve_seconds,
            repair_seconds=repaired.solution.solve_seconds,
            oracle_fix_seconds=oracle.solve_seconds,
        ))

    return EvaluationResult(tuple(rows))
