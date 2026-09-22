from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import (
    CONFIGS,
    ConfigEvaluation,
    PresolveConfig,
    generate_packing,
    instance_features,
    oracle_configuration,
)
from .model import PresolveSelector, predict_configuration

Policy = PresolveConfig | str


@dataclass(frozen=True)
class PolicySummary:
    policy: str
    mean_work_proxy: float
    mean_nodes: float
    mean_residual_size: float
    mean_residual_nnz: float
    mean_checks: float
    mean_normalized_regret: float
    oracle_match_rate: float


@dataclass(frozen=True)
class BenchmarkResult:
    summaries: tuple[PolicySummary, ...]
    label_counts: dict[str, int]

    def by_policy(self, policy: str) -> PolicySummary:
        for summary in self.summaries:
            if summary.policy == policy:
                return summary
        raise KeyError(policy)


def heuristic_configuration(features: np.ndarray) -> PresolveConfig:
    """Transparent rule-based baseline using only pre-presolve structural features."""

    nonpositive = float(features[5])
    infeasible = float(features[6])
    zero_columns = float(features[7])
    similarity = float(features[9])
    fixing_signal = nonpositive + infeasible + zero_columns
    row_signal = similarity >= 0.995
    if fixing_signal > 0.02 and row_signal:
        return "full"
    if fixing_signal > 0.02:
        return "fixing"
    if row_signal:
        return "rows"
    return "none"


def _evaluation_for_config(
    evaluations: tuple[ConfigEvaluation, ...], config: PresolveConfig
) -> ConfigEvaluation:
    return evaluations[CONFIGS.index(config)]


def benchmark_selector(
    model: PresolveSelector,
    n_instances: int = 80,
    n_vars: int = 32,
    n_constraints: int = 10,
    seed_start: int = 200_000,
) -> BenchmarkResult:
    if n_instances < 1:
        raise ValueError("n_instances must be positive")
    policies = [*CONFIGS, "heuristic", "learned", "oracle"]
    values: dict[str, list[tuple[ConfigEvaluation, float, bool]]] = {policy: [] for policy in policies}
    label_counts = {config: 0 for config in CONFIGS}

    for offset in range(n_instances):
        problem = generate_packing(n_vars=n_vars, n_constraints=n_constraints, seed=seed_start + offset)
        features = instance_features(problem)
        oracle = oracle_configuration(problem)
        label_counts[oracle.selected] += 1
        oracle_eval = _evaluation_for_config(oracle.evaluations, oracle.selected)

        choices: dict[str, PresolveConfig] = {config: config for config in CONFIGS}
        choices["heuristic"] = heuristic_configuration(features)
        choices["learned"] = predict_configuration(model, features)
        choices["oracle"] = oracle.selected

        for policy, config in choices.items():
            evaluation = _evaluation_for_config(oracle.evaluations, config)
            regret = (evaluation.work_proxy - oracle_eval.work_proxy) / max(
                abs(oracle_eval.work_proxy), 1.0
            )
            values[policy].append((evaluation, regret, config == oracle.selected))

    summaries: list[PolicySummary] = []
    for policy in policies:
        rows = values[policy]
        evaluations = [row[0] for row in rows]
        summaries.append(
            PolicySummary(
                policy=policy,
                mean_work_proxy=float(np.mean([item.work_proxy for item in evaluations])),
                mean_nodes=float(np.mean([item.nodes for item in evaluations])),
                mean_residual_size=float(
                    np.mean([item.residual_vars + item.residual_rows for item in evaluations])
                ),
                mean_residual_nnz=float(np.mean([item.residual_nnz for item in evaluations])),
                mean_checks=float(np.mean([item.checks for item in evaluations])),
                mean_normalized_regret=float(np.mean([row[1] for row in rows])),
                oracle_match_rate=float(np.mean([row[2] for row in rows])),
            )
        )
    return BenchmarkResult(summaries=tuple(summaries), label_counts=label_counts)
