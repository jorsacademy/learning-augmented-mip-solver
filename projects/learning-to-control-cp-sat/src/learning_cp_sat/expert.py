from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import candidate_features
from .instances import GraphColoringInstance
from .propagation import assign_and_propagate, colors_in, domain_size


@dataclass(frozen=True)
class ExpertCandidateSample:
    state_id: str
    variable: int
    features: np.ndarray
    target_score: float
    is_best: bool


def strong_lookahead_scores(
    instance: GraphColoringInstance,
    domains: list[int],
) -> tuple[dict[int, float], int]:
    """One-step CP analogue of strong branching for variable ordering.

    Every value of every candidate variable is tentatively assigned, then domain
    propagation is run. Variables that cause more immediate failures and stronger
    domain reduction receive larger scores.
    """
    scores: dict[int, float] = {}
    probes = 0
    scale = max(1, instance.n_vertices * instance.n_colors)

    for variable, mask in enumerate(domains):
        if domain_size(mask) <= 1:
            continue
        failures = 0
        reductions = 0
        values = colors_in(mask)
        for color in values:
            probes += 1
            propagated, removed, _ = assign_and_propagate(instance, domains, variable, color)
            if propagated is None:
                failures += 1
            else:
                reductions += removed
        failure_ratio = failures / len(values)
        mean_reduction = reductions / max(1, len(values) - failures)
        scores[variable] = 2.0 * failure_ratio + mean_reduction / scale
    return scores, probes


def choose_expert_variable(
    instance: GraphColoringInstance,
    domains: list[int],
) -> tuple[int, dict[int, float], int]:
    scores, probes = strong_lookahead_scores(instance, domains)
    if not scores:
        raise ValueError("no branching candidate")
    degrees = instance.degrees
    best = max(scores, key=lambda v: (scores[v], degrees[v], -v))
    return best, scores, probes


def collect_expert_samples(
    instance: GraphColoringInstance,
    *,
    max_states: int,
    node_limit: int,
    state_prefix: str,
) -> list[ExpertCandidateSample]:
    """Collect candidate-level imitation data along an expert DFS trajectory."""
    from .propagation import initial_domains

    samples: list[ExpertCandidateSample] = []
    domains = initial_domains(instance)
    states_seen = 0
    nodes = 0
    stop = False

    def dfs(current: list[int], depth: int) -> bool:
        nonlocal states_seen, nodes, stop
        if stop:
            return False
        if all(domain_size(mask) == 1 for mask in current):
            return True
        if nodes >= node_limit or states_seen >= max_states:
            stop = True
            return False

        variable, scores, _ = choose_expert_variable(instance, current)
        state_id = f"{state_prefix}:{states_seen}"
        states_seen += 1
        for candidate, score in scores.items():
            samples.append(
                ExpertCandidateSample(
                    state_id=state_id,
                    variable=candidate,
                    features=candidate_features(instance, current, candidate, depth),
                    target_score=score,
                    is_best=candidate == variable,
                )
            )

        nodes += 1
        for color in colors_in(current[variable]):
            child, _, _ = assign_and_propagate(instance, current, variable, color)
            if child is not None and dfs(child, depth + 1):
                return True
            if stop:
                return False
        return False

    dfs(domains, 0)
    return samples
