from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

import numpy as np

from .features import node_features
from .policy import ImitationPolicy
from .problem import KnapsackInstance

SelectionMode = Literal["best_bound", "dfs", "learned"]


@dataclass
class Node:
    level: int
    weight: float
    value: float
    bound: float
    decisions: tuple[int, ...]
    node_id: int


@dataclass
class TraceRecord:
    features: np.ndarray
    oracle_expand: int


@dataclass
class BnBResult:
    objective: float
    decisions: tuple[int, ...]
    nodes_popped: int
    nodes_expanded: int
    safe_prunes: int
    learned_prunes: int
    elapsed_seconds: float
    optimality_guaranteed: bool
    trace: list[TraceRecord] = field(default_factory=list)


class BranchAndBoundSolver:
    """Small exact B&B core with an explicitly unsafe optional learned-pruning mode."""

    def __init__(
        self,
        *,
        selection_mode: SelectionMode = "best_bound",
        policy: ImitationPolicy | None = None,
        allow_learned_pruning: bool = False,
        prune_threshold: float = 0.15,
    ) -> None:
        if selection_mode == "learned" and policy is None:
            raise ValueError("learned selection requires a policy")
        if allow_learned_pruning and policy is None:
            raise ValueError("learned pruning requires a policy")
        if not 0.0 <= prune_threshold <= 1.0:
            raise ValueError("prune_threshold must lie in [0, 1]")
        self.selection_mode = selection_mode
        self.policy = policy
        self.allow_learned_pruning = allow_learned_pruning
        self.prune_threshold = prune_threshold

    def _select_index(
        self, instance: KnapsackInstance, open_nodes: list[Node], incumbent: float
    ) -> int:
        if self.selection_mode == "best_bound":
            return max(
                range(len(open_nodes)),
                key=lambda i: (open_nodes[i].bound, open_nodes[i].level),
            )
        if self.selection_mode == "dfs":
            return max(
                range(len(open_nodes)),
                key=lambda i: (open_nodes[i].level, open_nodes[i].node_id),
            )
        assert self.policy is not None
        scores = []
        for node in open_nodes:
            feats = node_features(
                instance,
                level=node.level,
                weight=node.weight,
                value=node.value,
                bound=node.bound,
                incumbent=incumbent,
                decisions=node.decisions,
            )
            scores.append(self.policy.selection_score(feats))
        return max(
            range(len(open_nodes)),
            key=lambda i: (scores[i], open_nodes[i].bound, open_nodes[i].level),
        )

    def solve(
        self,
        instance: KnapsackInstance,
        *,
        oracle_decisions: tuple[int, ...] | None = None,
        collect_trace: bool = False,
    ) -> BnBResult:
        start = perf_counter()
        root = Node(
            level=0,
            weight=0.0,
            value=0.0,
            bound=instance.fractional_bound(0, 0.0, 0.0),
            decisions=(),
            node_id=0,
        )
        open_nodes = [root]
        incumbent = 0.0
        incumbent_bits = (0,) * instance.n_items
        next_node_id = 1
        nodes_popped = 0
        nodes_expanded = 0
        safe_prunes = 0
        learned_prunes = 0
        trace: list[TraceRecord] = []

        while open_nodes:
            idx = self._select_index(instance, open_nodes, incumbent)
            node = open_nodes.pop(idx)
            nodes_popped += 1

            if node.bound <= incumbent + 1e-10:
                safe_prunes += 1
                continue

            feats = node_features(
                instance,
                level=node.level,
                weight=node.weight,
                value=node.value,
                bound=node.bound,
                incumbent=incumbent,
                decisions=node.decisions,
            )
            if oracle_decisions is not None:
                label = int(node.decisions == oracle_decisions[: node.level])
                if collect_trace:
                    trace.append(TraceRecord(features=feats, oracle_expand=label))

            if (
                self.allow_learned_pruning
                and node.level > 0
                and self.policy is not None
                and self.policy.expand_probability(feats) < self.prune_threshold
            ):
                learned_prunes += 1
                continue

            if node.level == instance.n_items:
                if node.value > incumbent + 1e-10:
                    incumbent = node.value
                    incumbent_bits = node.decisions
                continue

            nodes_expanded += 1
            item = node.level

            include_weight = node.weight + instance.weights[item]
            if include_weight <= instance.capacity + 1e-10:
                include_value = node.value + instance.values[item]
                include_decisions = node.decisions + (1,)
                if item + 1 == instance.n_items and include_value > incumbent + 1e-10:
                    incumbent = include_value
                    incumbent_bits = include_decisions
                include_bound = instance.fractional_bound(item + 1, include_weight, include_value)
                if include_bound > incumbent + 1e-10:
                    open_nodes.append(
                        Node(
                            level=item + 1,
                            weight=include_weight,
                            value=include_value,
                            bound=include_bound,
                            decisions=include_decisions,
                            node_id=next_node_id,
                        )
                    )
                    next_node_id += 1

            exclude_decisions = node.decisions + (0,)
            exclude_bound = instance.fractional_bound(item + 1, node.weight, node.value)
            if item + 1 == instance.n_items and node.value > incumbent + 1e-10:
                incumbent = node.value
                incumbent_bits = exclude_decisions
            if exclude_bound > incumbent + 1e-10:
                open_nodes.append(
                    Node(
                        level=item + 1,
                        weight=node.weight,
                        value=node.value,
                        bound=exclude_bound,
                        decisions=exclude_decisions,
                        node_id=next_node_id,
                    )
                )
                next_node_id += 1

        elapsed = perf_counter() - start
        return BnBResult(
            objective=float(incumbent),
            decisions=incumbent_bits,
            nodes_popped=nodes_popped,
            nodes_expanded=nodes_expanded,
            safe_prunes=safe_prunes,
            learned_prunes=learned_prunes,
            elapsed_seconds=elapsed,
            optimality_guaranteed=not self.allow_learned_pruning,
            trace=trace,
        )
