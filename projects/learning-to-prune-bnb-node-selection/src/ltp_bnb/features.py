from __future__ import annotations

import numpy as np

from .problem import KnapsackInstance

FEATURE_NAMES = (
    "depth_fraction",
    "capacity_used_fraction",
    "capacity_slack_fraction",
    "node_value_fraction",
    "upper_bound_fraction",
    "remaining_bound_fraction",
    "incumbent_fraction",
    "bound_over_incumbent_gap_fraction",
    "remaining_items_fraction",
    "last_decision",
)


def node_features(
    instance: KnapsackInstance,
    *,
    level: int,
    weight: float,
    value: float,
    bound: float,
    incumbent: float,
    decisions: tuple[int, ...],
) -> np.ndarray:
    """Scale-free node/state features that do not use oracle labels or the true optimum."""
    total_value = max(instance.total_value, 1e-9)
    capacity = max(instance.capacity, 1e-9)
    last_decision = float(decisions[-1]) if decisions else 0.5
    return np.asarray(
        [
            level / instance.n_items,
            weight / capacity,
            max(instance.capacity - weight, 0.0) / capacity,
            value / total_value,
            bound / total_value,
            max(bound - value, 0.0) / total_value,
            incumbent / total_value,
            max(bound - incumbent, 0.0) / total_value,
            (instance.n_items - level) / instance.n_items,
            last_decision,
        ],
        dtype=float,
    )
