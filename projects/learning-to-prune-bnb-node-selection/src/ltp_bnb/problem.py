from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KnapsackInstance:
    """A 0-1 knapsack instance ordered by non-increasing value/weight ratio."""

    weights: np.ndarray
    values: np.ndarray
    capacity: float

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if weights.ndim != 1 or values.ndim != 1 or len(weights) != len(values):
            raise ValueError("weights and values must be one-dimensional arrays of equal length")
        if len(weights) == 0:
            raise ValueError("an instance must contain at least one item")
        if np.any(weights <= 0) or np.any(values < 0):
            raise ValueError("weights must be positive and values non-negative")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        ratios = values / weights
        if np.any(ratios[:-1] + 1e-12 < ratios[1:]):
            raise ValueError("items must be sorted by non-increasing value/weight ratio")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "capacity", float(self.capacity))

    @property
    def n_items(self) -> int:
        return len(self.weights)

    @property
    def total_value(self) -> float:
        return float(np.sum(self.values))

    def fractional_bound(self, level: int, weight: float, value: float) -> float:
        """Fractional-knapsack upper bound for a prefix-fixed B&B node."""
        if weight > self.capacity + 1e-12:
            return float("-inf")
        remaining = self.capacity - weight
        bound = float(value)
        for idx in range(level, self.n_items):
            item_weight = self.weights[idx]
            item_value = self.values[idx]
            if item_weight <= remaining + 1e-12:
                remaining -= item_weight
                bound += item_value
            else:
                if remaining > 0:
                    bound += item_value * (remaining / item_weight)
                break
        return bound


def generate_knapsack(seed: int, n_items: int = 14) -> KnapsackInstance:
    """Generate a deterministic random knapsack instance and sort it for the LP bound."""
    if n_items < 2:
        raise ValueError("n_items must be at least 2")
    rng = np.random.default_rng(seed)
    weights = rng.integers(2, 31, size=n_items).astype(float)
    values = weights * rng.uniform(1.2, 4.2, size=n_items) + rng.uniform(0.0, 18.0, size=n_items)
    capacity = float(max(1, int(np.sum(weights) * rng.uniform(0.35, 0.55))))
    order = np.argsort(-(values / weights), kind="stable")
    return KnapsackInstance(weights=weights[order], values=values[order], capacity=capacity)


def brute_force_optimum(instance: KnapsackInstance) -> tuple[float, tuple[int, ...]]:
    """Exact reference solve for small validation instances."""
    n = instance.n_items
    if n > 24:
        raise ValueError("brute_force_optimum is intentionally limited to <= 24 items")
    best_value = -1.0
    best_bits: tuple[int, ...] | None = None
    for mask in range(1 << n):
        weight = 0.0
        value = 0.0
        bits = [0] * n
        feasible = True
        for idx in range(n):
            if mask & (1 << idx):
                weight += instance.weights[idx]
                if weight > instance.capacity + 1e-12:
                    feasible = False
                    break
                value += instance.values[idx]
                bits[idx] = 1
        if feasible and value > best_value + 1e-12:
            best_value = value
            best_bits = tuple(bits)
    assert best_bits is not None
    return float(best_value), best_bits
