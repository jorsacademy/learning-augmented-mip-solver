from __future__ import annotations

import numpy as np

from .problem import BinaryPackingMIP

FEATURE_NAMES = (
    "profit_over_max",
    "profit_zscore",
    "mean_capacity_fraction",
    "max_capacity_fraction",
    "std_capacity_fraction",
    "profit_resource_efficiency",
    "column_l2",
    "constraint_coverage",
    "instance_capacity_pressure",
    "items_per_constraint_scaled",
)


def extract_variable_features(instance: BinaryPackingMIP) -> np.ndarray:
    """Return solver-state-free per-variable features computed only from MIP coefficients."""
    c = instance.c
    scaled_A = instance.A / instance.b[:, None]
    profit_over_max = c / max(float(c.max()), 1e-9)
    c_std = float(c.std())
    profit_zscore = (c - c.mean()) / (c_std if c_std > 1e-9 else 1.0)
    mean_load = scaled_A.mean(axis=0)
    max_load = scaled_A.max(axis=0)
    std_load = scaled_A.std(axis=0)
    efficiency = profit_over_max / (mean_load + 1e-6)
    column_l2 = np.sqrt(np.sum(scaled_A**2, axis=0))
    coverage = np.mean(instance.A > 0.0, axis=0)
    pressure = float(np.mean(instance.b / np.maximum(instance.A.sum(axis=1), 1e-9)))
    size_feature = instance.n_items / max(instance.n_constraints, 1) / 20.0
    features = np.column_stack(
        [
            profit_over_max,
            profit_zscore,
            mean_load,
            max_load,
            std_load,
            efficiency,
            column_l2,
            coverage,
            np.full(instance.n_items, pressure),
            np.full(instance.n_items, size_feature),
        ]
    )
    if not np.all(np.isfinite(features)):
        raise ValueError("non-finite features generated")
    return features.astype(np.float32)
