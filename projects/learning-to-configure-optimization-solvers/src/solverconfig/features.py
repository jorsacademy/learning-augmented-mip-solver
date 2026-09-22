from __future__ import annotations

from time import perf_counter

import numpy as np

from .instances import SetCoverInstance

FEATURE_NAMES = (
    "n_rows",
    "n_cols",
    "density",
    "cost_mean",
    "cost_std",
    "cost_cv",
    "row_degree_mean",
    "row_degree_std",
    "row_degree_min",
    "row_degree_max",
    "col_degree_mean",
    "col_degree_std",
    "col_degree_min",
    "col_degree_max",
    "singleton_row_fraction",
    "duplicate_column_fraction",
)


def extract_features(instance: SetCoverInstance) -> tuple[np.ndarray, float]:
    """Compute cheap static instance features; no solver calls are used."""
    start = perf_counter()
    a = instance.matrix
    c = instance.costs
    row_degree = a.sum(axis=1).astype(float)
    col_degree = a.sum(axis=0).astype(float)
    unique_cols = np.unique(a.T, axis=0).shape[0]
    duplicate_fraction = 1.0 - unique_cols / instance.n_cols
    mean_cost = float(c.mean())
    features = np.array(
        [
            instance.n_rows,
            instance.n_cols,
            float(a.mean()),
            mean_cost,
            float(c.std()),
            float(c.std() / max(mean_cost, 1e-12)),
            float(row_degree.mean()),
            float(row_degree.std()),
            float(row_degree.min()),
            float(row_degree.max()),
            float(col_degree.mean()),
            float(col_degree.std()),
            float(col_degree.min()),
            float(col_degree.max()),
            float(np.mean(row_degree == 1)),
            float(duplicate_fraction),
        ],
        dtype=float,
    )
    return features, perf_counter() - start
