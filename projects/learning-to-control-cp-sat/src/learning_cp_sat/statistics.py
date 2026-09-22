from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": math.nan, "std": math.nan, "median": math.nan, "p90": math.nan}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
    }


def paired_difference(
    a: Iterable[float],
    b: Iterable[float],
    *,
    bootstrap_seed: int = 0,
    bootstrap_samples: int = 2_000,
) -> dict[str, float | int]:
    a_array = np.asarray(list(a), dtype=np.float64)
    b_array = np.asarray(list(b), dtype=np.float64)
    if a_array.shape != b_array.shape or a_array.size == 0:
        raise ValueError("paired samples must have the same non-zero length")
    differences = a_array - b_array
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.choice(differences, size=(bootstrap_samples, differences.size), replace=True)
    means = draws.mean(axis=1)
    return {
        "n": int(differences.size),
        "mean_difference": float(differences.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }
