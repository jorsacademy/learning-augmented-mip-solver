from __future__ import annotations

import numpy as np

from .features import extract_variable_features
from .problem import BinaryPackingMIP, generate_instance
from .solver import solve_mip


def solved_dataset(
    seeds: list[int],
    *,
    n_items: int,
    n_constraints: int,
) -> tuple[np.ndarray, np.ndarray, list[BinaryPackingMIP]]:
    """Generate instances and label each variable with an optimal HiGHS assignment."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    instances: list[BinaryPackingMIP] = []
    for seed in seeds:
        instance = generate_instance(seed, n_items=n_items, n_constraints=n_constraints)
        solved = solve_mip(instance, mip_rel_gap=0.0)
        if not solved.success or solved.x is None:
            raise RuntimeError(f"failed to solve training instance {seed}: {solved.message}")
        xs.append(extract_variable_features(instance))
        ys.append(solved.x.astype(np.float32))
        instances.append(instance)
    return np.vstack(xs), np.concatenate(ys), instances
