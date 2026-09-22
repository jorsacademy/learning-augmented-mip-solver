from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SetCoverInstance:
    """Binary set-covering MILP: min c^T x subject to A x >= 1, x in {0,1}."""

    instance_id: str
    regime: str
    matrix: np.ndarray
    costs: np.ndarray
    generation_seed: int

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2:
            raise ValueError("matrix must be two-dimensional")
        if self.costs.ndim != 1 or self.costs.shape[0] != self.matrix.shape[1]:
            raise ValueError("cost vector length must equal number of columns")
        if np.any((self.matrix != 0) & (self.matrix != 1)):
            raise ValueError("set-cover matrix must be binary")
        if np.any(self.matrix.sum(axis=1) == 0):
            raise ValueError("every row must be coverable")
        if np.any(self.costs <= 0):
            raise ValueError("all costs must be strictly positive")

    @property
    def n_rows(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_cols(self) -> int:
        return int(self.matrix.shape[1])


def generate_set_cover(
    *,
    instance_id: str,
    regime: str,
    n_rows: int,
    n_cols: int,
    density: float,
    seed: int,
) -> SetCoverInstance:
    """Generate a feasible heterogeneous synthetic set-covering instance.

    Regimes alter cost structure and column overlap without creating train/test variants of a
    common base instance. Every generated seed defines an independent instance.
    """
    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("dimensions must be positive")

    rng = np.random.default_rng(seed)
    matrix = (rng.random((n_rows, n_cols)) < density).astype(np.int8)

    # Guarantee feasibility independently for every row.
    uncovered = np.flatnonzero(matrix.sum(axis=1) == 0)
    if uncovered.size:
        matrix[uncovered, rng.integers(0, n_cols, size=uncovered.size)] = 1

    if regime == "uniform":
        costs = rng.uniform(1.0, 10.0, size=n_cols)
    elif regime == "skewed_cost":
        costs = 1.0 + rng.lognormal(mean=0.7, sigma=0.9, size=n_cols)
    elif regime == "degree_correlated":
        degree = matrix.sum(axis=0).astype(float)
        noise = rng.uniform(0.0, 2.0, size=n_cols)
        costs = 1.0 + 0.25 * degree + noise
    elif regime == "near_duplicate":
        # Add controlled structural redundancy without making literal train/test variants.
        for col in range(1, n_cols, 5):
            source = col - 1
            flip_count = max(1, n_rows // 20)
            matrix[:, col] = matrix[:, source]
            flip_rows = rng.choice(n_rows, size=flip_count, replace=False)
            matrix[flip_rows, col] = 1 - matrix[flip_rows, col]
        uncovered = np.flatnonzero(matrix.sum(axis=1) == 0)
        if uncovered.size:
            matrix[uncovered, rng.integers(0, n_cols, size=uncovered.size)] = 1
        costs = rng.uniform(1.0, 10.0, size=n_cols)
    else:
        raise ValueError(f"unknown regime: {regime}")

    return SetCoverInstance(
        instance_id=instance_id,
        regime=regime,
        matrix=matrix,
        costs=costs.astype(float),
        generation_seed=seed,
    )


def generate_split(
    *,
    split: str,
    count: int,
    seed_start: int,
    row_range: tuple[int, int],
    col_range: tuple[int, int],
    density_range: tuple[float, float],
    regimes: tuple[str, ...],
) -> list[SetCoverInstance]:
    """Generate independent instances for one experimental split."""
    result: list[SetCoverInstance] = []
    for offset in range(count):
        seed = seed_start + offset
        rng = np.random.default_rng(seed)
        n_rows = int(rng.integers(row_range[0], row_range[1] + 1))
        n_cols = int(rng.integers(col_range[0], col_range[1] + 1))
        density = float(rng.uniform(density_range[0], density_range[1]))
        regime = regimes[offset % len(regimes)]
        result.append(
            generate_set_cover(
                instance_id=f"{split}-{offset:04d}",
                regime=regime,
                n_rows=n_rows,
                n_cols=n_cols,
                density=density,
                seed=seed,
            )
        )
    return result
