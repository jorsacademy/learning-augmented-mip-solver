from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryPackingMIP:
    """A binary packing MILP: maximize c^T x subject to A x <= b, x in {0,1}^n."""

    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    name: str = "instance"

    def __post_init__(self) -> None:
        c = np.asarray(self.c, dtype=float)
        A = np.asarray(self.A, dtype=float)
        b = np.asarray(self.b, dtype=float)
        if c.ndim != 1:
            raise ValueError("c must be one-dimensional")
        if A.ndim != 2 or A.shape[1] != c.size:
            raise ValueError("A must have shape (m, n) matching c")
        if b.shape != (A.shape[0],):
            raise ValueError("b must have one entry per constraint")
        if np.any(c <= 0) or np.any(A < 0) or np.any(b <= 0):
            raise ValueError("packing data must have positive profits/capacities and nonnegative A")
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "b", b)

    @property
    def n_items(self) -> int:
        return int(self.c.size)

    @property
    def n_constraints(self) -> int:
        return int(self.A.shape[0])

    def objective(self, x: np.ndarray) -> float:
        return float(self.c @ np.asarray(x, dtype=float))

    def feasible(self, x: np.ndarray, tol: float = 1e-8) -> bool:
        x = np.asarray(x, dtype=float)
        return bool(
            x.shape == (self.n_items,)
            and np.all(x >= -tol)
            and np.all(x <= 1.0 + tol)
            and np.all(np.abs(x - np.rint(x)) <= tol)
            and np.all(self.A @ x <= self.b + tol)
        )


def generate_instance(
    seed: int,
    *,
    n_items: int = 24,
    n_constraints: int = 5,
    capacity_ratio: float = 0.42,
) -> BinaryPackingMIP:
    """Generate a homogeneous but nontrivial family of binary packing MILPs.

    A latent item quality affects both profit and resource efficiency. This creates learnable
    cross-instance structure without exposing solution labels in the input features.
    """
    if n_items < 4 or n_constraints < 1:
        raise ValueError("n_items >= 4 and n_constraints >= 1 are required")
    if not 0.15 <= capacity_ratio <= 0.8:
        raise ValueError("capacity_ratio must be in [0.15, 0.8]")

    rng = np.random.default_rng(seed)
    quality = rng.normal(0.0, 1.0, size=n_items)

    profits = 12.0 + 4.5 * quality + rng.normal(0.0, 1.8, size=n_items)
    profits = np.clip(profits, 1.0, None)

    row_scale = rng.uniform(0.7, 1.4, size=(n_constraints, 1))
    raw = rng.lognormal(mean=0.0, sigma=0.38, size=(n_constraints, n_items))
    efficiency_effect = np.clip(1.0 - 0.12 * quality, 0.55, 1.55)
    A = row_scale * raw * efficiency_effect[None, :]

    # Add sparse structure while ensuring each column still participates in the model.
    mask = rng.random(size=A.shape) < 0.13
    A = np.where(mask, 0.0, A)
    empty_cols = np.where(np.all(A == 0.0, axis=0))[0]
    for col in empty_cols:
        A[rng.integers(0, n_constraints), col] = rng.uniform(0.5, 1.5)

    jitter = rng.uniform(0.93, 1.07, size=n_constraints)
    b = capacity_ratio * A.sum(axis=1) * jitter
    b = np.maximum(b, np.max(A, axis=1) * 1.05)

    return BinaryPackingMIP(c=profits, A=A, b=b, name=f"packing-{seed}")
