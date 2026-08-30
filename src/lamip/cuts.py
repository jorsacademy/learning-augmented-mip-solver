from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problem import BinaryPackingMIP


@dataclass(frozen=True)
class CoverCut:
    indices: tuple[int, ...]

    @property
    def rhs(self) -> float:
        return float(len(self.indices) - 1)


def _minimal_cover(weights: np.ndarray, capacity: float, ordering: np.ndarray) -> tuple[int, ...] | None:
    chosen: list[int] = []
    total = 0.0
    for index in ordering:
        chosen.append(int(index))
        total += float(weights[index])
        if total > capacity + 1e-12:
            break
    if total <= capacity + 1e-12:
        return None
    changed = True
    while changed:
        changed = False
        for index in tuple(chosen):
            if total - float(weights[index]) > capacity + 1e-12:
                chosen.remove(index)
                total -= float(weights[index])
                changed = True
                break
    return tuple(sorted(chosen))


def generate_cover_cuts(
    problem: BinaryPackingMIP,
    x: np.ndarray,
    violation_tol: float = 1e-8,
) -> list[CoverCut]:
    if x.shape != (problem.n_vars,):
        raise ValueError("x has wrong shape")
    cuts: list[CoverCut] = []
    seen: set[tuple[int, ...]] = set()
    profit_density = problem.profits / np.maximum(problem.A.mean(axis=0), 1e-6)
    global_orders = [
        np.argsort(-x),
        np.argsort(-np.minimum(x, 1.0 - x)),
        np.argsort(-profit_density),
    ]
    for row, capacity in zip(problem.A, problem.b, strict=True):
        active = np.flatnonzero(row > 1e-12)
        if active.size < 2:
            continue
        row_orders = [
            active[np.argsort(-x[active])],
            active[np.argsort(-row[active])],
            active[np.argsort(-(x[active] * row[active]))],
        ]
        for order in [*row_orders, *global_orders]:
            order = np.asarray([j for j in order if row[j] > 1e-12], dtype=int)
            if order.size < 2:
                continue
            cover = _minimal_cover(row, float(capacity), order)
            if cover is None or cover in seen:
                continue
            violation = float(x[list(cover)].sum() - (len(cover) - 1))
            if violation > violation_tol:
                cuts.append(CoverCut(cover))
                seen.add(cover)
    return cuts


def cut_row(n_vars: int, cut: CoverCut) -> np.ndarray:
    row = np.zeros(n_vars)
    row[list(cut.indices)] = 1.0
    return row


def cut_features(problem: BinaryPackingMIP, x: np.ndarray, cut: CoverCut) -> np.ndarray:
    idx = np.asarray(cut.indices, dtype=int)
    violation = float(x[idx].sum() - cut.rhs)
    efficacy = violation / max(float(np.sqrt(idx.size)), 1e-12)
    support = idx.size / problem.n_vars
    objective_share = float(problem.profits[idx].sum() / max(problem.profits.sum(), 1e-12))
    mean_fractionality = float(np.minimum(x[idx], 1.0 - x[idx]).mean())
    max_fractionality = float(np.minimum(x[idx], 1.0 - x[idx]).max())
    return np.asarray(
        [violation, efficacy, support, objective_share, mean_fractionality, max_fractionality],
        dtype=np.float32,
    )
