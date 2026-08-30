from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from scipy.optimize import linprog

from .cuts import CoverCut, cut_features, cut_row, generate_cover_cuts
from .models import LearnedComponents
from .presolve import PRESOLVE_CONFIGS, PresolveConfig, apply_presolve, instance_features
from .problem import BinaryPackingMIP


CutPolicy = Literal["none", "efficacy", "oracle", "learned"]
BranchPolicy = Literal["most_fractional", "strong", "learned"]
PrimalPolicy = Literal["none", "greedy", "learned"]
PresolvePolicy = Literal["none", "fixing", "redundancy", "full", "learned"]


@dataclass(frozen=True)
class SolverConfig:
    presolve: PresolvePolicy = "full"
    cuts: CutPolicy = "efficacy"
    branching: BranchPolicy = "most_fractional"
    primal: PrimalPolicy = "greedy"
    cut_rounds: int = 3
    max_nodes: int = 10_000
    integrality_tol: float = 1e-7


@dataclass(frozen=True)
class SolveResult:
    objective: float
    nodes_processed: int
    lp_solves: int
    optimal: bool
    presolve_config: str
    cuts_added: int
    initial_incumbent: float


@dataclass(frozen=True)
class _LPResult:
    objective: float
    x: np.ndarray


def _solve_lp(
    problem: BinaryPackingMIP,
    cuts: list[CoverCut],
    bounds: dict[int, tuple[float, float]] | None = None,
) -> _LPResult | None:
    rows = [row.copy() for row in problem.A]
    rhs = list(problem.b.astype(float))
    for cut in cuts:
        rows.append(cut_row(problem.n_vars, cut))
        rhs.append(cut.rhs)
    variable_bounds = [(0.0, 1.0)] * problem.n_vars
    if bounds:
        variable_bounds = variable_bounds.copy()
        for index, bound in bounds.items():
            variable_bounds[index] = bound
    result = linprog(
        -problem.profits,
        A_ub=np.asarray(rows),
        b_ub=np.asarray(rhs),
        bounds=variable_bounds,
        method="highs",
    )
    if result.status == 2:
        return None
    if not result.success or result.x is None or result.fun is None:
        raise RuntimeError(f"LP relaxation failed: {result.message}")
    return _LPResult(objective=float(-result.fun), x=np.asarray(result.x, dtype=float))


def _fractional_candidates(x: np.ndarray, tol: float) -> np.ndarray:
    fractionality = np.minimum(x, 1.0 - x)
    return np.flatnonzero(fractionality > tol)


def _branch_features(problem: BinaryPackingMIP, x: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    objective_scale = max(float(np.max(np.abs(problem.profits))), 1.0)
    coefficient_scale = max(float(np.max(np.abs(problem.A))), 1.0)
    density = np.mean(problem.A > 0, axis=0)
    rows = []
    for j in candidates:
        rows.append(
            [
                x[j],
                min(x[j], 1.0 - x[j]),
                problem.profits[j] / objective_scale,
                density[j],
                np.max(problem.A[:, j]) / coefficient_scale,
                np.mean(problem.A[:, j]) / coefficient_scale,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _primal_features(problem: BinaryPackingMIP) -> np.ndarray:
    objective_scale = max(float(np.max(np.abs(problem.profits))), 1.0)
    coefficient_scale = max(float(np.max(np.abs(problem.A))), 1.0)
    density = np.mean(problem.A > 0, axis=0)
    return np.stack(
        [
            problem.profits / objective_scale,
            density,
            np.mean(problem.A, axis=0) / coefficient_scale,
            np.max(problem.A, axis=0) / coefficient_scale,
            problem.profits / np.maximum(np.mean(problem.A, axis=0), 1e-6) / objective_scale,
        ],
        axis=1,
    ).astype(np.float32)


def _choose_presolve(
    problem: BinaryPackingMIP,
    policy: PresolvePolicy,
    learned: LearnedComponents,
) -> PresolveConfig:
    if policy != "learned":
        return policy
    if learned.presolve is None:
        raise ValueError("learned presolve policy requires a model")
    with torch.no_grad():
        logits = learned.presolve(torch.tensor(instance_features(problem)).unsqueeze(0))
    return PRESOLVE_CONFIGS[int(torch.argmax(logits, dim=1).item())]


def _select_cut(
    problem: BinaryPackingMIP,
    lp: _LPResult,
    cuts: list[CoverCut],
    candidates: list[CoverCut],
    policy: CutPolicy,
    learned: LearnedComponents,
) -> tuple[int, int]:
    features = np.stack([cut_features(problem, lp.x, cut) for cut in candidates])
    if policy == "efficacy":
        return int(np.argmax(features[:, 1])), 0
    if policy == "learned":
        if learned.cuts is None:
            raise ValueError("learned cut policy requires a model")
        with torch.no_grad():
            scores = learned.cuts(torch.tensor(features, dtype=torch.float32)).cpu().numpy()
        return int(np.argmax(scores)), 0
    if policy == "oracle":
        gains = []
        for candidate in candidates:
            child = _solve_lp(problem, [*cuts, candidate])
            child_objective = -np.inf if child is None else child.objective
            gains.append(lp.objective - child_objective)
        return int(np.argmax(gains)), len(candidates)
    raise ValueError("cut selector called with unsupported policy")


def _root_cut_loop(
    problem: BinaryPackingMIP,
    policy: CutPolicy,
    rounds: int,
    learned: LearnedComponents,
) -> tuple[list[CoverCut], _LPResult, int]:
    root = _solve_lp(problem, [])
    if root is None:
        raise RuntimeError("presolved root LP is infeasible")
    lp_solves = 1
    cuts: list[CoverCut] = []
    current = root
    if policy == "none":
        return cuts, current, lp_solves
    for _ in range(rounds):
        candidates = generate_cover_cuts(problem, current.x)
        existing = {cut.indices for cut in cuts}
        candidates = [cut for cut in candidates if cut.indices not in existing]
        if not candidates:
            break
        selected, extra = _select_cut(problem, current, cuts, candidates, policy, learned)
        lp_solves += extra
        cuts.append(candidates[selected])
        current = _solve_lp(problem, cuts)
        lp_solves += 1
        if current is None:
            raise RuntimeError("valid global cuts unexpectedly made root infeasible")
    return cuts, current, lp_solves


def _greedy_incumbent(
    problem: BinaryPackingMIP,
    policy: PrimalPolicy,
    learned: LearnedComponents,
) -> float:
    if policy == "none":
        return 0.0
    if policy == "greedy":
        denominator = np.maximum(problem.A.mean(axis=0), 1e-6)
        scores = problem.profits / denominator
    elif policy == "learned":
        if learned.primal is None:
            raise ValueError("learned primal policy requires a model")
        with torch.no_grad():
            scores = learned.primal(torch.tensor(_primal_features(problem))).cpu().numpy()
    else:
        raise ValueError("unknown primal policy")
    order = np.argsort(-scores)
    x = np.zeros(problem.n_vars)
    activity = np.zeros(problem.n_constraints)
    for j in order:
        if problem.profits[j] <= 0:
            continue
        candidate_activity = activity + problem.A[:, j]
        if np.all(candidate_activity <= problem.b + 1e-10):
            x[j] = 1.0
            activity = candidate_activity
    return float(problem.profits @ x)


def _choose_branch(
    problem: BinaryPackingMIP,
    lp: _LPResult,
    cuts: list[CoverCut],
    bounds: dict[int, tuple[float, float]],
    candidates: np.ndarray,
    policy: BranchPolicy,
    learned: LearnedComponents,
) -> tuple[int, int]:
    if policy == "most_fractional":
        values = np.minimum(lp.x[candidates], 1.0 - lp.x[candidates])
        return int(candidates[int(np.argmax(values))]), 0
    if policy == "learned":
        if learned.branching is None:
            raise ValueError("learned branching policy requires a model")
        features = _branch_features(problem, lp.x, candidates)
        with torch.no_grad():
            scores = learned.branching(torch.tensor(features)).cpu().numpy()
        return int(candidates[int(np.argmax(scores))]), 0
    if policy == "strong":
        scores = []
        solves = 0
        fallback = abs(lp.objective) + 1.0
        for j in candidates:
            down = dict(bounds)
            up = dict(bounds)
            down[int(j)] = (0.0, 0.0)
            up[int(j)] = (1.0, 1.0)
            down_lp = _solve_lp(problem, cuts, down)
            up_lp = _solve_lp(problem, cuts, up)
            solves += 2
            down_gain = fallback if down_lp is None else max(0.0, lp.objective - down_lp.objective)
            up_gain = fallback if up_lp is None else max(0.0, lp.objective - up_lp.objective)
            scores.append(min(down_gain, up_gain) + 1e-6 * max(down_gain, up_gain))
        return int(candidates[int(np.argmax(scores))]), solves
    raise ValueError("unknown branching policy")


def solve(
    original: BinaryPackingMIP,
    config: SolverConfig = SolverConfig(),
    learned: LearnedComponents | None = None,
) -> SolveResult:
    if config.cut_rounds < 0 or config.max_nodes < 1:
        raise ValueError("invalid solver limits")
    learned = LearnedComponents() if learned is None else learned.eval()
    selected_presolve = _choose_presolve(original, config.presolve, learned)
    presolved = apply_presolve(original, selected_presolve)
    problem = presolved.problem

    cuts, root_lp, lp_solves = _root_cut_loop(
        problem, config.cuts, config.cut_rounds, learned
    )
    incumbent = _greedy_incumbent(problem, config.primal, learned)
    initial_incumbent = incumbent + presolved.objective_offset

    stack: list[dict[int, tuple[float, float]]] = [{}]
    nodes = 0
    complete = True

    while stack:
        if nodes >= config.max_nodes:
            complete = False
            break
        bounds = stack.pop()
        lp = root_lp if not bounds and nodes == 0 else _solve_lp(problem, cuts, bounds)
        if not (not bounds and nodes == 0):
            lp_solves += 1
        nodes += 1
        if lp is None or lp.objective <= incumbent + 1e-9:
            continue
        candidates = _fractional_candidates(lp.x, config.integrality_tol)
        if candidates.size == 0:
            incumbent = max(incumbent, float(problem.profits @ np.rint(lp.x)))
            continue
        variable, extra_solves = _choose_branch(
            problem,
            lp,
            cuts,
            bounds,
            candidates,
            config.branching,
            learned,
        )
        lp_solves += extra_solves
        down = dict(bounds)
        up = dict(bounds)
        down[variable] = (0.0, 0.0)
        up[variable] = (1.0, 1.0)
        stack.append(down)
        stack.append(up)

    return SolveResult(
        objective=incumbent + presolved.objective_offset,
        nodes_processed=nodes,
        lp_solves=lp_solves,
        optimal=complete,
        presolve_config=selected_presolve,
        cuts_added=len(cuts),
        initial_incumbent=initial_incumbent,
    )
