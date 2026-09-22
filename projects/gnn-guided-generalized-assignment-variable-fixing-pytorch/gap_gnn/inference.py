from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from .graph import instance_to_features
from .planning import GAPInstance, GAPMILP, GAPSolution


@dataclass(frozen=True)
class FixingProposal:
    fixed_jobs: dict
    confidences: dict
    predicted_assignment: np.ndarray
    probabilities: np.ndarray


@dataclass(frozen=True)
class RepairResult:
    solution: GAPSolution
    initially_fixed_jobs: int
    finally_fixed_jobs: int
    relaxed_jobs: int
    wrong_initial_fixes: int


@dataclass(frozen=True)
class CalibrationResult:
    confidence_threshold: float
    validation_precision: float
    validation_coverage: float
    fixed_predictions: int


def calibrate_confidence_threshold(
    model,
    validation_instances,
    validation_labels: np.ndarray,
    *,
    target_precision: float = 0.95,
    margin_threshold: float = 0.05,
    minimum_coverage: float = 0.05,
) -> CalibrationResult:
    """
    Choose the lowest confidence threshold that maximizes coverage while
    meeting a target validation fixing precision.

    If the requested precision cannot be achieved at minimum coverage, return a
    threshold above 1 so no test variables are fixed. This is conservative by
    construction and avoids fabricating certainty.
    """
    if not 0.0 < target_precision <= 1.0:
        raise ValueError("target_precision must be in (0,1]")
    if not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be in [0,1]")

    confidences = []
    correctness = []

    model.eval()
    for instance, labels in zip(validation_instances, validation_labels):
        mf, jf, ef = instance_to_features(instance)
        with torch.no_grad():
            logits = model(mf[None, ...], jf[None, ...], ef[None, ...])[0]
            probs = torch.softmax(logits, dim=0).cpu().numpy()

        for j in range(instance.n_jobs):
            order = np.argsort(probs[:, j])[::-1]
            top, second = int(order[0]), int(order[1])
            margin = float(probs[top, j] - probs[second, j])
            if margin < margin_threshold:
                continue
            confidences.append(float(probs[top, j]))
            correctness.append(int(top == int(labels[j])))

    if not confidences:
        return CalibrationResult(1.01, 1.0, 0.0, 0)

    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correctness, dtype=float)
    total_possible = sum(x.n_jobs for x in validation_instances)

    best = None
    for threshold in np.unique(conf):
        mask = conf >= threshold - 1e-12
        count = int(mask.sum())
        coverage = count / max(total_possible, 1)
        precision = float(corr[mask].mean())
        if coverage + 1e-12 < minimum_coverage:
            continue
        if precision + 1e-12 < target_precision:
            continue
        candidate = (coverage, -float(threshold), precision, count, float(threshold))
        if best is None or candidate > best:
            best = candidate

    if best is None:
        return CalibrationResult(1.01, 1.0, 0.0, 0)

    coverage, _, precision, count, threshold = best
    return CalibrationResult(
        confidence_threshold=float(threshold),
        validation_precision=float(precision),
        validation_coverage=float(coverage),
        fixed_predictions=int(count),
    )


def propose_confident_fixings(
    model,
    instance: GAPInstance,
    *,
    confidence_threshold: float = 0.92,
    margin_threshold: float = 0.15,
) -> FixingProposal:
    model.eval()
    mf, jf, ef = instance_to_features(instance)
    with torch.no_grad():
        logits = model(mf[None, ...], jf[None, ...], ef[None, ...])[0]
        probabilities = torch.softmax(logits, dim=0).cpu().numpy()

    predicted = np.argmax(probabilities, axis=0)
    fixed, confidence = {}, {}
    for j in range(instance.n_jobs):
        order = np.argsort(probabilities[:, j])[::-1]
        top, second = order[0], order[1]
        p_top = float(probabilities[top, j])
        margin = p_top - float(probabilities[second, j])

        fits_alone = instance.resources[top, j] <= instance.capacities[top] + 1e-9
        if fits_alone and p_top >= confidence_threshold and margin >= margin_threshold:
            fixed[j] = int(top)
            confidence[j] = p_top

    return FixingProposal(
        fixed_jobs=fixed,
        confidences=confidence,
        predicted_assignment=predicted,
        probabilities=probabilities,
    )


def exact_repair_with_relaxation(
    instance: GAPInstance,
    proposal: FixingProposal,
    *,
    optimal_assignment: np.ndarray | None = None,
) -> RepairResult:
    """
    Solve the residual MILP exactly. If high-confidence fixings collectively
    make the residual problem infeasible, relax the least-confident fixing
    until feasibility is restored.
    """
    planner = GAPMILP(instance)
    current = dict(proposal.fixed_jobs)
    ordered = sorted(
        current,
        key=lambda j: proposal.confidences[j],
    )
    relaxed = 0

    while True:
        solution = planner.solve(fixed_jobs=current)
        if solution is not None:
            break
        if not ordered:
            solution = planner.solve()
            if solution is None:
                raise RuntimeError("base GAP unexpectedly infeasible")
            break
        j = ordered.pop(0)
        current.pop(j, None)
        relaxed += 1

    wrong = 0
    if optimal_assignment is not None:
        wrong = sum(
            int(optimal_assignment[j] != m)
            for j, m in proposal.fixed_jobs.items()
        )

    return RepairResult(
        solution=solution,
        initially_fixed_jobs=len(proposal.fixed_jobs),
        finally_fixed_jobs=len(current),
        relaxed_jobs=relaxed,
        wrong_initial_fixes=wrong,
    )


def oracle_fixing_for_same_jobs(
    instance: GAPInstance,
    proposal: FixingProposal,
    optimal_assignment: np.ndarray,
) -> GAPSolution:
    fixed = {
        j: int(optimal_assignment[j])
        for j in proposal.fixed_jobs
    }
    solution = GAPMILP(instance).solve(fixed_jobs=fixed)
    if solution is None:
        raise RuntimeError("oracle fixing cannot make an optimal assignment infeasible")
    return solution
