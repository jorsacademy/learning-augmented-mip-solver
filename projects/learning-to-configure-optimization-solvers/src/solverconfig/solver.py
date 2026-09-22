from __future__ import annotations

from dataclasses import asdict, dataclass
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any

import numpy as np

from .audit import audit_solution
from .configurations import SolverConfiguration
from .instances import SetCoverInstance


@dataclass(frozen=True)
class SolveResult:
    instance_id: str
    configuration: str
    status: str
    optimal: bool
    feasible: bool
    runtime: float
    objective: float | None
    objective_recomputed: float | None
    mip_gap: float | None
    node_count: int | None
    max_integrality_violation: float
    max_cover_violation: float
    decision_length_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _lp_text(instance: SetCoverInstance) -> str:
    terms = " + ".join(f"{cost:.12g} x{j}" for j, cost in enumerate(instance.costs))
    lines = ["Minimize", f" obj: {terms}", "Subject To"]
    for i, row in enumerate(instance.matrix):
        cols = np.flatnonzero(row)
        lhs = " + ".join(f"x{int(j)}" for j in cols)
        lines.append(f" cover_{i}: {lhs} >= 1")
    lines.extend(["Binary"])
    lines.extend(f" x{j}" for j in range(instance.n_cols))
    lines.append("End")
    return "\n".join(lines) + "\n"


def solve_with_highs(
    instance: SetCoverInstance,
    configuration: SolverConfiguration,
    *,
    time_limit: float,
    random_seed: int,
) -> SolveResult:
    """Solve one instance/config pair with a common budget and independent audit."""
    try:
        import highspy
    except ImportError as exc:  # pragma: no cover - exercised in environments without HiGHS
        raise RuntimeError("highspy is required to run solver experiments") from exc

    highs = highspy.Highs()
    common_options: dict[str, Any] = {
        "output_flag": False,
        "log_to_console": False,
        "threads": 1,
        "parallel": "off",
        "time_limit": float(time_limit),
        "random_seed": int(random_seed),
    }
    for name, value in {**common_options, **configuration.options}.items():
        option_status = highs.setOptionValue(name, value)
        if option_status == highspy.HighsStatus.kError:
            raise ValueError(f"HiGHS rejected option {name}={value!r}")

    with NamedTemporaryFile(mode="w", suffix=".lp", encoding="utf-8") as model_file:
        model_file.write(_lp_text(instance))
        model_file.flush()
        read_status = highs.readModel(model_file.name)
        if read_status == highspy.HighsStatus.kError:
            raise RuntimeError(f"HiGHS failed to read generated model {instance.instance_id}")

        start = perf_counter()
        run_status = highs.run()
        runtime = perf_counter() - start
        if run_status == highspy.HighsStatus.kError:
            raise RuntimeError(f"HiGHS solve error on {instance.instance_id}")

    model_status = highs.getModelStatus()
    status = highs.modelStatusToString(model_status)
    optimal = model_status == highspy.HighsModelStatus.kOptimal
    info = highs.getInfo()
    solution = highs.getSolution()

    values: list[float] | None = None
    if (
        getattr(info, "primal_solution_status", None)
        == highspy.SolutionStatus.kSolutionStatusFeasible
    ):
        values = list(solution.col_value)
    audit = audit_solution(instance, values)

    reported_objective = None
    if audit.feasible:
        reported_objective = float(
            getattr(info, "objective_function_value", highs.getObjectiveValue())
        )

    gap = getattr(info, "mip_gap", None)
    node_count = getattr(info, "mip_node_count", None)
    return SolveResult(
        instance_id=instance.instance_id,
        configuration=configuration.name,
        status=status,
        optimal=bool(optimal),
        feasible=audit.feasible,
        runtime=float(runtime),
        objective=reported_objective,
        objective_recomputed=audit.objective_recomputed,
        mip_gap=float(gap) if gap is not None and np.isfinite(gap) else None,
        node_count=int(node_count) if node_count is not None and node_count >= 0 else None,
        max_integrality_violation=audit.max_integrality_violation,
        max_cover_violation=audit.max_cover_violation,
        decision_length_ok=audit.decision_length_ok,
    )
