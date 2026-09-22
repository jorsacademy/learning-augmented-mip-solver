from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SolverConfiguration:
    name: str
    options: dict[str, Any]


CANDIDATE_CONFIGURATIONS: tuple[SolverConfiguration, ...] = (
    SolverConfiguration("default", {}),
    SolverConfiguration("presolve_off", {"presolve": "off"}),
    SolverConfiguration("heuristics_off", {"mip_heuristic_effort": 0.0}),
    SolverConfiguration("heuristics_high", {"mip_heuristic_effort": 0.20}),
    SolverConfiguration("symmetry_off", {"mip_detect_symmetry": False}),
    SolverConfiguration(
        "lean_search",
        {"presolve": "on", "mip_heuristic_effort": 0.0, "mip_detect_symmetry": False},
    ),
)


def by_name(name: str) -> SolverConfiguration:
    for config in CANDIDATE_CONFIGURATIONS:
        if config.name == name:
            return config
    raise KeyError(name)
