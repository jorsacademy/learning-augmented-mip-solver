from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .instances import GraphColoringInstance


@dataclass(frozen=True)
class AuditResult:
    feasible: bool
    missing_vertices: tuple[int, ...]
    out_of_domain_vertices: tuple[int, ...]
    violated_edges: tuple[tuple[int, int], ...]


def audit_assignment(
    instance: GraphColoringInstance,
    assignment: Sequence[int | None],
) -> AuditResult:
    missing: list[int] = []
    out_of_domain: list[int] = []

    for vertex in range(instance.n_vertices):
        if vertex >= len(assignment) or assignment[vertex] is None:
            missing.append(vertex)
            continue
        color = int(assignment[vertex])
        if not 0 <= color < instance.n_colors:
            out_of_domain.append(vertex)

    violated: list[tuple[int, int]] = []
    if not missing and not out_of_domain and len(assignment) == instance.n_vertices:
        for u, v in instance.edges:
            if assignment[u] == assignment[v]:
                violated.append((u, v))

    feasible = (
        len(assignment) == instance.n_vertices
        and not missing
        and not out_of_domain
        and not violated
    )
    return AuditResult(
        feasible=feasible,
        missing_vertices=tuple(missing),
        out_of_domain_vertices=tuple(out_of_domain),
        violated_edges=tuple(violated),
    )
