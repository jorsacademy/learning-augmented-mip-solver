from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class GraphColoringInstance:
    """Fixed-k graph-coloring CSP instance."""

    n_vertices: int
    n_colors: int
    edges: tuple[tuple[int, int], ...]
    seed: int
    expected_feasible: bool
    family: str = "planted"

    def __post_init__(self) -> None:
        if self.n_vertices <= 0:
            raise ValueError("n_vertices must be positive")
        if self.n_colors <= 0:
            raise ValueError("n_colors must be positive")
        normalized: set[tuple[int, int]] = set()
        for u, v in self.edges:
            if u == v:
                raise ValueError("self-loops are not allowed")
            if not (0 <= u < self.n_vertices and 0 <= v < self.n_vertices):
                raise ValueError("edge endpoint out of range")
            normalized.add((min(u, v), max(u, v)))
        if len(normalized) != len(self.edges):
            raise ValueError("duplicate or non-normalized edges detected")

    @property
    def degrees(self) -> tuple[int, ...]:
        degree = [0] * self.n_vertices
        for u, v in self.edges:
            degree[u] += 1
            degree[v] += 1
        return tuple(degree)

    @property
    def adjacency(self) -> tuple[tuple[int, ...], ...]:
        adj = [set() for _ in range(self.n_vertices)]
        for u, v in self.edges:
            adj[u].add(v)
            adj[v].add(u)
        return tuple(tuple(sorted(neighbors)) for neighbors in adj)


def generate_instance(
    *,
    n_vertices: int,
    n_colors: int,
    edge_probability: float,
    seed: int,
    infeasible: bool = False,
) -> GraphColoringInstance:
    """Generate a deterministic synthetic coloring instance.

    Feasible instances are planted: edges are sampled only across different planted
    colors. Infeasible instances additionally contain a (k+1)-clique, which is a
    constructive certificate of non-k-colorability.
    """
    if not 0.0 <= edge_probability <= 1.0:
        raise ValueError("edge_probability must lie in [0, 1]")
    if infeasible and n_vertices < n_colors + 1:
        raise ValueError("need at least k+1 vertices to plant an infeasible clique")

    rng = random.Random(seed)
    planted = [i % n_colors for i in range(n_vertices)]
    rng.shuffle(planted)

    edges: set[tuple[int, int]] = set()
    for u in range(n_vertices):
        for v in range(u + 1, n_vertices):
            if planted[u] != planted[v] and rng.random() < edge_probability:
                edges.add((u, v))

    family = "planted-feasible"
    if infeasible:
        clique = list(range(n_colors + 1))
        for i, u in enumerate(clique):
            for v in clique[i + 1 :]:
                edges.add((u, v))
        family = "planted-plus-kplus1-clique"

    return GraphColoringInstance(
        n_vertices=n_vertices,
        n_colors=n_colors,
        edges=tuple(sorted(edges)),
        seed=seed,
        expected_feasible=not infeasible,
        family=family,
    )


def generate_split(
    *,
    count: int,
    n_vertices: int,
    n_colors: int,
    edge_probability: float,
    base_seed: int,
    infeasible_fraction: float = 0.5,
) -> list[GraphColoringInstance]:
    if count <= 0:
        raise ValueError("count must be positive")
    if not 0.0 <= infeasible_fraction <= 1.0:
        raise ValueError("infeasible_fraction must lie in [0, 1]")

    flags = [i < round(count * infeasible_fraction) for i in range(count)]
    random.Random(base_seed + 9_973).shuffle(flags)
    return [
        generate_instance(
            n_vertices=n_vertices,
            n_colors=n_colors,
            edge_probability=edge_probability,
            seed=base_seed + 104_729 * i,
            infeasible=flags[i],
        )
        for i in range(count)
    ]
