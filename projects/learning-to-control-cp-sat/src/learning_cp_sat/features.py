from __future__ import annotations

import numpy as np

from .instances import GraphColoringInstance
from .propagation import domain_size, singleton_color

FEATURE_NAMES = (
    "domain_fraction",
    "degree_fraction",
    "remaining_degree_fraction",
    "singleton_neighbor_fraction",
    "saturation_fraction",
    "mean_neighbor_domain_fraction",
    "min_neighbor_domain_fraction",
    "depth_fraction",
)


def candidate_features(
    instance: GraphColoringInstance,
    domains: list[int],
    variable: int,
    depth: int,
) -> np.ndarray:
    neighbors = instance.adjacency[variable]
    degree = len(neighbors)
    neighbor_sizes = [domain_size(domains[u]) for u in neighbors]
    singleton_neighbors = [u for u in neighbors if domain_size(domains[u]) == 1]
    distinct_singletons = {singleton_color(domains[u]) for u in singleton_neighbors}
    remaining_degree = sum(domain_size(domains[u]) > 1 for u in neighbors)

    denom_vertices = max(1, instance.n_vertices - 1)
    denom_degree = max(1, degree)
    mean_neighbor_domain = (
        sum(neighbor_sizes) / len(neighbor_sizes) if neighbor_sizes else instance.n_colors
    )
    min_neighbor_domain = min(neighbor_sizes, default=instance.n_colors)

    return np.asarray(
        [
            domain_size(domains[variable]) / instance.n_colors,
            degree / denom_vertices,
            remaining_degree / denom_degree,
            len(singleton_neighbors) / denom_degree,
            len(distinct_singletons) / instance.n_colors,
            mean_neighbor_domain / instance.n_colors,
            min_neighbor_domain / instance.n_colors,
            depth / max(1, instance.n_vertices),
        ],
        dtype=np.float64,
    )
