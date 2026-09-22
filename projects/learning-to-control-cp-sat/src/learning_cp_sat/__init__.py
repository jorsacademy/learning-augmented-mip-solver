"""Learning-augmented search control for finite-domain CP and CP-SAT."""

from .instances import GraphColoringInstance, generate_instance, generate_split
from .oracle import OracleResult, brute_force_oracle
from .search import SearchResult, solve_coloring

__all__ = [
    "GraphColoringInstance",
    "OracleResult",
    "SearchResult",
    "brute_force_oracle",
    "generate_instance",
    "generate_split",
    "solve_coloring",
]
