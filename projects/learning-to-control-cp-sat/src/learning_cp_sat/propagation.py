from __future__ import annotations

from collections import deque

from .instances import GraphColoringInstance


def domain_size(mask: int) -> int:
    return mask.bit_count()


def singleton_color(mask: int) -> int:
    if mask <= 0 or mask & (mask - 1):
        raise ValueError("domain is not singleton")
    return mask.bit_length() - 1


def colors_in(mask: int) -> list[int]:
    return [color for color in range(mask.bit_length()) if mask & (1 << color)]


def initial_domains(instance: GraphColoringInstance) -> list[int]:
    full = (1 << instance.n_colors) - 1
    return [full] * instance.n_vertices


def assign_and_propagate(
    instance: GraphColoringInstance,
    domains: list[int],
    variable: int,
    color: int,
) -> tuple[list[int] | None, int, int]:
    """Assign one color and propagate singleton inequality constraints.

    Returns (new_domains, removed_values, propagation_events). A ``None`` domain
    indicates contradiction. The caller's domains are never mutated.
    """
    color_bit = 1 << color
    if not domains[variable] & color_bit:
        return None, 0, 0

    new_domains = domains.copy()
    before_total = sum(domain_size(mask) for mask in new_domains)
    new_domains[variable] = color_bit
    queue: deque[int] = deque([variable])
    propagated: set[int] = set()
    propagation_events = 0

    while queue:
        current = queue.popleft()
        if current in propagated:
            continue
        mask = new_domains[current]
        if domain_size(mask) != 1:
            continue
        propagated.add(current)
        forbidden = mask
        for neighbor in instance.adjacency[current]:
            neighbor_mask = new_domains[neighbor]
            if not neighbor_mask & forbidden:
                continue
            reduced = neighbor_mask & ~forbidden
            propagation_events += 1
            if reduced == 0:
                return None, before_total, propagation_events
            if reduced != neighbor_mask:
                new_domains[neighbor] = reduced
                if domain_size(reduced) == 1:
                    queue.append(neighbor)

    after_total = sum(domain_size(mask) for mask in new_domains)
    return new_domains, before_total - after_total, propagation_events
