from __future__ import annotations
import numpy as np
import torch
from .planning import GAPInstance


def instance_to_features(instance: GAPInstance):
    """
    Construct normalized bipartite graph features without using optimum labels.

    Returns:
      machine_features [M,4]
      job_features     [J,4]
      edge_features    [M,J,4]
    """
    c = instance.costs
    r = instance.resources
    cap = instance.capacities
    M, J = c.shape

    cost_mean = c.mean()
    cost_std = c.std() + 1e-6

    relative_resource = r / cap[:, None]
    machine_features = np.column_stack([
        cap / (cap.mean() + 1e-6),
        relative_resource.mean(axis=1),
        relative_resource.max(axis=1),
        (c.mean(axis=1) - cost_mean) / cost_std,
    ])

    min_cost = c.min(axis=0)
    job_features = np.column_stack([
        (min_cost - cost_mean) / cost_std,
        (c.mean(axis=0) - cost_mean) / cost_std,
        relative_resource.min(axis=0),
        relative_resource.mean(axis=0),
    ])

    sorted_cost = np.sort(c, axis=0)
    second = sorted_cost[1] if M > 1 else sorted_cost[0]
    edge_features = np.stack([
        (c - cost_mean) / cost_std,
        relative_resource,
        c / (min_cost[None, :] + 1e-6),
        (c - min_cost[None, :]) / (second[None, :] - min_cost[None, :] + 1e-3),
    ], axis=-1)

    return (
        torch.tensor(machine_features, dtype=torch.float32),
        torch.tensor(job_features, dtype=torch.float32),
        torch.tensor(edge_features, dtype=torch.float32),
    )


def batch_features(instances):
    triples = [instance_to_features(x) for x in instances]
    return tuple(torch.stack([t[k] for t in triples], dim=0) for k in range(3))
