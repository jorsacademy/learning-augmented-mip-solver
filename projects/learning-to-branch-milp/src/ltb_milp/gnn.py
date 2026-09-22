"""Pure-PyTorch bipartite graph model for MILP branching."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import Tensor, nn

from ltb_milp.branching import BranchDecision
from ltb_milp.graph import GraphBranchingState, GraphInputs, graph_inputs_from_node
from ltb_milp.problem import BinaryPackingMILP


class BipartiteBranchingGNN(nn.Module):
    """Two-way variable/constraint message passing followed by candidate scoring."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.variable_encoder = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU())
        self.constraint_encoder = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU())
        self.constraint_update = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.variable_update = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: GraphInputs) -> Tensor:
        variable_embeddings = self.variable_encoder(inputs.variable_features)
        constraint_embeddings = self.constraint_encoder(inputs.constraint_features)
        weights = inputs.edge_weights.abs()

        row_degree = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        constraint_messages = weights @ variable_embeddings / row_degree
        constraint_embeddings = constraint_embeddings + self.constraint_update(constraint_messages)

        column_degree = weights.sum(dim=0, keepdim=True).transpose(0, 1).clamp_min(1e-6)
        variable_messages = weights.transpose(0, 1) @ constraint_embeddings / column_degree
        variable_embeddings = variable_embeddings + self.variable_update(variable_messages)

        candidate_embeddings = variable_embeddings[inputs.candidate_indices]
        return self.scorer(candidate_embeddings).squeeze(-1)


def train_gnn_branching_model(
    model: nn.Module,
    states: Iterable[GraphBranchingState],
    *,
    epochs: int = 100,
    learning_rate: float = 1e-3,
) -> list[float]:
    """Train with listwise expert-choice cross entropy over each B&B node."""
    states = list(states)
    if not states:
        raise ValueError("states must not be empty")
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        losses: list[Tensor] = []
        for state in states:
            logits = model(state.inputs).unsqueeze(0)
            target = torch.tensor([state.expert_choice], dtype=torch.long)
            losses.append(torch.nn.functional.cross_entropy(logits, target))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


@torch.inference_mode()
def gnn_top1_expert_agreement(model: nn.Module, states: Iterable[GraphBranchingState]) -> float:
    """Fraction of graph states where the model matches the expert top choice."""
    states = list(states)
    if not states:
        raise ValueError("states must not be empty")
    model.eval()
    correct = 0
    for state in states:
        prediction = int(torch.argmax(model(state.inputs)))
        correct += prediction == state.expert_choice
    return correct / len(states)


def gnn_branch_policy(model: nn.Module):
    """Create an inference policy that does not call strong branching."""

    def policy(
        problem: BinaryPackingMILP,
        x: np.ndarray,
        objective: float,
        bounds: dict[int, tuple[float, float]],
    ) -> BranchDecision:
        del objective, bounds
        inputs = graph_inputs_from_node(problem, x)
        model.eval()
        with torch.inference_mode():
            local_choice = int(torch.argmax(model(inputs)))
        return BranchDecision(int(inputs.candidate_indices[local_choice]))

    return policy
