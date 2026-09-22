from __future__ import annotations
import torch
from torch import nn


class BipartiteGNN(nn.Module):
    """
    Pure-PyTorch bipartite message passing for GAP.

    No PyTorch-Geometric dependency: each layer builds edge messages from the
    current machine/job embeddings and aggregates them back to both partitions.
    """

    def __init__(
        self,
        machine_dim: int = 4,
        job_dim: int = 4,
        edge_dim: int = 4,
        hidden_dim: int = 48,
        layers: int = 3,
    ):
        super().__init__()
        self.machine_embed = nn.Linear(machine_dim, hidden_dim)
        self.job_embed = nn.Linear(job_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)

        self.edge_mlps = nn.ModuleList()
        self.machine_mlps = nn.ModuleList()
        self.job_mlps = nn.ModuleList()
        self.machine_norms = nn.ModuleList()
        self.job_norms = nn.ModuleList()

        for _ in range(layers):
            self.edge_mlps.append(nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ))
            self.machine_mlps.append(nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
            self.job_mlps.append(nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
            self.machine_norms.append(nn.LayerNorm(hidden_dim))
            self.job_norms.append(nn.LayerNorm(hidden_dim))

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, machine_features, job_features, edge_features):
        """
        machine_features [B,M,Fm]
        job_features     [B,J,Fj]
        edge_features    [B,M,J,Fe]
        returns logits   [B,M,J]
        """
        m = self.machine_embed(machine_features)
        j = self.job_embed(job_features)
        e0 = self.edge_embed(edge_features)

        for edge_mlp, machine_mlp, job_mlp, mn, jn in zip(
            self.edge_mlps,
            self.machine_mlps,
            self.job_mlps,
            self.machine_norms,
            self.job_norms,
        ):
            B, M, H = m.shape
            J = j.shape[1]
            me = m[:, :, None, :].expand(B, M, J, H)
            je = j[:, None, :, :].expand(B, M, J, H)
            edge_message = edge_mlp(torch.cat([me, je, e0], dim=-1))

            m_agg = edge_message.mean(dim=2)
            j_agg = edge_message.mean(dim=1)
            m = mn(m + machine_mlp(torch.cat([m, m_agg], dim=-1)))
            j = jn(j + job_mlp(torch.cat([j, j_agg], dim=-1)))

        B, M, H = m.shape
        J = j.shape[1]
        me = m[:, :, None, :].expand(B, M, J, H)
        je = j[:, None, :, :].expand(B, M, J, H)
        score_input = torch.cat([me, je, e0], dim=-1)
        return self.scorer(score_input).squeeze(-1)
