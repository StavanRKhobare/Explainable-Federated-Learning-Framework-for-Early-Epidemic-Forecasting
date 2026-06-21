"""
dual_task_head.py
─────────────────
Final prediction head.

Takes the fused embedding (local + spatial context) and outputs:
  1. predicted_cases     — regression  (continuous, normalized)
  2. outbreak_prob       — classification  (sigmoid probability)
"""

import torch
import torch.nn as nn


class DualTaskHead(nn.Module):
    """
    Input : fused_emb (N, 2 * embed_dim)  — concat of local + spatial embeddings
    Output: (cases_pred (N,), outbreak_prob (N,))
    """

    def __init__(self, in_dim: int = 128, dropout: float = 0.15):
        super().__init__()

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Regression head — predicts normalised case count
        self.reg_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # Classification head — predicts outbreak probability
        self.clf_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, fused_emb: torch.Tensor):
        h = self.trunk(fused_emb)                          # (N, 64)
        cases_pred    = self.reg_head(h).squeeze(-1)       # (N,)
        outbreak_prob = self.clf_head(h).squeeze(-1)       # (N,)
        return cases_pred, outbreak_prob
