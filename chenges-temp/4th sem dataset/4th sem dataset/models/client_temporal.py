"""
client_temporal.py
──────────────────
Local (per-district) temporal model.

Two parallel branches:
  1. GRU  → captures smooth 6-month seasonal macro-trends
  2. Temporal GAT → captures sharp 2-3 week biological incubation spikes

Their outputs are fused by an MLP and projected to a 64-dim embedding.
This embedding is the ONLY thing sent to the central server (privacy-safe).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class TemporalGAT(nn.Module):
    """
    Treats each of the T time-steps as a node in a small fully-connected graph.
    GAT attention learns WHICH past week is most relevant (incubation window).

    Input : (N * T, D_feat)   — all districts' time-step nodes packed together
    Output: (N, embed_dim)    — one aggregate vector per district
    """

    def __init__(self, in_dim: int, hidden_dim: int, embed_dim: int,
                 n_heads: int = 4, T: int = 4, dropout: float = 0.1):
        super().__init__()
        self.T = T
        self.embed_dim = embed_dim

        # Fully-connected temporal graph edges (T nodes × T-1 directed edges each)
        src, dst = [], []
        for i in range(T):
            for j in range(T):
                if i != j:
                    src.append(i)
                    dst.append(j)
        self.register_buffer('t_edge_src', torch.tensor(src, dtype=torch.long))
        self.register_buffer('t_edge_dst', torch.tensor(dst, dtype=torch.long))

        # Two GAT layers on the temporal graph
        self.gat1 = GATConv(in_dim, hidden_dim // n_heads,
                             heads=n_heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, embed_dim,
                             heads=1, concat=False, dropout=dropout)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N, T, D_feat)
        returns: (N, embed_dim)
        """
        N, T, D = x.shape

        # Flatten to (N*T, D) and build batch-offset edge indices
        x_flat = x.reshape(N * T, D)          # (N*T, D)

        # Each district gets its own copy of the T-node graph
        # Offset edges per district: district k → edges offset by k*T
        offsets = torch.arange(N, device=x.device) * T   # (N,)
        src = (self.t_edge_src.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        dst = (self.t_edge_dst.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)       # (2, N*(T*(T-1)))

        # GAT forward pass
        h = F.elu(self.norm1(self.gat1(x_flat, edge_index)))
        h = self.norm2(self.gat2(h, edge_index))          # (N*T, embed_dim)
        h = h.reshape(N, T, self.embed_dim)

        # Aggregate time-step representations → single district embedding
        # Use last time-step + mean pooling as complementary signals
        out = (h[:, -1, :] + h.mean(dim=1)) / 2.0        # (N, embed_dim)
        return self.drop(out)


class ClientTemporalModel(nn.Module):
    """
    Full local (client) model for one district.

    Parallel branches:
        GRU(T, D_feat) → h_gru   (gru_hidden)
        TemporalGAT    → h_tgat  (embed_dim)

    Fused by MLP → local_embedding  (embed_dim)
    Non-linear projection preserves privacy (one-way function).
    """

    def __init__(self,
                 n_dynamic_feat: int = 12,
                 n_static_feat:  int = 2,
                 gru_hidden:     int = 64,
                 tgat_hidden:    int = 64,
                 embed_dim:      int = 64,
                 n_heads:        int = 4,
                 T:              int = 4,
                 dropout:        float = 0.1):
        super().__init__()

        self.gru = nn.GRU(
            input_size=n_dynamic_feat,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0.0
        )

        self.tgat = TemporalGAT(
            in_dim=n_dynamic_feat,
            hidden_dim=tgat_hidden,
            embed_dim=embed_dim,
            n_heads=n_heads,
            T=T,
            dropout=dropout
        )

        # Static feature encoder (population, density)
        self.static_enc = nn.Sequential(
            nn.Linear(n_static_feat, 16),
            nn.ReLU(),
        ) if n_static_feat > 0 else None

        static_out = 16 if n_static_feat > 0 else 0
        fusion_in  = gru_hidden + embed_dim + static_out

        # Fusion MLP — non-linear projection for privacy
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self,
                x_dyn:  torch.Tensor,
                x_stat: torch.Tensor | None = None) -> torch.Tensor:
        """
        x_dyn  : (N, T, D_dyn)   — dynamic weekly features
        x_stat : (N, D_stat)     — static node features
        returns: (N, embed_dim)  — local temporal embedding (sent to server)
        """
        # GRU branch
        _, h_n    = self.gru(x_dyn)
        h_gru     = h_n[-1]             # (N, gru_hidden) — last layer's hidden

        # Temporal GAT branch
        h_tgat    = self.tgat(x_dyn)    # (N, embed_dim)

        # Static features
        parts = [h_gru, h_tgat]
        if self.static_enc is not None and x_stat is not None:
            parts.append(self.static_enc(x_stat))

        fused = torch.cat(parts, dim=-1)
        return self.fusion(fused)        # (N, embed_dim)
