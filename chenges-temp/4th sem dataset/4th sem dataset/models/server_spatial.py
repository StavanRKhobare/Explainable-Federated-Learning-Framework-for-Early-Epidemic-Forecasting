"""
server_spatial.py
─────────────────
Server-side Spatial DGAT.

The server receives one 64-dim embedding per district (private — no raw data).
It builds a graph where:
    Nodes  = districts (640+)
    Edges  = shared borders (from graph_edges.csv) — STATIC topology
    Weights = border length normalized — used as edge features

The GAT attention coefficients are LEARNED, so the server discovers
which neighbours are most epidemiologically relevant over training.

Output: spatially refined embedding per district (same dim as input).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class SpatialDGAT(nn.Module):
    """
    Two-layer Spatial GAT on the district border graph.

    Input : node_emb   (N, embed_dim) — local temporal embeddings from clients
            edge_index (2, E)          — border adjacency (static)
            edge_attr  (E, 1)          — normalised border length
    Output: (N, embed_dim)             — spatially refined embeddings
    """

    def __init__(self,
                 embed_dim:    int = 64,
                 hidden_dim:   int = 64,
                 n_heads:      int = 4,
                 dropout:      float = 0.15):
        super().__init__()

        # Edge feature encoder so GAT can use border length
        self.edge_enc = nn.Linear(1, n_heads)   # (E,1) → (E, n_heads) per-head bias

        # Layer 1: embed_dim → hidden_dim  (multi-head, concat)
        self.gat1 = GATConv(
            in_channels=embed_dim,
            out_channels=hidden_dim // n_heads,
            heads=n_heads,
            concat=True,
            dropout=dropout,
            edge_dim=n_heads,          # use encoded border length
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Layer 2: hidden_dim → embed_dim  (single head, no concat)
        self.gat2 = GATConv(
            in_channels=hidden_dim,
            out_channels=embed_dim,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop  = nn.Dropout(dropout)

        # Residual projection (in case embed_dim != hidden_dim)
        self.res_proj = nn.Identity()

    def forward(self,
                node_emb:   torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr:  torch.Tensor) -> torch.Tensor:
        """
        node_emb  : (N, embed_dim)
        edge_index: (2, E)
        edge_attr : (E,)  or (E, 1)  — normalised border lengths
        returns   : (N, embed_dim)
        """
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)    # (E, 1)

        # Encode edge features
        e_feat = self.edge_enc(edge_attr)          # (E, n_heads)

        # GAT Layer 1  (with edge features)
        h = F.elu(self.norm1(self.gat1(node_emb, edge_index, e_feat)))  # (N, hidden)

        # GAT Layer 2  (no edge features — implicit in refined node states)
        h = self.norm2(self.gat2(h, edge_index))   # (N, embed_dim)

        # Residual: add original embedding so local context isn't lost
        return self.drop(h + self.res_proj(node_emb))                   # (N, embed_dim)
