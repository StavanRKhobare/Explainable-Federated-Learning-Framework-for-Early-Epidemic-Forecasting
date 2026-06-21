"""
train_fedxgnn.py  —  Kaggle Training Script
============================================
Split-Federated DGAT for Dengue Epidemic Forecasting
Architecture: Split-FedSTGNN (client GRU+TemporalGAT  →  server SpatialGAT  →  dual-task head)

USAGE (Kaggle):
  1. Upload master_dataset_clean.csv + graph_edges.csv to a Kaggle Dataset
  2. Set INPUT_DIR to /kaggle/input/<your-dataset-name>/
  3. Run all cells / kernel

What gets TRAINED:
  • ClientTemporalModel  — GRU + Temporal GAT per district (local trends + incubation spikes)
  • SpatialDGAT          — learns which neighbouring districts matter most (edge attention weights)
  • DualTaskHead         — regression (cases) + classification (is_outbreak)

What is STATIC (not trained, used as structure):
  • The border graph topology (which districts are connected) — from graph_edges.csv
  • The edge weight values (border lengths) — fed as input features to the Spatial GAT

Targets:
  • cases      → MSE regression loss
  • is_outbreak → BCE classification loss
"""

# ── 0. Installs (Kaggle) ──────────────────────────────────────────────────────
import subprocess, sys, torch

def _pyg_install():
    """
    Install torch-geometric + pre-built scatter/sparse wheels.
    Detects the running PyTorch version and CUDA version automatically
    so we never trigger a slow C++ source build.
    """
    # e.g. "2.1.0" from "2.1.0+cu121"
    torch_ver = torch.__version__.split("+")[0]

    # e.g. "cu121" or "cpu"
    cuda_tag = (
        "cu" + torch.version.cuda.replace(".", "")
        if torch.cuda.is_available() and torch.version.cuda
        else "cpu"
    )

    wheel_url = f"https://data.pyg.org/whl/torch-{torch_ver}+{cuda_tag}.html"
    print(f"[PyG install] Using wheel index: {wheel_url}")

    pkgs = ["torch-geometric", "torch-scatter", "torch-sparse"]
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q",
         *pkgs, "--find-links", wheel_url]
    )
    print("[PyG install] Done.")

try:
    import torch_geometric
except ImportError:
    _pyg_install()
    import torch_geometric   # re-import after install

# ── 1. Imports ────────────────────────────────────────────────────────────────
import os, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── 2. Configuration ──────────────────────────────────────────────────────────
INPUT_DIR = "final_datasets/"          # ← change to your dataset name
OUTPUT_DIR = "outputs/"

CFG = dict(
    data_path   = INPUT_DIR + "master_dataset_clean.csv",
    edges_path  = INPUT_DIR + "graph_edges.csv",
    output_dir  = OUTPUT_DIR,

    # Sliding window
    lookback    = 4,

    # Features
    dynamic_features = [
        "temp_k", "preci_mm", "LAI",
        "cases_lag1", "cases_lag2", "cases_lag3",
        "cases_roll4w", "growth_rate", "cases_per_100k",
        "week_sin", "week_cos", "is_monsoon",
    ],
    static_features = ["population_2024", "pop_density_per_km2_2024"],
    target_reg  = "cases",
    target_clf  = "is_outbreak",

    # Model dims — kept small relative to dataset size (1437 rows)
    # Rule of thumb: params/samples < 0.1 for generalization
    gru_hidden      = 32,    # was 64
    tgat_hidden     = 32,    # was 64
    embed_dim       = 32,    # was 64  → model now ~25K params
    temporal_heads  = 4,
    spatial_heads   = 4,

    # Training
    epochs          = 150,
    lr              = 5e-4,   # kept same as the user was actually at 5e-4 which is 0.0005
    weight_decay    = 1e-3,   # increased weight decay
    alpha           = 0.4,    # 40% reg loss, 60% clf loss (was 0.6)
    train_ratio     = 0.75,
    dropout         = 0.50,   # increased dropout for regularization
    patience        = 30,     # was 25
)

os.makedirs(CFG["output_dir"], exist_ok=True)

# ── 3. Data Loading ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1 — Loading & Preprocessing Data")
print("="*60)

df = pd.read_csv(CFG["data_path"])
print(f"Loaded  : {df.shape[0]} rows × {df.shape[1]} cols")
print(f"Columns : {df.columns.tolist()}")

# Filter for Dengue if Disease column is present
if "Disease" in df.columns:
    df = df[df["Disease"].str.lower() == "dengue"].copy()
    print(f"After Dengue filter: {df.shape[0]} rows")
else:
    print("No 'Disease' column → assuming dataset is Dengue-only")

# Verify essential columns
REQUIRED = ["iso_year", "iso_week", "censuscode", "cases", "is_outbreak"]
for col in REQUIRED:
    assert col in df.columns, f"Missing required column: {col}"

df = df.sort_values(["censuscode", "iso_year", "iso_week"]).reset_index(drop=True)

# ── 3a. Node index mapping ────────────────────────────────────────────────────
unique_codes = sorted(df["censuscode"].unique())
node_to_idx  = {c: i for i, c in enumerate(unique_codes)}
df["node_idx"] = df["censuscode"].map(node_to_idx)
N_NODES = len(unique_codes)
print(f"Districts (nodes): {N_NODES}")

# ── 3b. Time-step index ────────────────────────────────────────────────────────
ts = (df[["iso_year","iso_week"]]
      .drop_duplicates()
      .sort_values(["iso_year","iso_week"])
      .reset_index(drop=True))
ts["t_idx"] = range(len(ts))
df = df.merge(ts, on=["iso_year","iso_week"])
N_TIME = len(ts)
print(f"Unique time steps : {N_TIME}")

# ── 3c. Feature normalisation (STRICTLY NO LEAKAGE) ───────────────────────────
avail_dyn  = [f for f in CFG["dynamic_features"]  if f in df.columns]
avail_stat = [f for f in CFG["static_features"]   if f in df.columns]
miss_dyn   = set(CFG["dynamic_features"]) - set(avail_dyn)
if miss_dyn:
    print(f"WARNING — missing dynamic features (zeroed): {miss_dyn}")

# Calculate train cutoff to prevent future validation data from leaking into scalers
LB = CFG["lookback"]
split_idx = int((N_TIME - LB) * CFG["train_ratio"])
train_time_cutoff = split_idx + LB 
train_mask = df["t_idx"] < train_time_cutoff

scaler_dyn  = StandardScaler()
scaler_stat = StandardScaler()

# FIT ONLY ON TRAIN PERIOD
scaler_dyn.fit(df.loc[train_mask, avail_dyn])
df[avail_dyn] = scaler_dyn.transform(df[avail_dyn])

if avail_stat:
    scaler_stat.fit(df.loc[train_mask, avail_stat])
    df[avail_stat] = scaler_stat.transform(df[avail_stat])

# Normalise regression target (FIT ONLY ON TRAIN PERIOD)
cases_mean = df.loc[train_mask, "cases"].mean()
cases_std  = df.loc[train_mask, "cases"].std() + 1e-8
df["cases_norm"] = (df["cases"] - cases_mean) / cases_std

N_DYN  = len(CFG["dynamic_features"])
N_STAT = len(avail_stat)

# ── 3d. Build dense (T, N, F) tensors ────────────────────────────────────────
X      = torch.zeros(N_TIME, N_NODES, N_DYN,  dtype=torch.float32)
X_stat = torch.zeros(N_NODES, N_STAT,          dtype=torch.float32)
Y_reg  = torch.zeros(N_TIME, N_NODES,          dtype=torch.float32)
Y_clf  = torch.zeros(N_TIME, N_NODES,          dtype=torch.float32)
Obs_mask = torch.zeros(N_TIME, N_NODES,        dtype=torch.bool)
Obs_mask = torch.zeros(N_TIME, N_NODES,        dtype=torch.bool)

for row in df.itertuples(index=False):
    t = row.t_idx
    n = row.node_idx
    for fi, feat in enumerate(CFG["dynamic_features"]):
        if feat in avail_dyn:
            X[t, n, fi] = getattr(row, feat, 0.0)
    Y_reg[t, n] = row.cases_norm
    Y_clf[t, n] = row.is_outbreak
    Obs_mask[t, n] = True
    Obs_mask[t, n] = True

# Static features — take per-node mean across all time steps
if avail_stat:
    for n_val, grp in df.groupby("node_idx"):
        for si, sf in enumerate(avail_stat):
            X_stat[int(n_val), si] = grp[sf].mean()

print(f"\nTensor shapes:")
print(f"  X      : {tuple(X.shape)}   (time, nodes, dyn_features)")
print(f"  X_stat : {tuple(X_stat.shape)}     (nodes, static_features)")
print(f"  Y_reg  : {tuple(Y_reg.shape)}       (time, nodes)")
print(f"  Y_clf  : {tuple(Y_clf.shape)}       (time, nodes)")

# ── 4. Spatial Graph Construction ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2 — Building Spatial Graph")
print("="*60)

edges_df = pd.read_csv(CFG["edges_path"])
print(f"Edges file cols: {edges_df.columns.tolist()}")

# Support both censuscode-based and district-name-based edge files
if "source_censuscode" in edges_df.columns:
    SRC_COL, DST_COL, WT_COL = "source_censuscode", "target_censuscode", "shared_border_km"
elif "district_1" in edges_df.columns:
    # Build a district-name → censuscode → node_idx lookup
    name_to_idx = (df.drop_duplicates("node_idx")
                     .set_index("district")["node_idx"]
                     .to_dict())
    SRC_COL, DST_COL, WT_COL = "district_1", "district_2", "border_length"
else:
    raise ValueError(f"Cannot parse graph_edges.csv. Columns: {edges_df.columns.tolist()}")

ei, ew, skipped = [], [], 0
for row in edges_df.itertuples(index=False):
    src_key = getattr(row, SRC_COL)
    dst_key = getattr(row, DST_COL)
    wt      = getattr(row, WT_COL, 1.0)

    if SRC_COL == "source_censuscode":
        s = node_to_idx.get(src_key)
        d = node_to_idx.get(dst_key)
    else:
        s = name_to_idx.get(src_key)
        d = name_to_idx.get(dst_key)

    if s is None or d is None:
        skipped += 1
        continue

    wt = float(wt) if pd.notna(wt) else 1.0
    ei.extend([[s, d], [d, s]])
    ew.extend([wt, wt])

print(f"Edges loaded : {len(ei)//2} unique  |  skipped: {skipped}")

edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()  # (2, E)
edge_attr  = torch.tensor(ew, dtype=torch.float32)                 # (E,)
# Normalise to [0, 1]
edge_attr  = (edge_attr - edge_attr.min()) / (edge_attr.max() - edge_attr.min() + 1e-8)

print(f"edge_index shape: {tuple(edge_index.shape)}")

# ── 5. Sliding Window Dataset ─────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3 — Creating Sliding Windows  (lookback={})".format(CFG["lookback"]))
print("="*60)

LB = CFG["lookback"]
windows = []
for t in range(N_TIME - LB):
    x_win = X[t:t+LB].permute(1, 0, 2)   # (N, LB, D_dyn)
    y_r   = Y_reg[t + LB]                  # (N,)
    y_c   = Y_clf[t + LB]                  # (N,)
    obs_m = Obs_mask[t + LB]               # (N,)
    windows.append((x_win, y_r, y_c, obs_m))

print(f"Total windows : {len(windows)}")

split     = int(len(windows) * CFG["train_ratio"])
train_win = windows[:split]
val_win   = windows[split:]
print(f"Train : {len(train_win)}  |  Val : {len(val_win)}")

# ── 6. Model Definitions (inline — copy of models/ for single-file Kaggle) ────
print("\n" + "="*60)
print("STEP 4 — Defining Models")
print("="*60)

# ── 6a. Temporal GAT ──────────────────────────────────────────────────────────
class TemporalGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, embed_dim, n_heads=4, T=4, dropout=0.15):
        super().__init__()
        self.T, self.embed_dim = T, embed_dim

        src, dst = [], []
        for i in range(T):
            for j in range(T):
                if i != j:
                    src.append(i); dst.append(j)
        self.register_buffer("te_src", torch.tensor(src, dtype=torch.long))
        self.register_buffer("te_dst", torch.tensor(dst, dtype=torch.long))

        self.gat1 = GATConv(in_dim, hidden_dim // n_heads, heads=n_heads,
                             concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, embed_dim, heads=1,
                             concat=False, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        N, T, D = x.shape
        x_flat = x.reshape(N * T, D)

        offsets = torch.arange(N, device=x.device) * T
        src = (self.te_src.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        dst = (self.te_dst.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        ei  = torch.stack([src, dst], dim=0)

        h = F.elu(self.norm1(self.gat1(x_flat, ei)))
        h = self.norm2(self.gat2(h, ei)).reshape(N, T, self.embed_dim)
        return self.drop((h[:, -1, :] + h.mean(1)) / 2.0)


# ── 6b. Client Temporal Model ─────────────────────────────────────────────────
class ClientTemporalModel(nn.Module):
    def __init__(self, n_dyn, n_stat, gru_h=64, tgat_h=64, embed_dim=64,
                 n_heads=4, T=4, dropout=0.15):
        super().__init__()
        self.gru  = nn.GRU(n_dyn, gru_h, num_layers=2,
                            batch_first=True, dropout=dropout)
        self.tgat = TemporalGAT(n_dyn, tgat_h, embed_dim, n_heads, T, dropout)

        static_out = 16 if n_stat > 0 else 0
        self.static_enc = nn.Sequential(nn.Linear(n_stat, 16), nn.ReLU()) if n_stat > 0 else None

        fusion_in = gru_h + embed_dim + static_out
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
        )

    def forward(self, x_dyn, x_stat=None):
        _, h_n  = self.gru(x_dyn)
        h_gru   = h_n[-1]                              # (N, gru_h)
        h_tgat  = self.tgat(x_dyn)                     # (N, embed_dim)

        parts = [h_gru, h_tgat]
        if self.static_enc is not None and x_stat is not None:
            parts.append(self.static_enc(x_stat))
        return self.fusion(torch.cat(parts, dim=-1))    # (N, embed_dim)


# ── 6c. Spatial DGAT ──────────────────────────────────────────────────────────
class SpatialDGAT(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=64, n_heads=4, dropout=0.15):
        super().__init__()
        self.edge_enc = nn.Linear(1, n_heads)
        self.gat1     = GATConv(embed_dim, hidden_dim // n_heads, heads=n_heads,
                                concat=True, dropout=dropout, edge_dim=n_heads)
        self.norm1    = nn.LayerNorm(hidden_dim)
        self.gat2     = GATConv(hidden_dim, embed_dim, heads=1,
                                concat=False, dropout=dropout)
        self.norm2    = nn.LayerNorm(embed_dim)
        self.drop     = nn.Dropout(dropout)

    def forward(self, node_emb, edge_index, edge_attr):
        ea = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        ef = self.edge_enc(ea)
        h  = F.elu(self.norm1(self.gat1(node_emb, edge_index, ef)))
        h  = self.norm2(self.gat2(h, edge_index))
        return self.drop(h + node_emb)                  # (N, embed_dim)


# ── 6d. Dual-Task Head ────────────────────────────────────────────────────────
class DualTaskHead(nn.Module):
    def __init__(self, in_dim=128, dropout=0.15):
        super().__init__()
        self.trunk    = nn.Sequential(nn.Linear(in_dim, 64), nn.LayerNorm(64),
                                      nn.GELU(), nn.Dropout(dropout))
        self.reg_head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        # No Sigmoid here — BCEWithLogitsLoss takes raw logits (numerically stable)
        self.clf_head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, fused):
        h = self.trunk(fused)
        return self.reg_head(h).squeeze(-1), self.clf_head(h).squeeze(-1)


# ── 6e. Full Pipeline ─────────────────────────────────────────────────────────
class FedXGNN(nn.Module):
    """
    Combines all three components into one trainable pipeline.
    Simulates the Split-Federated forward pass in a single graph.
    """
    def __init__(self, cfg, n_dyn, n_stat):
        super().__init__()
        E = cfg["embed_dim"]
        self.client = ClientTemporalModel(
            n_dyn=n_dyn, n_stat=n_stat,
            gru_h=cfg["gru_hidden"], tgat_h=cfg["tgat_hidden"],
            embed_dim=E, n_heads=cfg["temporal_heads"],
            T=cfg["lookback"], dropout=cfg["dropout"],
        )
        self.server = SpatialDGAT(
            embed_dim=E, hidden_dim=E,
            n_heads=cfg["spatial_heads"], dropout=cfg["dropout"],
        )
        self.head   = DualTaskHead(in_dim=2 * E, dropout=cfg["dropout"])

    def forward(self, x_dyn, x_stat, edge_index, edge_attr):
        # Step 1 — Client: local temporal embedding
        local_emb  = self.client(x_dyn, x_stat)          # (N, E)

        # Step 2 — Server: spatial refinement using neighbor embeddings
        spatial_emb = self.server(local_emb, edge_index, edge_attr)  # (N, E)

        # Step 3 — Head: dual-task prediction
        fused = torch.cat([local_emb, spatial_emb], dim=-1)  # (N, 2E)
        return self.head(fused)                               # cases_pred, outbreak_prob


# ── 7. Training Setup ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5 — Training")
print("="*60)

model = FedXGNN(CFG, N_DYN, N_STAT).to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(),
                               lr=CFG["lr"],
                               weight_decay=CFG["weight_decay"])
# ReduceLROnPlateau: halves LR when val AUPRC stalls for 8 epochs.
# This is more adaptive than CosineAnnealing for irregular validation signals.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-6
)

mse_loss = nn.MSELoss()

# ── Focal Loss: better than pos_weight BCE for rare-event imbalance ─────────
# Focal Loss down-weights easy negatives (the 99% zero rows) and forces the
# model to focus on the hard, rare positive examples (actual outbreaks).
# alpha_fl=0.75 means positive class gets 3x the focal weight.
_n_pos = float(Y_clf[Obs_mask].sum())
_n_neg = float((Y_clf[Obs_mask] == 0).sum())
print(f"Outbreak class balance (observed) → pos={int(_n_pos)} neg={int(_n_neg)} ratio={_n_neg/(_n_pos+1e-8):.1f}:1")
# Using class weight approach to penalise imbalance gently
_pos_weight = torch.tensor([5.0], dtype=torch.float32).to(DEVICE)
clf_loss_fn = nn.BCEWithLogitsLoss(pos_weight=_pos_weight)

# Move graph to device once
edge_index_d = edge_index.to(DEVICE)
edge_attr_d  = edge_attr.to(DEVICE)
X_stat_d     = X_stat.to(DEVICE)


def run_epoch(window_list, train=True):
    model.train(train)
    total_loss = total_reg = total_clf = 0.0
    all_y_reg, all_p_reg = [], []
    all_y_clf, all_p_clf = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x_dyn, y_r, y_c, obs_m in window_list:
            x_dyn = x_dyn.to(DEVICE)
            y_r   = y_r.to(DEVICE)
            y_c   = y_c.to(DEVICE)
            obs_m = obs_m.to(DEVICE)
            obs_m = obs_m.to(DEVICE)
            # Guard: missing lag features (sparse rows) become NaN in the tensor.
            # Replace NaN/Inf with 0 so GRU/GAT activations stay finite.
            x_dyn = torch.nan_to_num(x_dyn, nan=0.0, posinf=0.0, neginf=0.0)

            if train:
                optimizer.zero_grad()

            cases_pred, outbreak_prob = model(x_dyn, X_stat_d,
                                               edge_index_d, edge_attr_d)

            # Only compute loss on districts that actually reported (non-zero target).
            # NOTE: do NOT use feature-based active_nodes — StandardScaler makes
            # week_sin/cos non-zero for every node, collapsing the mask to all-True.
            # Apply mask to only evaluate on observed data, not zero-padded fake nodes
            mask = obs_m
            if mask.sum() == 0:
                continue

            loss_r = mse_loss(cases_pred[mask], y_r[mask])
            # Focal Loss: focuses on hard positive examples, ignores easy negatives
            loss_c = clf_loss_fn(outbreak_prob[mask], y_c[mask])
            loss   = CFG["alpha"] * loss_r + (1 - CFG["alpha"]) * loss_c

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            total_reg  += loss_r.item()
            total_clf  += loss_c.item()

            all_y_reg.extend(y_r[mask].cpu().numpy())
            all_p_reg.extend(cases_pred[mask].detach().cpu().numpy())
            all_y_clf.extend(y_c[mask].cpu().numpy())
            # Convert logits → probabilities only for AUC metric reporting
            all_p_clf.extend(outbreak_prob[mask].detach().sigmoid().cpu().numpy())

    n = len(window_list)

    # ── Regression metrics ────────────────────────────────────────────────────
    mae  = mean_absolute_error(all_y_reg, all_p_reg) if all_y_reg else 0.0
    rmse = float(np.sqrt(np.mean((np.array(all_y_reg) - np.array(all_p_reg))**2))) if all_y_reg else 0.0

    # ── Classification metrics ────────────────────────────────────────────────
    from sklearn.metrics import (average_precision_score,
                                  precision_score, recall_score, f1_score)
    try:
        y_true_arr = np.array(all_y_clf)
        y_prob_arr = np.array(all_p_clf)

        has_both = len(set(all_y_clf)) > 1
        auc   = roc_auc_score(y_true_arr, y_prob_arr) if has_both else 0.5
        auprc = average_precision_score(y_true_arr, y_prob_arr) if has_both else 0.0

        # ── Find the threshold that maximises F1 ──────────────────────────────
        # Focal Loss often suppresses probabilities for extreme minority classes.
        # Max probability might only be 0.05, so we must scan lower than 0.1.
        best_f1, best_thr = 0.0, 0.05
        for thr in np.arange(0.01, 0.6, 0.02):
            y_pred_t = (y_prob_arr >= thr).astype(int)
            f1_t = f1_score(y_true_arr, y_pred_t, zero_division=0)
            if f1_t > best_f1:
                best_f1, best_thr = f1_t, float(thr)

        y_pred_arr = (y_prob_arr >= best_thr).astype(int)
        f1   = best_f1
        prec = precision_score(y_true_arr, y_pred_arr, zero_division=0)
        rec  = recall_score(y_true_arr, y_pred_arr, zero_division=0)
    except Exception:
        auc = auprc = f1 = prec = rec = 0.0
        best_thr = 0.5

    return total_loss / n, total_reg / n, total_clf / n, mae, rmse, auc, auprc, f1, prec, rec


# ── 8. Training Loop ──────────────────────────────────────────────────────────
history = dict(
    train_loss=[], val_loss=[],
    train_mae=[], val_mae=[],
    train_rmse=[], val_rmse=[],
    train_auc=[], val_auc=[],
    train_auprc=[], val_auprc=[],
    train_f1=[], val_f1=[],
)

best_val_auprc = -1.0    # early-stop on val AUPRC
patience_ctr   = 0

for epoch in range(1, CFG["epochs"] + 1):
    tr = run_epoch(train_win, train=True)
    vl = run_epoch(val_win,   train=False)

    tr_loss, tr_reg, tr_clf, tr_mae, tr_rmse, tr_auc, tr_auprc, tr_f1, tr_prec, tr_rec = tr
    vl_loss, vl_reg, vl_clf, vl_mae, vl_rmse, vl_auc, vl_auprc, vl_f1, vl_prec, vl_rec = vl

    scheduler.step(vl_auprc)  # ReduceLROnPlateau: halves LR when val AUPRC stalls

    history["train_loss"].append(tr_loss);  history["val_loss"].append(vl_loss)
    history["train_mae"].append(tr_mae);    history["val_mae"].append(vl_mae)
    history["train_rmse"].append(tr_rmse);  history["val_rmse"].append(vl_rmse)
    history["train_auc"].append(tr_auc);    history["val_auc"].append(vl_auc)
    history["train_auprc"].append(tr_auprc); history["val_auprc"].append(vl_auprc)
    history["train_f1"].append(tr_f1);      history["val_f1"].append(vl_f1)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Ep {epoch:03d} | "
              f"TrLoss {tr_loss:.3f} (Reg {tr_reg:.3f} Clf {tr_clf:.3f}) "
              f"MAE {tr_mae:.3f} AUC {tr_auc:.3f} AUPRC {tr_auprc:.3f} F1 {tr_f1:.3f} | "
              f"Val MAE {vl_mae:.3f} AUC {vl_auc:.3f} AUPRC {vl_auprc:.3f} F1 {vl_f1:.3f} "
              f"Prec {vl_prec:.3f} Rec {vl_rec:.3f}")

    # ── Early stopping on val AUPRC (not val_loss) ────────────────────────────
    # val_loss is dominated by pos_weight-scaled clf loss and is misleading.
    # AUPRC is the right metric: it measures precision-recall trade-off on
    # imbalanced binary outbreak classification.
    if vl_auprc > best_val_auprc:
        best_val_auprc = vl_auprc
        patience_ctr   = 0
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "cfg":         CFG,
            "cases_mean":  cases_mean,
            "cases_std":   cases_std,
            "node_to_idx": node_to_idx,
        }, os.path.join(CFG["output_dir"], "fedxgnn_best.pt"))
        print(f"  ✓ Best AUPRC={best_val_auprc:.4f}  saved.")
    else:
        patience_ctr += 1
        if patience_ctr >= CFG["patience"]:
            print(f"\nEarly stopping at epoch {epoch}")
            break

# ── 9. Save Final Model & Plots ───────────────────────────────────────────────
torch.save(model.state_dict(),
           os.path.join(CFG["output_dir"], "fedxgnn_final.pt"))
print("\nSaved final model.")

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
fig.suptitle("FED-X-GNN Training Curves", fontsize=14)

axes[0,0].plot(history["train_loss"],  label="Train")
axes[0,0].plot(history["val_loss"],    label="Val")
axes[0,0].set_title("Total Loss"); axes[0,0].legend(); axes[0,0].grid(True)

axes[0,1].plot(history["train_mae"],   label="Train")
axes[0,1].plot(history["val_mae"],     label="Val")
axes[0,1].set_title("MAE (Regression, normalised)"); axes[0,1].legend(); axes[0,1].grid(True)

axes[0,2].plot(history["train_rmse"],  label="Train")
axes[0,2].plot(history["val_rmse"],    label="Val")
axes[0,2].set_title("RMSE (Regression, normalised)"); axes[0,2].legend(); axes[0,2].grid(True)

axes[1,0].plot(history["train_auc"],   label="Train")
axes[1,0].plot(history["val_auc"],     label="Val")
axes[1,0].set_title("AUC-ROC (Outbreak)"); axes[1,0].legend(); axes[1,0].grid(True)

axes[1,1].plot(history["train_auprc"], label="Train")
axes[1,1].plot(history["val_auprc"],   label="Val")
axes[1,1].set_title("AUPRC (Outbreak)"); axes[1,1].legend(); axes[1,1].grid(True)

axes[1,2].plot(history["train_f1"],    label="Train")
axes[1,2].plot(history["val_f1"],      label="Val")
axes[1,2].set_title("F1-Score (Outbreak)"); axes[1,2].legend(); axes[1,2].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(CFG["output_dir"], "training_curves.png"), dpi=120)
print("Saved training_curves.png")

# ── 10. Final Summary ─────────────────────────────────────────────────────────
best_vl_auc   = max(history["val_auc"])   if history["val_auc"]   else 0.0
best_vl_auprc = max(history["val_auprc"]) if history["val_auprc"] else 0.0
best_vl_f1    = max(history["val_f1"])    if history["val_f1"]    else 0.0
best_vl_mae   = min(history["val_mae"])   if history["val_mae"]   else 0.0

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"  Best Val AUPRC  : {best_val_auprc:.4f}   ← primary metric (early stopping)")
print(f"  Best Val AUC    : {best_vl_auc:.4f}")
print(f"  Best Val F1     : {best_vl_f1:.4f}")
print(f"  Best Val MAE    : {best_vl_mae:.4f}   (normalised cases)")
print(f"\nSaved files in {CFG['output_dir']}:")
print("  fedxgnn_best.pt       — best AUPRC checkpoint (use for inference)")
print("  fedxgnn_final.pt      — final epoch weights")
print("  training_curves.png   — 6-panel metric curves")
