"""
run_inference.py
Loads fedxgnn_best.pt and runs inference on the full dataset.
Exports real outbreak probabilities + attention weights to predictions.json
for the dashboard to consume.
"""
import os, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

DEVICE    = torch.device("cpu")
DATA_PATH = "final_datasets/training_dataset_real_weather.csv"
EDGE_PATH = "final_datasets/graph_edges.csv"
MODEL_PT  = "outputs/fedxgnn_best.pt"
OUT_JSON  = "predictions.json"

CFG = dict(
    lookback        = 4,
    dynamic_features= ["temp_k","preci_mm","LAI","cases_lag1","cases_lag2","cases_lag3","week_sin","week_cos","is_monsoon"],
    static_features = ["population_2024","pop_density_per_km2_2024"],
    target_reg      = "cases",
    target_clf      = "is_outbreak",
    gru_hidden      = 32,
    tgat_hidden     = 32,
    embed_dim       = 32,
    temporal_heads  = 4,
    spatial_heads   = 4,
    train_ratio     = 0.75,
    dropout         = 0.0,   # eval mode: dropout off
)

# ── MODEL DEFINITIONS ──────────────────────────────────────────────────────────
class TemporalGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, embed_dim, n_heads=4, T=4, dropout=0.0):
        super().__init__()
        self.T, self.embed_dim = T, embed_dim
        src, dst = [], []
        for i in range(T):
            for j in range(T):
                if i != j:
                    src.append(i); dst.append(j)
        self.register_buffer("te_src", torch.tensor(src, dtype=torch.long))
        self.register_buffer("te_dst", torch.tensor(dst, dtype=torch.long))
        self.gat1  = GATConv(in_dim,    hidden_dim // n_heads, heads=n_heads, concat=True,  dropout=dropout)
        self.gat2  = GATConv(hidden_dim, embed_dim,            heads=1,       concat=False, dropout=dropout)
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


class ClientTemporalModel(nn.Module):
    def __init__(self, n_dyn, n_stat, gru_h=32, tgat_h=32, embed_dim=32, n_heads=4, T=4, dropout=0.0):
        super().__init__()
        self.gru  = nn.GRU(n_dyn, gru_h, num_layers=2, batch_first=True, dropout=0.0)
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
        h_gru   = h_n[-1]
        h_tgat  = self.tgat(x_dyn)
        parts   = [h_gru, h_tgat]
        if self.static_enc is not None and x_stat is not None:
            parts.append(self.static_enc(x_stat))
        return self.fusion(torch.cat(parts, dim=-1))


class SpatialDGAT(nn.Module):
    def __init__(self, embed_dim=32, hidden_dim=32, n_heads=4, dropout=0.0):
        super().__init__()
        self.edge_enc = nn.Linear(1, n_heads)
        self.gat1     = GATConv(embed_dim, hidden_dim // n_heads, heads=n_heads, concat=True, dropout=dropout, edge_dim=n_heads)
        self.norm1    = nn.LayerNorm(hidden_dim)
        self.gat2     = GATConv(hidden_dim, embed_dim, heads=1, concat=False, dropout=dropout)
        self.norm2    = nn.LayerNorm(embed_dim)
        self.drop     = nn.Dropout(dropout)

    def forward(self, node_emb, edge_index, edge_attr):
        ea = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        ef = self.edge_enc(ea)
        h  = F.elu(self.norm1(self.gat1(node_emb, edge_index, ef)))
        h  = self.norm2(self.gat2(h, edge_index))
        return self.drop(h + node_emb)


class DualTaskHead(nn.Module):
    def __init__(self, in_dim=64, dropout=0.0):
        super().__init__()
        self.trunk    = nn.Sequential(nn.Linear(in_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout))
        self.reg_head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        self.clf_head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, fused):
        h = self.trunk(fused)
        return self.reg_head(h).squeeze(-1), self.clf_head(h).squeeze(-1)


class FedXGNN(nn.Module):
    def __init__(self, cfg, n_dyn, n_stat):
        super().__init__()
        E = cfg["embed_dim"]
        self.client = ClientTemporalModel(n_dyn=n_dyn, n_stat=n_stat, gru_h=cfg["gru_hidden"],
                                          tgat_h=cfg["tgat_hidden"], embed_dim=E,
                                          n_heads=cfg["temporal_heads"], T=cfg["lookback"], dropout=cfg["dropout"])
        self.server = SpatialDGAT(embed_dim=E, hidden_dim=E, n_heads=cfg["spatial_heads"], dropout=cfg["dropout"])
        self.head   = DualTaskHead(in_dim=2 * E, dropout=cfg["dropout"])

    def forward(self, x_dyn, x_stat, edge_index, edge_attr):
        local_emb   = self.client(x_dyn, x_stat)
        spatial_emb = self.server(local_emb, edge_index, edge_attr)
        fused       = torch.cat([local_emb, spatial_emb], dim=-1)
        return self.head(fused)


# ── DATA LOADING ───────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df = df.sort_values(["censuscode","iso_year","iso_week"]).reset_index(drop=True)

unique_codes = sorted(df["censuscode"].unique())
node_to_idx  = {c: i for i, c in enumerate(unique_codes)}
idx_to_code  = {i: c for c, i in node_to_idx.items()}
df["node_idx"] = df["censuscode"].map(node_to_idx)
N_NODES = len(unique_codes)

ts = df[["iso_year","iso_week"]].drop_duplicates().sort_values(["iso_year","iso_week"]).reset_index(drop=True)
ts["t_idx"] = range(len(ts))
df = df.merge(ts, on=["iso_year","iso_week"])
N_TIME = len(ts)

avail_dyn  = [f for f in CFG["dynamic_features"] if f in df.columns]
avail_stat = [f for f in CFG["static_features"]  if f in df.columns]

LB         = CFG["lookback"]
split_idx  = int((N_TIME - LB) * CFG["train_ratio"])
train_cut  = split_idx + LB
train_mask = df["t_idx"] < train_cut

scaler_dyn  = StandardScaler()
scaler_stat = StandardScaler()
scaler_dyn.fit(df.loc[train_mask, avail_dyn])
df[avail_dyn] = scaler_dyn.transform(df[avail_dyn])
if avail_stat:
    scaler_stat.fit(df.loc[train_mask, avail_stat])
    df[avail_stat] = scaler_stat.transform(df[avail_stat])

cases_mean = df.loc[train_mask, "cases"].mean()
cases_std  = df.loc[train_mask, "cases"].std() + 1e-8
df["cases_norm"] = (df["cases"] - cases_mean) / cases_std

N_DYN  = len(CFG["dynamic_features"])
N_STAT = len(avail_stat)

X        = torch.zeros(N_TIME, N_NODES, N_DYN,  dtype=torch.float32)
X_stat   = torch.zeros(N_NODES, N_STAT,          dtype=torch.float32)
Y_clf    = torch.zeros(N_TIME, N_NODES,           dtype=torch.float32)
Obs_mask = torch.zeros(N_TIME, N_NODES,           dtype=torch.bool)

for row in df.itertuples(index=False):
    t = row.t_idx; n = row.node_idx
    for fi, feat in enumerate(CFG["dynamic_features"]):
        if feat in avail_dyn:
            X[t, n, fi] = getattr(row, feat, 0.0)
    Y_clf[t, n]    = row.is_outbreak
    Obs_mask[t, n] = True

if avail_stat:
    for n_val, grp in df.groupby("node_idx"):
        for si, sf in enumerate(avail_stat):
            X_stat[int(n_val), si] = grp[sf].mean()

# Graph
edges_df = pd.read_csv(EDGE_PATH)
ei, ew = [], []
for row in edges_df.itertuples(index=False):
    s = node_to_idx.get(row.source_censuscode)
    d = node_to_idx.get(row.target_censuscode)
    if s is None or d is None: continue
    wt = float(row.shared_border_km) if pd.notna(row.shared_border_km) else 1.0
    ei.extend([[s, d], [d, s]])
    ew.extend([wt, wt])

edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
edge_attr  = torch.tensor(ew, dtype=torch.float32)
edge_attr  = (edge_attr - edge_attr.min()) / (edge_attr.max() - edge_attr.min() + 1e-8)

# Sliding windows
windows = []
for t in range(N_TIME - LB):
    x_win = X[t:t+LB].permute(1, 0, 2)
    y_c   = Y_clf[t + LB]
    obs_m = Obs_mask[t + LB]
    t_target = t + LB
    windows.append((x_win, y_c, obs_m, t_target))

print(f"Total windows: {len(windows)} | Nodes: {N_NODES}")

# ── LOAD MODEL ─────────────────────────────────────────────────────────────────
print("Loading model checkpoint...")
model = FedXGNN(CFG, N_DYN, N_STAT).to(DEVICE)
ckpt  = torch.load(MODEL_PT, map_location=DEVICE, weights_only=False)
# The checkpoint might be the whole model or just state_dict
if isinstance(ckpt, dict) and "model_state" in ckpt:
    model.load_state_dict(ckpt["model_state"])
elif isinstance(ckpt, dict) and any(k.startswith("client") for k in ckpt.keys()):
    model.load_state_dict(ckpt)
else:
    # It's the full model object
    model = ckpt
model.eval()
print("Model loaded.")

# ── INFERENCE ──────────────────────────────────────────────────────────────────
print("Running inference on all windows...")
# Store predictions: {t_idx: {node_idx: prob}}
all_probs = {}  # t_idx -> np array (N_NODES,)
all_true  = {}

X_stat_d   = X_stat.to(DEVICE)
edge_idx_d = edge_index.to(DEVICE)
edge_att_d = edge_attr.to(DEVICE)

with torch.no_grad():
    for x_win, y_c, obs_m, t_idx in windows:
        x_d = torch.nan_to_num(x_win.to(DEVICE), nan=0.0)
        _, outbreak_logit = model(x_d, X_stat_d, edge_idx_d, edge_att_d)
        probs = outbreak_logit.sigmoid().cpu().numpy()
        all_probs[t_idx] = probs
        all_true[t_idx]  = y_c.numpy()

print(f"Inference complete. {len(all_probs)} time steps covered.")

# ── BUILD OUTPUT ───────────────────────────────────────────────────────────────
# Per-window records: list of {year, week, predictions: [{censuscode, prob, true_label}]}
timeline = []
for t_idx in sorted(all_probs.keys()):
    row_ts = ts.iloc[t_idx]
    probs  = all_probs[t_idx]
    truths = all_true[t_idx]
    preds  = []
    for n_idx in range(N_NODES):
        code = idx_to_code[n_idx]
        preds.append({
            "censuscode": int(code),
            "prob": round(float(probs[n_idx]), 4),
            "true": int(truths[n_idx]),
        })
    timeline.append({
        "t_idx": int(t_idx),
        "year":  int(row_ts.iso_year),
        "week":  int(row_ts.iso_week),
        "predictions": preds,
    })

# Also build per-district summary
district_info = df[["censuscode","district","state","lat","lon"]].drop_duplicates("censuscode").to_dict(orient="records")

# Neighbors from graph edges
neighbors = {}
for row in edges_df.itertuples(index=False):
    s, d = int(row.source_censuscode), int(row.target_censuscode)
    bdr  = float(row.shared_border_km) if pd.notna(row.shared_border_km) else 1.0
    if s not in neighbors: neighbors[s] = []
    if d not in neighbors: neighbors[d] = []
    if len(neighbors[s]) < 8: neighbors[s].append({"code": d, "border": round(bdr,1)})
    if len(neighbors[d]) < 8: neighbors[d].append({"code": s, "border": round(bdr,1)})

for d in district_info:
    d["neighbors"] = neighbors.get(d["censuscode"], [])

out = {
    "districts": district_info,
    "timeline":  timeline,
    "val_start_t": int(split_idx + LB),
    "n_time": int(N_TIME),
    "generated": pd.Timestamp.now().isoformat(),
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out, f)

size_mb = os.path.getsize(OUT_JSON) / 1e6
print(f"Saved {OUT_JSON} ({size_mb:.1f} MB)")
print(f"Timeline entries: {len(timeline)}")
print(f"Val window start: t={split_idx + LB}")

# Quick sanity check
probs_all = [p["prob"] for t in timeline for p in t["predictions"]]
true_all  = [p["true"] for t in timeline for p in t["predictions"]]
print(f"Prob range: {min(probs_all):.3f} – {max(probs_all):.3f}")
print(f"Positive rate: {sum(true_all)/len(true_all)*100:.2f}%")
