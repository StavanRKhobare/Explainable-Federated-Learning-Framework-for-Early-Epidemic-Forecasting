"""
FastAPI backend for FedXGNN Epidemic Intelligence Dashboard.
Loads the trained model once on startup and exposes endpoints for:
  1. Spatial graph visualization
  2. Split-federated learning step-by-step demo
  3. Live model inference
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# ── Paths (relative to project root) ──────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "training_dataset_enhanced_v2.csv")
EDGE_PATH    = os.path.join(PROJECT_ROOT, "data", "graph", "graph_edges.csv")
MODEL_PT     = os.path.join(PROJECT_ROOT, "model", "fedxgnn_best.pt")

DEVICE = torch.device("cpu")

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
    dropout         = 0.5,  # Phase 3 model trained with dropout=0.5
    target_log      = True, # Phase 3 model trained on log(1+y)
)

# ── MODEL DEFINITIONS (same as run_inference.py) ─────────────────────────────
class TemporalGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, embed_dim, n_heads=4, T=4, dropout=0.0):
        super().__init__()
        self.T, self.embed_dim = T, embed_dim
        src, dst = [], []
        for i in range(T):
            for j in range(T):
                if i != j: src.append(i); dst.append(j)
        self.register_buffer("te_src", torch.tensor(src, dtype=torch.long))
        self.register_buffer("te_dst", torch.tensor(dst, dtype=torch.long))
        self.gat1  = GATConv(in_dim, hidden_dim // n_heads, heads=n_heads, concat=True, dropout=dropout)
        self.gat2  = GATConv(hidden_dim, embed_dim, heads=1, concat=False, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        N, T, D = x.shape
        x_flat = x.reshape(N * T, D)
        offsets = torch.arange(N, device=x.device) * T
        src = (self.te_src.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        dst = (self.te_dst.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        ei = torch.stack([src, dst], dim=0)
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
        _, h_n = self.gru(x_dyn)
        h_gru = h_n[-1]
        h_tgat = self.tgat(x_dyn)
        parts = [h_gru, h_tgat]
        if self.static_enc is not None and x_stat is not None:
            parts.append(self.static_enc(x_stat))
        return self.fusion(torch.cat(parts, dim=-1))

class SpatialDGAT(nn.Module):
    def __init__(self, embed_dim=32, hidden_dim=32, n_heads=4, dropout=0.0):
        super().__init__()
        self.edge_enc = nn.Linear(1, n_heads)
        self.gat1 = GATConv(embed_dim, hidden_dim // n_heads, heads=n_heads, concat=True, dropout=dropout, edge_dim=n_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.gat2 = GATConv(hidden_dim, embed_dim, heads=1, concat=False, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, node_emb, edge_index, edge_attr):
        ea = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        ef = self.edge_enc(ea)
        h = F.elu(self.norm1(self.gat1(node_emb, edge_index, ef)))
        h = self.norm2(self.gat2(h, edge_index))
        return self.drop(h + node_emb)

class DualTaskHead(nn.Module):
    def __init__(self, in_dim=64, dropout=0.0):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(in_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout))
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
        self.head = DualTaskHead(in_dim=2 * E, dropout=cfg["dropout"])

    def forward(self, x_dyn, x_stat, edge_index, edge_attr):
        local_emb = self.client(x_dyn, x_stat)
        spatial_emb = self.server(local_emb, edge_index, edge_attr)
        fused = torch.cat([local_emb, spatial_emb], dim=-1)
        return self.head(fused)

# ── GLOBAL STATE ──────────────────────────────────────────────────────────────
model = None
model_params = {}

X = None          # (N_TIME, N_NODES, N_DYN)
X_stat = None     # (N_NODES, N_STAT)
Y_clf = None
Obs_mask = None
Y_reg = None      # (N_TIME, N_NODES) Actual cases (raw)
edge_index = None
edge_attr = None
windows = None
ts = None         # time step df
unique_codes = None
node_to_idx = None
idx_to_code = None
district_info = None   # list of dicts with name/state/lat/lon
edges_raw = None       # raw edges df
N_NODES = 0
N_TIME = 0
N_DYN = 0
N_STAT = 0
LB = 4
scaler_dyn = None

def load_everything():
    global model, model_params
    global X, X_stat, Y_clf, Obs_mask, edge_index, edge_attr, scaler_dyn
    global windows, ts, unique_codes, node_to_idx, idx_to_code
    global district_info, edges_raw, N_NODES, N_TIME, N_DYN, N_STAT, LB

    print("[BOOT] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values(["censuscode","iso_year","iso_week"]).reset_index(drop=True)

    unique_codes = sorted(df["censuscode"].unique())
    node_to_idx = {c: i for i, c in enumerate(unique_codes)}
    idx_to_code = {i: c for c, i in node_to_idx.items()}
    df["node_idx"] = df["censuscode"].map(node_to_idx)
    N_NODES = len(unique_codes)

    ts_df = df[["iso_year","iso_week"]].drop_duplicates().sort_values(["iso_year","iso_week"]).reset_index(drop=True)
    ts_df["t_idx"] = range(len(ts_df))
    df = df.merge(ts_df, on=["iso_year","iso_week"])
    N_TIME = len(ts_df)
    ts = ts_df

    avail_dyn  = [f for f in CFG["dynamic_features"] if f in df.columns]
    avail_stat = [f for f in CFG["static_features"]  if f in df.columns]

    LB = CFG["lookback"]
    split_idx = int((N_TIME - LB) * CFG["train_ratio"])
    train_cut = split_idx + LB
    train_mask = df["t_idx"] < train_cut

    # Apply log1p to ALL case-related columns to match the training pipeline exactly.
    # The model/fedxgnn_best.pt (epoch 193) was trained after log1p was applied to
    # cases, cases_lag1, cases_lag2, cases_lag3 BEFORE StandardScaler.
    log_cols = [c for c in df.columns if "cases" in c.lower()]
    for c in log_cols:
        df[c] = np.log1p(df[c])

    scaler_dyn = StandardScaler()
    scaler_stat = StandardScaler()
    scaler_dyn.fit(df.loc[train_mask, avail_dyn])
    df[avail_dyn] = scaler_dyn.transform(df[avail_dyn])
    if avail_stat:
        scaler_stat.fit(df.loc[train_mask, avail_stat])
        df[avail_stat] = scaler_stat.transform(df[avail_stat])

    N_DYN = len(CFG["dynamic_features"])
    N_STAT = len(avail_stat)

    X = torch.zeros(N_TIME, N_NODES, N_DYN, dtype=torch.float32)
    X_stat = torch.zeros(N_NODES, N_STAT, dtype=torch.float32)
    Y_clf = torch.zeros(N_TIME, N_NODES, dtype=torch.float32)
    Y_reg = torch.zeros(N_TIME, N_NODES, dtype=torch.float32)
    Obs_mask = torch.zeros(N_TIME, N_NODES, dtype=torch.bool)

    for row in df.itertuples(index=False):
        t = row.t_idx; n = row.node_idx
        for fi, feat in enumerate(CFG["dynamic_features"]):
            if feat in avail_dyn:
                X[t, n, fi] = getattr(row, feat, 0.0)
        Y_clf[t, n] = row.is_outbreak
        # Y_reg stores the log1p(cases) value since that's what's in df now
        # We'll expm1 it when displaying to show real counts
        Y_reg[t, n] = getattr(row, "cases", 0.0)
        Obs_mask[t, n] = True

    if avail_stat:
        for n_val, grp in df.groupby("node_idx"):
            for si, sf in enumerate(avail_stat):
                X_stat[int(n_val), si] = grp[sf].mean()

    # District info
    district_info = df[["censuscode","district","state","lat","lon"]].drop_duplicates("censuscode").to_dict(orient="records")

    # Graph edges
    edges_raw = pd.read_csv(EDGE_PATH)
    ei_list, ew_list = [], []
    for row in edges_raw.itertuples(index=False):
        s = node_to_idx.get(row.source_censuscode)
        d = node_to_idx.get(row.target_censuscode)
        if s is None or d is None: continue
        wt = float(row.shared_border_km) if pd.notna(row.shared_border_km) else 1.0
        ei_list.extend([[s, d], [d, s]])
        ew_list.extend([wt, wt])

    edge_index = torch.tensor(ei_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(ew_list, dtype=torch.float32)
    edge_attr_norm = (edge_attr - edge_attr.min()) / (edge_attr.max() - edge_attr.min() + 1e-8)

    # Sliding windows
    windows = []
    for t in range(N_TIME - LB):
        x_win = X[t:t+LB].permute(1, 0, 2)
        y_c = Y_clf[t + LB]
        y_r = Y_reg[t + LB]
        obs_m = Obs_mask[t + LB]
        windows.append((x_win, y_c, y_r, obs_m, t + LB))

    # Load single model
    print("[BOOT] Loading model checkpoint...")
    ckpt = torch.load(MODEL_PT, map_location=DEVICE, weights_only=False)
    m_cfg = CFG.copy()
    if isinstance(ckpt, dict) and "cfg" in ckpt:
        ckpt_cfg = ckpt["cfg"]
        for key in ["gru_hidden","tgat_hidden","embed_dim","temporal_heads","spatial_heads","dropout","lookback","train_ratio","target_log"]:
            if key in ckpt_cfg: m_cfg[key] = ckpt_cfg[key]
    m_cfg["dropout"] = 0.0

    mdl = FedXGNN(m_cfg, N_DYN, N_STAT).to(DEVICE)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        mdl.load_state_dict(ckpt["model_state"])
    elif isinstance(ckpt, dict) and any(k.startswith("client") for k in ckpt.keys()):
        mdl.load_state_dict(ckpt)
    else:
        mdl = ckpt
    mdl.eval()
    model = mdl
    model_params = {
        "cfg": m_cfg,
        "is_target_log": m_cfg.get("target_log", False),
        "cases_mean": ckpt.get("cases_mean", 0.0) if isinstance(ckpt, dict) else 0.0,
        "cases_std": ckpt.get("cases_std", 1.0) if isinstance(ckpt, dict) else 1.0,
    }
    print(f"[BOOT] Model loaded — epoch {ckpt.get('epoch','?')}, {sum(p.numel() for p in mdl.parameters())} params, target_log={m_cfg.get('target_log')}")

    # Store normalized edge_attr globally (topology is shared)
    globals()["edge_attr"] = edge_attr_norm
    globals()["edge_attr_raw"] = torch.tensor(ew_list, dtype=torch.float32)

    print(f"[BOOT] Ready — {N_NODES} districts, {N_TIME} timesteps, {len(windows)} windows")


# ── Run inference for a single window ─────────────────────────────────────────
def run_window(t_win_idx):
    """Run inference on window index (0-based into windows list)."""
    if t_win_idx < 0 or t_win_idx >= len(windows):
        return None
    
    x_win, y_c, y_r, obs_m, t_target = windows[t_win_idx]
    
    with torch.no_grad():
        x_d = torch.nan_to_num(x_win.to(DEVICE), nan=0.0)
        cases_pred, logit = model(x_d, X_stat.to(DEVICE), edge_index.to(DEVICE), edge_attr.to(DEVICE))
        probs = logit.sigmoid().cpu().numpy()
        preds_r_norm = cases_pred.cpu().numpy()
        
        # Inverse transform: un-normalize then un-log
        c_mean = model_params["cases_mean"]
        c_std  = model_params["cases_std"]
        preds_r = preds_r_norm * c_std + c_mean
        if model_params["is_target_log"]:
            preds_r = np.expm1(preds_r)
        preds_r = np.maximum(preds_r, 0.0)
    
    # Y_reg is stored as log1p(cases) — convert back to raw for display
    actual_cases = np.expm1(y_r.numpy())
    return probs, preds_r, y_c.numpy(), actual_cases, t_target


def run_federated_demo(t_win_idx, district_indices):
    """Run inference step-by-step, capturing intermediate tensors for the demo."""
    if t_win_idx < 0 or t_win_idx >= len(windows):
        return None
    
    x_win, y_c, y_r, obs_m, t_target = windows[t_win_idx]
    x_d = torch.nan_to_num(x_win.to(DEVICE), nan=0.0)
    x_s = X_stat.to(DEVICE)
    ei = edge_index.to(DEVICE)
    ea = edge_attr.to(DEVICE)

    with torch.no_grad():
        _, h_n = model.client.gru(x_d)
        h_gru = h_n[-1]
        h_tgat = model.client.tgat(x_d)
        parts = [h_gru, h_tgat]
        if model.client.static_enc is not None:
            parts.append(model.client.static_enc(x_s))
        client_emb = model.client.fusion(torch.cat(parts, dim=-1))
        spatial_emb = model.server(client_emb, ei, ea)
        fused = torch.cat([client_emb, spatial_emb], dim=-1)
        cases_pred_norm, logit = model.head(fused)
        probs = logit.sigmoid()
        
        c_mean = model_params["cases_mean"]
        c_std  = model_params["cases_std"]
        cases_pred = cases_pred_norm.cpu().numpy() * c_std + c_mean
        if model_params["is_target_log"]:
            cases_pred = np.expm1(cases_pred)
        cases_pred = np.maximum(cases_pred, 0.0)

    # Y_reg stored as log1p — convert back
    actual_cases = np.expm1(y_r.numpy())

    results = []
    for n_idx in district_indices:
        code = idx_to_code[n_idx]
        info = next((d for d in district_info if d["censuscode"] == code), {})
        raw_feats = {}
        feat_names = CFG["dynamic_features"]
        raw_feats_array = scaler_dyn.inverse_transform([x_d[n_idx, -1].cpu().numpy()])[0]
        for fi, fn in enumerate(feat_names):
            raw_feats[fn] = round(float(raw_feats_array[fi]), 4)

        results.append({
            "node_idx": n_idx,
            "censuscode": int(code),
            "district": info.get("district", "Unknown"),
            "state": info.get("state", "Unknown"),
            "raw_features": raw_feats,
            "gru_output": [round(float(v), 4) for v in h_gru[n_idx].tolist()[:8]],
            "tgat_output": [round(float(v), 4) for v in h_tgat[n_idx].tolist()[:8]],
            "client_embedding": [round(float(v), 4) for v in client_emb[n_idx].tolist()[:8]],
            "spatial_embedding": [round(float(v), 4) for v in spatial_emb[n_idx].tolist()[:8]],
            "outbreak_prob": round(float(probs[n_idx]), 4),
            "cases_pred": round(float(cases_pred[n_idx]), 4),
            "actual_cases": float(actual_cases[n_idx]),
            "ground_truth": int(y_c[n_idx]),
        })
    return results


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="FedXGNN Epidemic Intelligence API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def startup():
    load_everything()


@app.get("/api/districts")
def get_districts():
    """Return all district metadata."""
    return district_info


@app.get("/api/graph")
def get_graph(t: int = Query(default=-1, description="Window index, -1 for last")):
    """Return spatial graph with nodes, edges, and predictions."""
    t_idx = t if t >= 0 else len(windows) - 1
    result = run_window(t_idx)
    if result is None:
        return {"error": "Invalid time window"}
    probs, preds_r, truths_c, truths_r, t_target = result
    row_ts = ts.iloc[t_target]

    nodes = []
    for n_idx in range(N_NODES):
        code = idx_to_code[n_idx]
        info = next((d for d in district_info if d["censuscode"] == code), {})
        nodes.append({
            "id": n_idx,
            "censuscode": int(code),
            "name": info.get("district", ""),
            "state": info.get("state", ""),
            "lat": float(info.get("lat", 0)),
            "lon": float(info.get("lon", 0)),
            "prob": round(float(probs[n_idx]), 4),
            "pred_cases": round(float(preds_r[n_idx]), 2),
            "actual_cases": int(truths_r[n_idx]),
            "truth": int(truths_c[n_idx]),
        })

    # Unique undirected edges
    seen = set()
    edges = []
    edge_attr_raw = globals().get("edge_attr_raw", edge_attr)
    for i in range(edge_index.shape[1]):
        s, d = int(edge_index[0, i]), int(edge_index[1, i])
        key = (min(s, d), max(s, d))
        if key not in seen:
            seen.add(key)
            edges.append({
                "source": s,
                "target": d,
                "weight_km": round(float(edge_attr_raw[i]), 1),
                "weight_norm": round(float(edge_attr[i]), 4),
            })

    return {
        "nodes": nodes,
        "edges": edges,
        "year": int(row_ts.iso_year),
        "week": int(row_ts.iso_week),
        "t_idx": t_idx,
        "total_windows": len(windows),
    }


@app.get("/api/timeline")
def get_timeline(start: int = 0, end: int = -1):
    """Return predictions for a range of time windows (compact format)."""
    end_idx = end if end >= 0 else len(windows)
    start_idx = max(0, start)
    end_idx = min(end_idx, len(windows))

    timeline = []
    for wi in range(start_idx, end_idx):
        result = run_window(wi)
        if result is None: continue
        probs, preds_r, truths_c, truths_r, t_target = result
        row_ts = ts.iloc[t_target]

        # Only include districts with prob > 0.05 or truth == 1 to keep response small
        preds = []
        for n_idx in range(N_NODES):
            if probs[n_idx] > 0.05 or truths_c[n_idx] == 1:
                code = idx_to_code[n_idx]
                info = next((d for d in district_info if d["censuscode"] == code), {})
                preds.append({
                    "code": int(code),
                    "name": info.get("district", ""),
                    "state": info.get("state", ""),
                    "prob": round(float(probs[n_idx]), 4),
                    "pred_cases": round(float(preds_r[n_idx]), 2),
                    "actual_cases": int(truths_r[n_idx]),
                    "truth": int(truths_c[n_idx]),
                })

        timeline.append({
            "t": wi,
            "year": int(row_ts.iso_year),
            "week": int(row_ts.iso_week),
            "alerts": sorted(preds, key=lambda x: -x["prob"]),
            "n_high_risk": sum(1 for p in preds if p["prob"] > 0.3),
            "n_outbreaks": sum(1 for p in preds if p["truth"] == 1),
        })
    return {"timeline": timeline, "total": len(windows)}


@app.get("/api/federated-demo")
def federated_demo(
    d1: int = Query(description="Censuscode of district 1"),
    d2: int = Query(description="Censuscode of district 2"),
    t: int = Query(default=-1, description="Window index"),
):
    """Step-by-step split-federated learning demo for two districts."""
    n1 = node_to_idx.get(d1)
    n2 = node_to_idx.get(d2)
    if n1 is None or n2 is None:
        return {"error": f"District codes {d1} or {d2} not found"}
    t_idx = t if t >= 0 else len(windows) - 1
    results = run_federated_demo(t_idx, [n1, n2])
    if results is None:
        return {"error": "Invalid time window"}

    # Check if they are neighbors
    are_neighbors = False
    for i in range(edge_index.shape[1]):
        if (int(edge_index[0, i]) == n1 and int(edge_index[1, i]) == n2) or \
           (int(edge_index[0, i]) == n2 and int(edge_index[1, i]) == n1):
            are_neighbors = True
            break

    row_ts = ts.iloc[windows[t_idx][3]]
    return {
        "districts": results,
        "are_neighbors": are_neighbors,
        "year": int(row_ts.iso_year),
        "week": int(row_ts.iso_week),
        "t_idx": t_idx,
        "model_params": 38010,
        "embed_dim": CFG["embed_dim"],
        "lookback": CFG["lookback"],
    }


@app.get("/api/predict")
def predict(t: int = Query(default=-1, description="Window index")):
    """Run full inference and return top-risk districts + all predictions."""
    t_idx = t if t >= 0 else len(windows) - 1
    result = run_window(t_idx)
    if result is None:
        return {"error": "Invalid time window"}
    probs, preds_r, truths_c, truths_r, t_target = result
    row_ts = ts.iloc[t_target]

    all_preds = []
    for n_idx in range(N_NODES):
        code = idx_to_code[n_idx]
        info = next((d for d in district_info if d["censuscode"] == code), {})
        all_preds.append({
            "code": int(code),
            "name": info.get("district", ""),
            "state": info.get("state", ""),
            "lat": float(info.get("lat", 0)),
            "lon": float(info.get("lon", 0)),
            "prob": round(float(probs[n_idx]), 4),
            "pred_cases": round(float(preds_r[n_idx]), 2),
            "actual_cases": int(truths_r[n_idx]),
            "truth": int(truths_c[n_idx]),
        })

    all_preds.sort(key=lambda x: -x["prob"])

    return {
        "year": int(row_ts.iso_year),
        "week": int(row_ts.iso_week),
        "t_idx": t_idx,
        "total_windows": len(windows),
        "predictions": all_preds,
        "top_10": all_preds[:10],
        "n_outbreaks_true": int(truths_c.sum()),
        "n_high_risk": sum(1 for p in all_preds if p["prob"] > 0.3),
        "avg_prob": round(float(probs.mean()), 4),
        "max_prob": round(float(probs.max()), 4),
    }


@app.get("/api/model-info")
def model_info():
    """Return model architecture info."""
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "name": "FedXGNN (Split-Federated Graph Neural Network)",
        "total_params": total_params,
        "config": CFG,
        "n_districts": N_NODES,
        "n_timesteps": N_TIME,
        "n_windows": len(windows),
        "n_edges": edge_index.shape[1] // 2,
        "components": {
            "client": {
                "gru": "2-layer GRU (32 hidden)",
                "tgat": "Temporal GAT (4 heads, 32 dim)",
                "fusion": "Linear(80→128→32) with LayerNorm + GELU",
            },
            "server": {
                "dgat": "Spatial Double GAT (4 heads, 32 dim)",
                "edge_encoder": "Linear(1→4) border-length encoding",
            },
            "head": {
                "trunk": "Linear(64→64) with LayerNorm + GELU",
                "regression": "Linear(64→32→1) case count",
                "classification": "Linear(64→32→1) outbreak probability",
            }
        }
    }


@app.get("/api/district-node/{censuscode}")
def get_district_node(censuscode: int):
    """Return single district prediction with full details for a given time window."""
    n_idx = node_to_idx.get(censuscode)
    if n_idx is None:
        return {"error": f"District {censuscode} not found"}
    t_idx = len(windows) - 1
    result = run_federated_demo(t_idx, [n_idx])
    if result is None:
        return {"error": "Inference failed"}
    row_ts = ts.iloc[windows[t_idx][4]]
    # Get neighbors
    neighbors = []
    for i in range(edge_index.shape[1]):
        s, d = int(edge_index[0, i]), int(edge_index[1, i])
        if s == n_idx:
            code = idx_to_code[d]
            info = next((di for di in district_info if di["censuscode"] == code), {})
            neighbors.append({"code": int(code), "name": info.get("district", ""), "state": info.get("state", "")})
    return {
        "district": result[0],
        "neighbors": neighbors,
        "year": int(row_ts.iso_year),
        "week": int(row_ts.iso_week),
    }


from pydantic import BaseModel
from typing import List, Dict, Optional

class WeekData(BaseModel):
    temp_k: float = 0.0
    preci_mm: float = 0.0
    LAI: float = 0.0
    cases_lag1: float = 0.0
    cases_lag2: float = 0.0
    cases_lag3: float = 0.0
    week_sin: float = 0.0
    week_cos: float = 0.0
    is_monsoon: float = 0.0

class DistrictInput(BaseModel):
    censuscode: int
    weeks: List[WeekData]

class CustomPredictRequest(BaseModel):
    districts: List[DistrictInput]


@app.post("/api/custom-predict")
def custom_predict(req: CustomPredictRequest):
    """
    Accept user-uploaded JSON data for districts (4 weeks each),
    overlay onto the existing graph, run step-by-step inference,
    and return intermediate activations + predictions.
    """
    if len(req.districts) < 1 or len(req.districts) > 5:
        return {"error": "Provide 1-5 districts"}

    # Use the last window as a baseline
    t_last = len(windows) - 1
    x_win_base, y_c, y_r, obs_m, t_target = windows[t_last]
    x_d = torch.nan_to_num(x_win_base.to(DEVICE), nan=0.0).clone()
    x_s = X_stat.to(DEVICE)
    ei = edge_index.to(DEVICE)
    ea = edge_attr.to(DEVICE)

    target_indices = []
    feat_names = CFG["dynamic_features"]

    for d_input in req.districts:
        n_idx = node_to_idx.get(d_input.censuscode)
        if n_idx is None:
            return {"error": f"District code {d_input.censuscode} not in graph"}
        target_indices.append(n_idx)

        weeks = d_input.weeks
        if len(weeks) != 4:
            return {"error": f"District {d_input.censuscode}: Need exactly 4 weeks, got {len(weeks)}"}

        # Overlay user data into the tensor
        for t_w, week in enumerate(weeks):
            week_dict = week.model_dump()
            raw_vals = []
            for fn in feat_names:
                v = float(week_dict.get(fn, 0.0))
                # Apply log1p to cases just like training
                if "cases" in fn.lower():
                    v = np.log1p(v)
                raw_vals.append(v)
            # scale the values using scaler_dyn so the model doesn't blow up
            scaled_vals = scaler_dyn.transform([raw_vals])[0]
            for fi, _ in enumerate(feat_names):
                x_d[n_idx, t_w, fi] = scaled_vals[fi]

    # Run step-by-step inference on the full graph with overlaid user data
    with torch.no_grad():
        _, h_n = model.client.gru(x_d)
        h_gru = h_n[-1]
        h_tgat = model.client.tgat(x_d)
        parts = [h_gru, h_tgat]
        if model.client.static_enc is not None:
            parts.append(model.client.static_enc(x_s))
        client_emb = model.client.fusion(torch.cat(parts, dim=-1))
        spatial_emb = model.server(client_emb, ei, ea)
        fused = torch.cat([client_emb, spatial_emb], dim=-1)
        cases_pred_norm, logit = model.head(fused)
        probs = logit.sigmoid()

        c_mean = model_params["cases_mean"]
        c_std  = model_params["cases_std"]
        cases_pred = cases_pred_norm.cpu().numpy() * c_std + c_mean
        if model_params["is_target_log"]:
            cases_pred = np.expm1(cases_pred)
        cases_pred = np.maximum(cases_pred, 0.0)

    results = []
    for n_idx in target_indices:
        code = idx_to_code[n_idx]
        info = next((d for d in district_info if d["censuscode"] == code), {})
        raw_feats = {}
        raw_feats_array = scaler_dyn.inverse_transform([x_d[n_idx, -1].cpu().numpy()])[0]
        for fi, fn in enumerate(feat_names):
            raw_feats[fn] = round(float(raw_feats_array[fi]), 4)

        # Find edge weight between districts if there are 2
        edge_weight = None
        if len(target_indices) == 2:
            other = target_indices[1] if n_idx == target_indices[0] else target_indices[0]
            for i in range(edge_index.shape[1]):
                if int(edge_index[0, i]) == n_idx and int(edge_index[1, i]) == other:
                    edge_weight = round(float(globals().get("edge_attr_raw", edge_attr)[i]), 2)
                    break

        results.append({
            "node_idx": n_idx,
            "censuscode": int(code),
            "district": info.get("district", "Unknown"),
            "state": info.get("state", "Unknown"),
            "lat": float(info.get("lat", 0)),
            "lon": float(info.get("lon", 0)),
            "raw_features": raw_feats,
            "gru_output": [round(float(v), 4) for v in h_gru[n_idx].tolist()],
            "tgat_output": [round(float(v), 4) for v in h_tgat[n_idx].tolist()],
            "client_embedding": [round(float(v), 4) for v in client_emb[n_idx].tolist()],
            "spatial_embedding": [round(float(v), 4) for v in spatial_emb[n_idx].tolist()],
            "outbreak_prob": round(float(probs[n_idx]), 4),
            "cases_pred": round(float(cases_pred[n_idx]), 4),
            "edge_weight_km": edge_weight,
        })

    # Check if the two districts are neighbors
    are_neighbors = False
    if len(target_indices) == 2:
        n1, n2 = target_indices
        for i in range(edge_index.shape[1]):
            if (int(edge_index[0, i]) == n1 and int(edge_index[1, i]) == n2):
                are_neighbors = True
                break

    return {
        "districts": results,
        "are_neighbors": are_neighbors,
        "embed_dim": CFG["embed_dim"],
        "lookback": CFG["lookback"],
        "total_nodes_in_graph": N_NODES,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
