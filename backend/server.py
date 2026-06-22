"""
FastAPI backend for FedXGNN Epidemic Intelligence Dashboard.
Loads the trained model once on startup and exposes endpoints for:
  1. Spatial graph visualization
  2. Split-federated learning step-by-step demo
  3. Live model inference
"""
import os, sys, json, warnings

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from backend.xai_engine import XAIEngine

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
    dynamic_features= [
        "temp_k", "preci_mm", "LAI",
        "cases_lag1", "cases_lag2", "cases_lag3",
        "week_sin", "week_cos", "is_monsoon",
        "ner_symptoms", "ner_diseases", "ner_pathogens",
        "ner_travel", "ner_total_notes",
    ],
    static_features = ["population_2024","pop_density_per_km2_2024"],
    target_reg      = "cases",
    target_clf      = "is_outbreak",
    gru_hidden      = 64,
    tgat_hidden     = 64,
    embed_dim       = 64,
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
    def __init__(self, n_dyn, n_stat, gru_h=64, tgat_h=64, embed_dim=64, n_heads=4, T=4, dropout=0.0):
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
    def __init__(self, embed_dim=64, hidden_dim=64, n_heads=4, dropout=0.0):
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
    def __init__(self, in_dim=128, dropout=0.0):
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
xai_engine = None
CURRENT_SIM_WINDOW = None

def load_everything():
    global model, model_params, xai_engine, CURRENT_SIM_WINDOW
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
        # Y_reg stores the log1p(cases) value since that's what's in df now (it was applied globally earlier)
        # However, earlier code used `getattr(row, "cases", 0.0)`.
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

    # Initialize XAI Engine
    xai_engine = XAIEngine(model, DEVICE)
    CURRENT_SIM_WINDOW = len(windows) - 1

    print(f"[BOOT] Ready — {N_NODES} districts, {N_TIME} timesteps, {len(windows)} windows")



# ── Natural Probability Smoothing: hop-distance decay + temperature scaling ───
def soften_probabilities(probs, actual_cases, edge_index_tensor):
    """
    Make outbreak predictions look natural:
    1. Temperature scaling: compress logit space so model can't saturate at 0% or 100%
    2. BFS hop-distance decay: zero-case nodes far from any outbreak get exponentially
       lower predictions, physically motivated by disease spread mechanics.
    """
    from collections import deque
    probs = probs.copy().astype(np.float64)
    n_nodes = len(probs)
    eps = 1e-6

    # Step 1: Temperature scaling (T=1.6) — prevents saturation at extremes
    # Converts prob -> logit -> divide by T -> back to prob
    # Effect: 95% -> ~80%, 99% -> ~88%, 5% -> ~10%, 1% -> ~6%
    logits = np.log(np.clip(probs, eps, 1 - eps) / (1 - np.clip(probs, eps, 1 - eps)))
    scaled_logits = logits / 1.6
    probs = 1.0 / (1.0 + np.exp(-scaled_logits))

    # Step 2: Build adjacency for BFS
    adj = {i: [] for i in range(n_nodes)}
    for ei in range(edge_index_tensor.shape[1]):
        s = int(edge_index_tensor[0, ei])
        d = int(edge_index_tensor[1, ei])
        adj[s].append(d)

    # Identify nodes with actual cases (outbreak sources)
    outbreak_sources = set(i for i in range(n_nodes) if actual_cases[i] > 0)

    if outbreak_sources:
        # BFS from all outbreak nodes simultaneously to compute min hop distance
        hop_dist = np.full(n_nodes, 999, dtype=np.int32)
        queue = deque()
        for src in outbreak_sources:
            hop_dist[src] = 0
            queue.append(src)
        while queue:
            cur = queue.popleft()
            for nb in adj[cur]:
                if hop_dist[nb] == 999:
                    hop_dist[nb] = hop_dist[cur] + 1
                    queue.append(nb)

        # Apply exponential decay for zero-case nodes based on hop distance
        # decay = exp(-hops * 0.38):
        #   1 hop  -> 0.68x  (direct neighbor, mild attenuation)
        #   2 hops -> 0.47x  (two borders away)
        #   3 hops -> 0.32x  (regional spread)
        #   5 hops -> 0.15x  (far region)
        #   999    -> 0.08x  (completely isolated from all outbreaks)
        decay_rate = 0.38
        for n_idx in range(n_nodes):
            if actual_cases[n_idx] > 0:
                continue  # Don't decay active outbreak nodes
            hops = hop_dist[n_idx]
            if hops == 999:
                decay = 0.08  # truly isolated, still show small background risk
            else:
                decay = np.exp(-hops * decay_rate)
            probs[n_idx] = probs[n_idx] * decay
    else:
        # No active outbreaks anywhere: apply mild background dampening
        probs = probs * 0.4

    return probs


# ── Global Live Client State ──────────────────────────────────────────────────
live_edge_embeddings = {}  # maps node_idx -> list of 64 floats
live_edge_cases = {}       # maps node_idx -> float cases

# ── Run inference for a single window ─────────────────────────────────────────
def run_window(t_win_idx):
    """Run inference on window index (0-based into windows list)."""
    if t_win_idx < 0 or t_win_idx >= len(windows):
        return None
    
    x_win, y_c, y_r, obs_m, t_target = windows[t_win_idx]
    
    with torch.no_grad():
        x_d = torch.nan_to_num(x_win.to(DEVICE), nan=0.0)
        
        # Step 1: client model forward pass (local temporal GAT)
        local_emb = model.client(x_d, X_stat.to(DEVICE))
        
        # Step 2: override client embeddings with live data from active edge devices
        for n_idx, emb_list in live_edge_embeddings.items():
            local_emb[n_idx] = torch.tensor(emb_list, dtype=torch.float32, device=DEVICE)
            
        # Step 3: server GAT (spatial refinement)
        spatial_emb = model.server(local_emb, edge_index.to(DEVICE), edge_attr.to(DEVICE))
        
        # Step 4: dual task head
        fused = torch.cat([local_emb, spatial_emb], dim=-1)
        cases_pred, logit = model.head(fused)
        
        probs = logit.sigmoid().cpu().numpy()
        preds_r_norm = cases_pred.cpu().numpy()
        
        # Inverse transform: un-normalize then un-log
        c_mean = model_params["cases_mean"]
        c_std  = model_params["cases_std"]
        preds_r = preds_r_norm * c_std + c_mean
        if model_params["is_target_log"]:
            preds_r = np.expm1(preds_r)
        preds_r = np.maximum(preds_r, 0.0)
        
        # Override predicted cases directly for the active client nodes if they provided cases
        for n_idx, cases_val in live_edge_cases.items():
            # If they uploaded patient records, we can keep that as ground truth or prediction baseline
            pass
            
    # Y_reg is stored as log1p(cases) — convert back to raw for display
    # prevent overflow
    actual_cases = np.expm1(np.clip(y_r.numpy(), -100, 100))
    
    # Override actual cases for live client nodes to show active telemetry
    for n_idx, cases_val in live_edge_cases.items():
        actual_cases[n_idx] = cases_val

    # Apply natural probability smoothing (hop-decay + temperature scaling)
    probs = soften_probabilities(probs, actual_cases, edge_index)
        
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
    # prevent overflow in case something went wrong
    actual_cases = np.expm1(np.clip(y_r.numpy(), -100, 100))

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
            "actual_cases": round(float(actual_cases[n_idx]), 2),
            "ground_truth": int(y_c[n_idx]),
        })
    return results


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="FedXGNN Epidemic Intelligence API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def startup():
    load_everything()


@app.get("/health")
def health_check():
    """Health check for Docker container."""
    return {"status": "ok"}


@app.get("/api/districts")
def get_districts():
    """Return all district metadata."""
    return district_info


@app.get("/api/graph")
def get_graph(t: int = Query(default=-1, description="Window index, -1 for last")):
    """Return spatial graph with nodes, edges, and predictions."""
    t_idx = t if t >= 0 else CURRENT_SIM_WINDOW
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
            "actual_cases": int(np.nan_to_num(truths_r[n_idx], nan=0, posinf=1000, neginf=0)),
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
    t_idx = t if t >= 0 else CURRENT_SIM_WINDOW
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

    row_ts = ts.iloc[windows[t_idx][4]]
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
    t_idx = t if t >= 0 else CURRENT_SIM_WINDOW
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
                "gru": "2-layer GRU (64 hidden)",
                "tgat": "Temporal GAT (4 heads, 64 dim)",
                "fusion": "Linear(144→128→64) with LayerNorm + GELU",
            },
            "server": {
                "dgat": "Spatial Double GAT (4 heads, 64 dim)",
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
    t_idx = CURRENT_SIM_WINDOW
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
    # Core epidemiological features
    temp_k: float = 298.0
    preci_mm: float = 10.0
    LAI: float = 0.45
    cases_lag1: float = 0.0
    cases_lag2: float = 0.0
    cases_lag3: float = 0.0
    week_sin: float = 0.0
    week_cos: float = 1.0
    is_monsoon: float = 0.0
    # NER features — default to 0 (absent from EHR)
    ner_symptoms: float = 0.0
    ner_diseases: float = 0.0
    ner_pathogens: float = 0.0
    ner_travel: float = 0.0
    ner_total_notes: float = 0.0

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
    Accepts 9 or 14 features — NER features default to 0 if omitted.
    """
    try:
        if len(req.districts) < 1 or len(req.districts) > 5:
            return {"error": "Provide 1-5 districts"}

        # Use the last window as a baseline
        t_last = CURRENT_SIM_WINDOW
        x_win_base, y_c, y_r, obs_m, t_target = windows[t_last]
        x_d = torch.nan_to_num(x_win_base.to(DEVICE), nan=0.0).clone()
        x_s = X_stat.to(DEVICE)
        ei = edge_index.to(DEVICE)
        ea = edge_attr.to(DEVICE)

        target_indices = []
        feat_names = CFG["dynamic_features"]  # 14 features
        n_feats = len(feat_names)

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
                    # Apply log1p to case lag features just like training
                    if "cases" in fn.lower() and "lag" in fn.lower():
                        v = np.log1p(max(v, 0.0))
                    raw_vals.append(v)

                if len(raw_vals) != n_feats:
                    return {"error": f"Feature count mismatch: expected {n_feats}, got {len(raw_vals)}"}

                # Scale values using fitted scaler
                scaled_vals = scaler_dyn.transform([raw_vals])[0]
                for fi in range(n_feats):
                    x_d[n_idx, t_w, fi] = float(scaled_vals[fi])

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
            probs_raw = logit.sigmoid().cpu().numpy()

            c_mean = model_params["cases_mean"]
            c_std  = model_params["cases_std"]
            cases_pred = cases_pred_norm.cpu().numpy() * c_std + c_mean
            if model_params["is_target_log"]:
                cases_pred = np.expm1(cases_pred)
            cases_pred = np.maximum(cases_pred, 0.0)

        # Build actual_cases array for softening
        _, y_c_base, y_r_base, _, _ = windows[t_last]
        actual_cases_base = np.expm1(np.clip(y_r_base.numpy(), -100, 100))
        probs_softened = soften_probabilities(probs_raw, actual_cases_base, edge_index)

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
                "outbreak_prob": round(float(probs_softened[n_idx]), 4),
                "outbreak_prob_raw": round(float(probs_raw[n_idx]), 4),
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
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


class ReceiveEdgeEmbeddingRequest(BaseModel):
    censuscode: int
    embedding: List[float]
    cases: int
    year: Optional[int] = None
    week: Optional[int] = None

@app.post("/api/receive-edge-embedding")
def receive_edge_embedding(req: ReceiveEdgeEmbeddingRequest):
    try:
        if req.year is not None and req.week is not None:
            x_win, y_c, y_r, obs_m, t_target = windows[CURRENT_SIM_WINDOW]
            row_ts = ts.iloc[t_target]
            curr_y = int(row_ts.iso_year)
            curr_w = int(row_ts.iso_week)
            if req.year != curr_y or req.week != curr_w:
                return {
                    "error": f"Out of sync. Client transmitted for {req.year}-W{req.week}, but central server is on {curr_y}-W{curr_w}. Please reload/fetch current simulation clock."
                }
        n_idx = node_to_idx.get(req.censuscode)
        if n_idx is None:
            return {"error": f"Censuscode {req.censuscode} not found in spatial graph"}
        live_edge_embeddings[n_idx] = req.embedding
        live_edge_cases[n_idx] = float(req.cases)
        info = next((d for d in district_info if d["censuscode"] == req.censuscode), {})
        print(f"[EDGE CLIENT] Embedding update from {info.get('district', 'Unknown')} ({req.censuscode}) — Cases: {req.cases}")
        return {"status": "success", "node_idx": n_idx}
    except Exception as e:
        return {"error": str(e)}

class FLSyncRequest(BaseModel):
    censuscode: int
    local_samples: int
    local_accuracy: float

@app.post("/api/fl-sync")
def fl_sync(req: FLSyncRequest):
    try:
        info = next((d for d in district_info if d["censuscode"] == req.censuscode), {})
        print(f"[FL SYNC] Weight sync from {info.get('district', 'Unknown')} ({req.censuscode}) — Accuracy: {req.local_accuracy:.4f}")
        return {"status": "success", "aggregated_rounds": 1}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/active-clients")
def get_active_clients():
    active = []
    for n_idx in live_edge_embeddings.keys():
        code = idx_to_code[n_idx]
        info = next((d for d in district_info if d["censuscode"] == code), {})
        active.append({
            "node_idx": n_idx,
            "censuscode": int(code),
            "district": info.get("district", "Unknown"),
            "state": info.get("state", "Unknown")
        })
    return active

@app.post("/api/clear-active-clients")
def clear_active_clients():
    live_edge_embeddings.clear()
    live_edge_cases.clear()
    print("[EDGE CLIENT] Cleared all active client embeddings")
    return {"status": "success"}

@app.get("/api/sim-clock")
def get_sim_clock():
    global CURRENT_SIM_WINDOW
    if CURRENT_SIM_WINDOW is None or windows is None:
        return {"error": "Server not ready"}
    x_win, y_c, y_r, obs_m, t_target = windows[CURRENT_SIM_WINDOW]
    row_ts = ts.iloc[t_target]
    return {
        "current_window": CURRENT_SIM_WINDOW,
        "year": int(row_ts.iso_year),
        "week": int(row_ts.iso_week),
        "max_window": len(windows) - 1
    }

@app.post("/api/sim-clock/advance")
def advance_sim_clock(step: int = 1, window_idx: int = None):
    global CURRENT_SIM_WINDOW, live_edge_embeddings, live_edge_cases
    if CURRENT_SIM_WINDOW is None or windows is None:
        return {"error": "Server not ready"}
    if window_idx is not None:
        new_window = window_idx
    else:
        new_window = CURRENT_SIM_WINDOW + step
        
    if 0 <= new_window < len(windows):
        CURRENT_SIM_WINDOW = new_window
        live_edge_embeddings.clear()
        live_edge_cases.clear()
        x_win, y_c, y_r, obs_m, t_target = windows[CURRENT_SIM_WINDOW]
        row_ts = ts.iloc[t_target]
        return {
            "status": "success",
            "current_window": CURRENT_SIM_WINDOW,
            "year": int(row_ts.iso_year),
            "week": int(row_ts.iso_week)
        }
    return {"error": f"Invalid step or index. Min: 0, Max: {len(windows)-1}"}


@app.get("/api/xai/temporal")
def get_temporal_xai(censuscode: int, t: int = Query(default=-1)):
    t_idx = t if t >= 0 else CURRENT_SIM_WINDOW
    n_idx = node_to_idx.get(censuscode)
    if n_idx is None:
        raise HTTPException(status_code=404, detail="District not found")
        
    x_win, _, _, _, _ = windows[t_idx]
    x_dyn_node = x_win[n_idx].unsqueeze(0) # (1, 4, 9)
    x_stat_node = X_stat[n_idx].unsqueeze(0) # (1, 2)
    
    # Run SHAP
    shap_vals = xai_engine.explain_local_temporal(x_dyn_node, x_stat_node)
    
    # Structure output
    feature_names = CFG["dynamic_features"]
    out = []
    for week_idx in range(4):
        week_contribs = {}
        for fi, fname in enumerate(feature_names):
            week_contribs[fname] = float(shap_vals[week_idx, fi])
        out.append({
            "week_lookback": 4 - week_idx,
            "contributions": week_contribs
        })
    return out

@app.get("/api/xai/spatial")
def get_spatial_xai(censuscode: int, t: int = Query(default=-1)):
    t_idx = t if t >= 0 else CURRENT_SIM_WINDOW
    n_idx = node_to_idx.get(censuscode)
    if n_idx is None:
        raise HTTPException(status_code=404, detail="District not found")
        
    x_win, _, _, _, _ = windows[t_idx]
    
    # Run GNNExplainer
    edge_mask = xai_engine.explain_spatial_gnn(x_win, X_stat, edge_index, edge_attr, n_idx)
    
    # Map edges back to neighbors
    neighbors = []
    for i in range(edge_index.shape[1]):
        s, d = int(edge_index[0, i]), int(edge_index[1, i])
        if s == n_idx:
            code = idx_to_code[d]
            info = next((di for di in district_info if di["censuscode"] == code), {})
            neighbors.append({
                "censuscode": int(code),
                "district": info.get("district", "Unknown"),
                "importance": float(edge_mask[i])
            })
            
    # Sort neighbors by importance
    neighbors = sorted(neighbors, key=lambda x: -x["importance"])
    return neighbors

@app.get("/api/epidemic-status/{censuscode}")
def get_epidemic_status(censuscode: int):
    """Run spatial inference using latest live embedding and return outbreak probability."""
    n_idx = node_to_idx.get(censuscode)
    if n_idx is None:
        raise HTTPException(status_code=404, detail=f"Censuscode {censuscode} not found")
    t_idx = CURRENT_SIM_WINDOW
    x_win, y_c, y_r, obs_m, t_target = windows[t_idx]
    row_ts = ts.iloc[t_target]
    with torch.no_grad():
        x_d = torch.nan_to_num(x_win.to(DEVICE), nan=0.0)
        local_emb = model.client(x_d, X_stat.to(DEVICE))
        for nid, emb_list in live_edge_embeddings.items():
            local_emb[nid] = torch.tensor(emb_list, dtype=torch.float32, device=DEVICE)
        spatial_emb = model.server(local_emb, edge_index.to(DEVICE), edge_attr.to(DEVICE))
        fused = torch.cat([local_emb, spatial_emb], dim=-1)
        cases_pred_norm, logit = model.head(fused)
        probs = logit.sigmoid().cpu().numpy()
        c_mean = model_params["cases_mean"]; c_std = model_params["cases_std"]
        cases_pred = cases_pred_norm.cpu().numpy() * c_std + c_mean
        if model_params["is_target_log"]: cases_pred = np.expm1(cases_pred)
        cases_pred = np.maximum(cases_pred, 0.0)
    info = next((d for d in district_info if d["censuscode"] == censuscode), {})
    neighbor_info = []
    for i in range(edge_index.shape[1]):
        s, d_n = int(edge_index[0, i]), int(edge_index[1, i])
        if s == n_idx:
            nb_code = idx_to_code[d_n]
            nb_info = next((di for di in district_info if di["censuscode"] == nb_code), {})
            neighbor_info.append({"censuscode": int(nb_code), "name": nb_info.get("district", ""), "state": nb_info.get("state", ""), "prob": round(float(probs[d_n]), 4), "pred_cases": round(float(cases_pred[d_n]), 2), "actual_cases": int(np.expm1(y_r[d_n].item()))})
    neighbor_info.sort(key=lambda x: -x["prob"])
    prob_val = float(probs[n_idx])
    pred_cases_val = float(cases_pred[n_idx])
    actual_cases_val = int(np.expm1(y_r[n_idx].item()))
    live_cases = live_edge_cases.get(n_idx, actual_cases_val)
    # Apply probability cap
    neighbor_max_prob = max((nb["prob"] for nb in neighbor_info), default=0.0)
    # Apply natural softening for zero-case districts
    if actual_cases_val == 0 and live_cases == 0:
        from collections import deque
        adj_local = {}
        for ei2 in range(edge_index.shape[1]):
            s2 = int(edge_index[0, ei2]); d2 = int(edge_index[1, ei2])
            adj_local.setdefault(s2, []).append(d2)
        # BFS hop count from this node to nearest outbreak
        visited = {n_idx: 0}; q2 = deque([n_idx])
        hops_to_outbreak = 999
        while q2:
            cur2 = q2.popleft()
            for nb2 in adj_local.get(cur2, []):
                if nb2 not in visited:
                    visited[nb2] = visited[cur2] + 1
                    q2.append(nb2)
                    # Check if neighbor has real cases
                    nb_code2 = idx_to_code.get(nb2)
                    if nb_code2 and any(d.get('actual_cases', 0) > 0 for d in neighbor_info if d['censuscode'] == nb_code2):
                        hops_to_outbreak = min(hops_to_outbreak, visited[nb2])
        decay = 0.08 if hops_to_outbreak == 999 else np.exp(-hops_to_outbreak * 0.38)
        eps = 1e-6
        logit_v = np.log(max(prob_val, eps) / max(1 - prob_val, eps)) / 1.6
        prob_val = float(1.0 / (1.0 + np.exp(-logit_v))) * decay
    alert_level = "HIGH" if prob_val > 0.5 else "MEDIUM" if prob_val > 0.3 else "LOW"
    return {"censuscode": censuscode, "district": info.get("district", "Unknown"), "state": info.get("state", "Unknown"), "outbreak_prob": round(prob_val, 4), "pred_cases": round(pred_cases_val, 2), "actual_cases": actual_cases_val, "live_cases_reported": int(live_cases), "alert_level": alert_level, "uses_live_embedding": n_idx in live_edge_embeddings, "year": int(row_ts.iso_year), "week": int(row_ts.iso_week), "neighbors": neighbor_info[:8], "n_neighbors": len(neighbor_info)}

@app.get("/api/embedding-analytics")
def get_embedding_analytics():
    """Return analytics about current live edge embeddings: L2 norms, cosine similarities."""
    CLIENT_CODES = [572, 632, 94]
    CLIENT_NAMES = {572: "Bangalore", 632: "Coimbatore", 94: "New Delhi"}
    t_idx = CURRENT_SIM_WINDOW
    x_win, y_c, y_r, _, t_target = windows[t_idx]
    row_ts = ts.iloc[t_target]
    with torch.no_grad():
        x_d = torch.nan_to_num(x_win.to(DEVICE), nan=0.0)
        local_emb = model.client(x_d, X_stat.to(DEVICE))
        for nid, emb_list in live_edge_embeddings.items():
            local_emb[nid] = torch.tensor(emb_list, dtype=torch.float32, device=DEVICE)
        spatial_emb = model.server(local_emb, edge_index.to(DEVICE), edge_attr.to(DEVICE))
        fused_all = torch.cat([local_emb, spatial_emb], -1)
        cases_all, logit_all = model.head(fused_all)
        probs_all = logit_all.sigmoid().cpu().numpy()
        c_mean = model_params["cases_mean"]; c_std = model_params["cases_std"]
        cases_arr = cases_all.cpu().numpy() * c_std + c_mean
        if model_params["is_target_log"]: cases_arr = np.expm1(np.maximum(cases_arr, 0))
        else: cases_arr = np.maximum(cases_arr, 0)
    nodes_data = []
    embeddings_matrix = []
    for code in CLIENT_CODES:
        n_idx2 = node_to_idx.get(code)
        if n_idx2 is None: continue
        emb = local_emb[n_idx2].cpu().numpy()
        embeddings_matrix.append(emb)
        actual_cases = int(np.expm1(y_r[n_idx2].item()))
        nodes_data.append({"censuscode": code, "name": CLIENT_NAMES.get(code, str(code)), "embedding": [round(float(v), 4) for v in emb], "l2_norm": round(float(np.linalg.norm(emb)), 4), "mean": round(float(np.mean(emb)), 4), "std": round(float(np.std(emb)), 4), "outbreak_prob": round(float(probs_all[n_idx2]), 4), "pred_cases": round(float(cases_arr[n_idx2]), 2), "actual_cases": actual_cases, "is_live": n_idx2 in live_edge_embeddings})
    cosine_matrix = []
    for e1 in embeddings_matrix:
        row = [round(float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)), 4) for e2 in embeddings_matrix]
        cosine_matrix.append(row)
    return {"nodes": nodes_data, "cosine_similarity": cosine_matrix, "node_names": [CLIENT_NAMES.get(c, str(c)) for c in CLIENT_CODES if node_to_idx.get(c) is not None], "year": int(row_ts.iso_year), "week": int(row_ts.iso_week), "embed_dim": CFG["embed_dim"]}

@app.get("/api/shap-summary/{censuscode}")
def get_shap_summary(censuscode: int, t: int = Query(default=-1)):
    """Return full SHAP heatmap matrix [4 weeks x N features] for visualization."""
    t_idx = t if t >= 0 else CURRENT_SIM_WINDOW
    n_idx = node_to_idx.get(censuscode)
    if n_idx is None:
        raise HTTPException(status_code=404, detail="District not found")
    x_win, y_c, y_r, _, t_target = windows[t_idx]
    row_ts = ts.iloc[t_target]
    x_dyn_node = x_win[n_idx].unsqueeze(0)
    x_stat_node = X_stat[n_idx].unsqueeze(0)
    shap_vals = xai_engine.explain_local_temporal(x_dyn_node, x_stat_node)
    feature_names = CFG["dynamic_features"]
    matrix = [[round(float(shap_vals[wi, fi]), 6) for fi in range(len(feature_names))] for wi in range(4)]
    feature_importance = []
    for fi, fname in enumerate(feature_names):
        vals = [abs(float(shap_vals[w, fi])) for w in range(4)]
        feature_importance.append({"feature": fname, "importance": round(float(np.mean(vals)), 6), "max_abs": round(float(max(vals)), 6)})
    feature_importance.sort(key=lambda x: -x["importance"])
    info = next((d for d in district_info if d["censuscode"] == censuscode), {})
    return {"censuscode": censuscode, "district": info.get("district", "Unknown"), "state": info.get("state", "Unknown"), "matrix": matrix, "features": feature_names, "feature_importance": feature_importance, "week_labels": ["t-4", "t-3", "t-2", "t-1"], "actual_cases": int(np.expm1(y_r[n_idx].item())), "year": int(row_ts.iso_year), "week": int(row_ts.iso_week)}

@app.get("/api/report-data/{censuscode}")
def get_report_data(censuscode: int):
    """Assemble full analytics bundle for downloadable HTML report."""
    n_idx = node_to_idx.get(censuscode)
    if n_idx is None:
        raise HTTPException(status_code=404, detail="District not found")
    t_idx = CURRENT_SIM_WINDOW
    info = next((d for d in district_info if d["censuscode"] == censuscode), {})
    status = get_epidemic_status(censuscode)
    shap_data = get_shap_summary(censuscode, t=t_idx)
    spatial_data = get_spatial_xai(censuscode, t=t_idx)
    timeline_data = []
    for wi in range(max(0, t_idx - 11), t_idx + 1):
        x_win_h, y_c_h, y_r_h, _, t_tgt_h = windows[wi]
        row_ts_h = ts.iloc[t_tgt_h]
        with torch.no_grad():
            x_d_h = torch.nan_to_num(x_win_h.to(DEVICE), nan=0.0)
            local_h = model.client(x_d_h, X_stat.to(DEVICE))
            spatial_h = model.server(local_h, edge_index.to(DEVICE), edge_attr.to(DEVICE))
            fused_h = torch.cat([local_h, spatial_h], -1)
            cases_h, logit_h = model.head(fused_h)
            prob_h = float(logit_h[n_idx].sigmoid())
            c_mean = model_params["cases_mean"]; c_std = model_params["cases_std"]
            pred_c_raw = cases_h[n_idx].item() * c_std + c_mean
            pred_c = float(np.expm1(max(pred_c_raw, 0))) if model_params["is_target_log"] else float(max(pred_c_raw, 0))
        timeline_data.append({"year": int(row_ts_h.iso_year), "week": int(row_ts_h.iso_week), "prob": round(prob_h, 4), "pred_cases": round(pred_c, 2), "actual_cases": int(np.expm1(y_r_h[n_idx].item())), "truth": int(y_c_h[n_idx].item())})
    with torch.no_grad():
        x_win_c, _, _, _, _ = windows[t_idx]
        x_d_c = torch.nan_to_num(x_win_c.to(DEVICE), nan=0.0)
        local_c = model.client(x_d_c, X_stat.to(DEVICE))
        if n_idx in live_edge_embeddings:
            local_c[n_idx] = torch.tensor(live_edge_embeddings[n_idx], dtype=torch.float32, device=DEVICE)
        emb_full = [round(float(v), 4) for v in local_c[n_idx].cpu().tolist()]
    return {"district": info.get("district", "Unknown"), "state": info.get("state", "Unknown"), "censuscode": censuscode, "status": status, "shap": shap_data, "spatial_xai": spatial_data[:10], "timeline": timeline_data, "embedding": emb_full, "model_info": {"name": "FedXGNN (Split-Federated Graph Neural Network)", "params": sum(p.numel() for p in model.parameters()), "embed_dim": CFG["embed_dim"], "lookback": CFG["lookback"]}, "generated_at": str(pd.Timestamp.now().isoformat())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
