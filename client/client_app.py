import os
import sys
import argparse
import uvicorn
import torch
import numpy as np
import pandas as pd
import requests
import json
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add parent directory to path so we can import model definition from backend
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from backend.server import FedXGNN, ClientTemporalModel, CFG

# Import EHR parser
from client.ehr_parser import EHRParser

app = FastAPI(title="Hospital Edge Client Node")
parser = EHRParser()

# Global state for client
CLIENT_CONFIG = {
    "censuscode": 572, # Bangalore default
    "name": "Bangalore Hospital",
    "server_url": "http://localhost:8000",
    "port": 8001
}

# Local variables
model = None
local_df = None
scaler_dyn = None
scaler_stat = None
uploaded_cases = [] # parsed cases for the current simulation week
avail_dyn = []     # features the scaler was actually fitted on (may be 9 if NER cols missing)

def init_client(censuscode, server_url):
    global model, local_df, scaler_dyn, scaler_stat, avail_dyn
    
    # 1. Load full dataset to fit scalers exactly like backend to avoid mismatch
    data_path = os.path.join(PROJECT_ROOT, "data", "training_dataset_enhanced_v2.csv")
    df = pd.read_csv(data_path)
    df = df.sort_values(["censuscode","iso_year","iso_week"]).reset_index(drop=True)
    
    unique_codes = sorted(df["censuscode"].unique())
    node_to_idx = {c: i for i, c in enumerate(unique_codes)}
    df["node_idx"] = df["censuscode"].map(node_to_idx)
    
    ts_df = df[["iso_year","iso_week"]].drop_duplicates().sort_values(["iso_year","iso_week"]).reset_index(drop=True)
    ts_df["t_idx"] = range(len(ts_df))
    df = df.merge(ts_df, on=["iso_year","iso_week"])
    
    # Fit scalers
    from sklearn.preprocessing import StandardScaler
    avail_dyn_cols = [f for f in CFG["dynamic_features"] if f in df.columns]
    avail_stat = [f for f in CFG["static_features"] if f in df.columns]
    
    LB = CFG["lookback"]
    split_idx = int((len(ts_df) - LB) * CFG["train_ratio"])
    train_cut = split_idx + LB
    train_mask = df["t_idx"] < train_cut
    
    # Match the log1p transform from server
    log_cols = [c for c in df.columns if "cases" in c.lower()]
    for c in log_cols:
        df[c] = np.log1p(df[c])
        
    scaler_dyn = StandardScaler()
    scaler_stat = StandardScaler()
    avail_dyn = avail_dyn_cols  # store globally so transmit knows which features to scale
    scaler_dyn.fit(df.loc[train_mask, avail_dyn_cols])
    
    if avail_stat:
        scaler_stat.fit(df.loc[train_mask, avail_stat])
        
    # Get this client's historical data
    local_df = df[df["censuscode"] == censuscode].copy().sort_values("t_idx")
    
    # Load Model
    model_path = os.path.join(PROJECT_ROOT, "model", "fedxgnn_best.pt")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    m_cfg = CFG.copy()
    m_cfg["dropout"] = 0.0
    
    model = FedXGNN(m_cfg, len(CFG["dynamic_features"]), len(CFG["static_features"]))
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[*] Initialized edge client for {CLIENT_CONFIG['name']} (Census: {censuscode})")

# HTML Template for Dashboard
# Load template
DASHBOARD_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "dashboard.html")
with open(DASHBOARD_TEMPLATE_PATH, "r", encoding="utf-8") as f:
    DASHBOARD_HTML_TEMPLATE = f.read()


class TransmitRequest(BaseModel):
    cases: int
    temp_k: float
    preci_mm: float
    year: Optional[int] = None
    week: Optional[int] = None

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    # Inject server config into the template
    import json
    init_json = json.dumps({
        "censuscode": CLIENT_CONFIG["censuscode"],
        "name": CLIENT_CONFIG["name"],
        "port": CLIENT_CONFIG["port"]
    })
    
    html = DASHBOARD_HTML_TEMPLATE.replace(
        '<script id="init-data" type="application/json">\n        {}\n    </script>',
        f'<script id="init-data" type="application/json">\n        {init_json}\n    </script>'
    )
    return html

@app.get("/api/local-timeline")
def get_local_timeline():
    # Return the last 4 weeks of records from local_df
    records = local_df.tail(4).to_dict(orient="records")
    out = []
    for r in records:
        out.append({
            "year": int(r["iso_year"]),
            "week": int(r["iso_week"]),
            "temp_k": float(r["temp_k"]),
            "preci_mm": float(r["preci_mm"]),
            "LAI": float(r["LAI"]),
            "cases": int(np.expm1(r["cases"])) if "cases" in r else 0
        })
    return out

@app.post("/api/upload-ehr")
async def upload_ehr(file: UploadFile = File(...)):
    # Save the file temporarily
    os.makedirs(os.path.join(PROJECT_ROOT, "client", "temp_uploads"), exist_ok=True)
    file_path = os.path.join(PROJECT_ROOT, "client", "temp_uploads", file.filename)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    # Parse text
    text = parser.extract_text_from_file(file_path)
    parsed = parser.parse_ehr(text)
    
    # Remove file after parsing
    try:
        os.remove(file_path)
    except:
        pass
        
    record = {
        "filename": file.filename,
        "temperature_c": parsed["temperature_c"],
        "dengue_status": parsed["dengue_status"]
    }
    uploaded_cases.append(record)
    return record

@app.get("/api/ehrs")
def get_ehrs():
    return uploaded_cases

@app.delete("/api/ehrs/{idx}")
def delete_ehr(idx: int):
    if 0 <= idx < len(uploaded_cases):
        uploaded_cases.pop(idx)
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="EHR record not found")

@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Bulk upload a CSV of patient records. Columns: temperature (F or C), dengue_status (0/1)."""
    try:
        import io, csv as csv_mod
        content = await file.read()
        text = content.decode("utf-8", errors="ignore")
        reader = csv_mod.DictReader(io.StringIO(text))
        total = 0; positive = 0; negative = 0
        for row in reader:
            total += 1
            # Parse temperature
            temp_raw = float(row.get("temperature", row.get("temp", 37.0)) or 37.0)
            # Auto-detect F vs C (if > 45 assume Fahrenheit)
            temp_c = (temp_raw - 32) * 5.0 / 9.0 if temp_raw > 45 else temp_raw
            # Parse dengue status
            ds_raw = str(row.get("dengue_status", row.get("status", "-1"))).strip().lower()
            if ds_raw in ("1", "positive", "pos", "yes", "detected", "reactive"):
                dengue_status = 1; positive += 1
            elif ds_raw in ("0", "negative", "neg", "no", "not detected", "non-reactive"):
                dengue_status = 0; negative += 1
            else:
                dengue_status = -1
            uploaded_cases.append({
                "filename": f"{file.filename}:row{total}",
                "temperature_c": round(temp_c, 2),
                "dengue_status": dengue_status,
            })
        return {"status": "ok", "total": total, "positive": positive, "negative": negative}
    except Exception as e:
        return {"error": str(e)}


# District lat/lon for weather lookup
DISTRICT_COORDS = {
    572: (12.9716, 77.5946),   # Bangalore
    632: (11.0168, 76.9558),   # Coimbatore
    94:  (28.6139, 77.2090),   # Delhi
}

@app.get("/api/live-weather")
def live_weather():
    """
    Fetch live weather for this district using Open-Meteo (free, no API key).
    Returns temp_k, temp_c, preci_mm for the most recent day.
    """
    try:
        censuscode = CLIENT_CONFIG["censuscode"]
        coords = DISTRICT_COORDS.get(censuscode, (20.5937, 78.9629))  # fallback: India center
        lat, lon = coords
        # Open-Meteo free API: daily temperature_2m_mean and precipitation_sum
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_mean,precipitation_sum"
            f"&timezone=Asia%2FKolkata&forecast_days=1"
        )
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})
        temp_c = float((daily.get("temperature_2m_mean") or [25.0])[0])
        preci_mm = float((daily.get("precipitation_sum") or [0.0])[0])
        temp_k = temp_c + 273.15
        return {
            "temp_c": round(temp_c, 2),
            "temp_k": round(temp_k, 2),
            "preci_mm": round(preci_mm, 2),
            "lat": lat, "lon": lon,
            "district_code": censuscode,
        }
    except Exception as e:
        # Fallback to sensible defaults if API is unreachable
        return {"error": str(e), "temp_k": 299.0, "temp_c": 25.85, "preci_mm": 5.0}

@app.post("/api/transmit")
def transmit_embedding(req: TransmitRequest):
    # 1. Prepare dynamic input features (last 3 weeks from local_df + current weekly report)
    # The model expects lookback=4.
    last_3 = local_df.tail(3).copy()
    
    # Convert last_3 features back to scale so we can append new one and scale everything together
    # Wait, we can just construct a raw data array, transform it using scaler_dyn, and then feed it to the model.
    raw_history = []
    for row in last_3.itertuples():
        raw_row = []
        for feat in CFG["dynamic_features"]:
            raw_row.append(getattr(row, feat, 0.0))
        raw_history.append(raw_row)
        
    # Append the new week's features (all 14 dynamic features)
    is_monsoon = 1.0 if req.preci_mm > 50.0 else 0.0
    current_raw = [
        req.temp_k,
        req.preci_mm,
        2.5,                                                  # LAI default
        float(np.log1p(max(req.cases, 0))),                   # cases_lag1 (log1p)
        float(raw_history[-1][3]) if len(raw_history) > 0 else 0.0,  # cases_lag2
        float(raw_history[-2][3]) if len(raw_history) > 1 else 0.0,  # cases_lag3
        0.5,   # week_sin
        0.8,   # week_cos
        is_monsoon,
        # NER features — pulled from uploaded EHR records if available, else 0
        float(sum(1 for e in uploaded_cases if e.get('dengue_status', -1) == 1)),  # ner_symptoms
        float(len(uploaded_cases)),                           # ner_diseases (note count proxy)
        0.0,                                                  # ner_pathogens
        0.0,                                                  # ner_travel
        float(len(uploaded_cases)),                           # ner_total_notes
    ]
    raw_history.append(current_raw)

    # Verify we have exactly 14 features matching the model
    n_feats = len(CFG["dynamic_features"])
    for i, row in enumerate(raw_history):
        if len(row) < n_feats:
            # Pad missing NER features with 0 for historical rows
            raw_history[i] = row + [0.0] * (n_feats - len(row))
        raw_history[i] = raw_history[i][:n_feats]  # truncate if somehow over
    
    # Scale only the features the scaler was fitted on (avail_dyn, typically 9)
    # Then assemble the full 14-feature tensor, leaving NER slots at 0
    all_feat_names = CFG["dynamic_features"]  # 14 features
    n_all = len(all_feat_names)
    avail_indices = [all_feat_names.index(f) for f in avail_dyn]  # positions of scalable features

    # Build avail_dyn-only rows for scaling (historical + current)
    history_for_scale = []
    for row_14 in raw_history:
        row_avail = [row_14[i] for i in avail_indices]
        history_for_scale.append(row_avail)

    scaled_avail = scaler_dyn.transform(history_for_scale)  # (4, len(avail_dyn))

    # Assemble full (4, 14) array: scaled where available, raw-0 for NER
    scaled_history = np.zeros((len(raw_history), n_all), dtype=np.float32)
    for row_i, scaled_row in enumerate(scaled_avail):
        for col_j, feat_idx in enumerate(avail_indices):
            scaled_history[row_i, feat_idx] = scaled_row[col_j]
    # NER features (not in scaler) stay 0 — already initialized above

    # Convert to tensor expected by ClientTemporalModel: (1, 4, 14)
    x_dyn = torch.tensor(scaled_history, dtype=torch.float32).unsqueeze(0)

    # Static features
    x_stat = torch.zeros(1, len(CFG["static_features"]), dtype=torch.float32)

    # Run Client Temporal Model → produces 64-dim embedding
    with torch.no_grad():
        client_model = model.client
        embedding = client_model(x_dyn, x_stat)  # (1, 64)
        emb_list = embedding.squeeze(0).tolist()
        
    # Send 64-dim embedding to central server
    try:
        server_endpoint = f"{CLIENT_CONFIG['server_url']}/api/receive-edge-embedding"
        res = requests.post(server_endpoint, json={
            "censuscode": CLIENT_CONFIG["censuscode"],
            "embedding": emb_list,
            "cases": req.cases,
            "year": req.year,
            "week": req.week
        }, timeout=5)
        srv_resp = res.json() if res.status_code == 200 else {"error": f"HTTP {res.status_code}"}
        return {
            "embedding": emb_list,
            "embedding_dim": len(emb_list),
            "server_status": "success" if res.status_code == 200 else f"Failed ({res.status_code})",
            "server_response": srv_resp,
        }
    except Exception as e:
        return {
            "embedding": emb_list,
            "embedding_dim": len(emb_list),
            "server_status": f"Connection Error: {e}",
        }

@app.get("/api/epidemic-status")
def get_epidemic_status():
    """Proxy endpoint to get spatial risk from central server for this specific node."""
    try:
        server_endpoint = f"{CLIENT_CONFIG['server_url']}/api/epidemic-status/{CLIENT_CONFIG['censuscode']}"
        res = requests.get(server_endpoint)
        if res.status_code == 200:
            return {"status": res.json()}
        return {"error": "Server returned error"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/sim-clock")
def get_sim_clock():
    """Proxy endpoint to get current central simulation clock."""
    try:
        server_endpoint = f"{CLIENT_CONFIG['server_url']}/api/sim-clock"
        res = requests.get(server_endpoint)
        if res.status_code == 200:
            return res.json()
        return {"error": "Server returned error"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/shap-summary/{censuscode}")
def get_shap_summary(censuscode: int):
    """Proxy endpoint to get SHAP feature importances from central server."""
    try:
        server_endpoint = f"{CLIENT_CONFIG['server_url']}/api/shap-summary/{censuscode}"
        res = requests.get(server_endpoint)
        if res.status_code == 200:
            return res.json()
        return {"error": f"Central server returned status {res.status_code}"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/download-report")
def download_report():
    """Generate and return an HTML report file for this client."""
    try:
        server_endpoint = f"{CLIENT_CONFIG['server_url']}/api/report-data/{CLIENT_CONFIG['censuscode']}"
        res = requests.get(server_endpoint)
        if res.status_code != 200:
            raise HTTPException(status_code=500, detail="Could not fetch report data")
        
        data = res.json()
        
        # Build simple standalone HTML report
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Epidemic Risk Report - {data['district']}</title>
            <style>
                body {{ font-family: sans-serif; padding: 40px; color: #333; }}
                h1 {{ color: #2563eb; }}
                .alert {{ padding: 15px; border-radius: 8px; margin-bottom: 20px; font-weight: bold; }}
                .alert-HIGH {{ background: #fee2e2; color: #b91c1c; }}
                .alert-MEDIUM {{ background: #fef3c7; color: #b45309; }}
                .alert-LOW {{ background: #dcfce7; color: #15803d; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background: #f8fafc; }}
            </style>
        </head>
        <body>
            <h1>Federated Intelligence Report</h1>
            <p><strong>District:</strong> {data['district']} ({data['censuscode']})</p>
            <p><strong>Generated At:</strong> {data['generated_at']}</p>
            
            <div class="alert alert-{data['status']['alert_level']}">
                Current Risk Assessment: {data['status']['alert_level']}
                <br>Outbreak Probability: {data['status']['outbreak_prob'] * 100:.1f}%
                <br>Predicted Cases: {data['status']['pred_cases']:.1f}
            </div>
            
            <h2>Key Risk Drivers (SHAP)</h2>
            <table>
                <tr><th>Feature</th><th>Importance Score</th></tr>
                {''.join([f"<tr><td>{f['feature']}</td><td>{f['importance']:.4f}</td></tr>" for f in data['shap']['feature_importance'][:5]])}
            </table>
            
            <h2>Recent History</h2>
            <table>
                <tr><th>Week</th><th>Actual Cases</th><th>Risk Probability</th></tr>
                {''.join([f"<tr><td>{d['year']}-W{d['week']}</td><td>{d['actual_cases']}</td><td>{d['prob']*100:.1f}%</td></tr>" for d in reversed(data['timeline'][-4:])])}
            </table>
            
            <p style="margin-top: 40px; font-size: 0.8em; color: #888;">
                Generated by FedXGNN Hospital Edge Node
            </p>
        </body>
        </html>
        """
        
        return HTMLResponse(content=report_html, headers={
            "Content-Disposition": f"attachment; filename=RiskReport_{data['district']}.html"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/run-fl")
def run_fl():
    try:
        res = requests.post(f"{CLIENT_CONFIG['server_url']}/api/fl-sync", json={
            "censuscode": CLIENT_CONFIG["censuscode"],
            "local_samples": len(uploaded_cases) + 10,
            "local_accuracy": 0.88
        })
        return res.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser_arg = argparse.ArgumentParser(description="Start Hospital Edge Client Node")
    parser_arg.add_argument("--port", type=int, default=8001, help="Port to run client dashboard")
    parser_arg.add_argument("--censuscode", type=int, default=572, help="District Census Code")
    parser_arg.add_argument("--name", type=str, default="Bangalore Hospital", help="Hospital Name")
    parser_arg.add_argument("--server", type=str, default="http://localhost:8000", help="Central server URL")
    
    args = parser_arg.parse_args()
    
    CLIENT_CONFIG["censuscode"] = args.censuscode
    CLIENT_CONFIG["name"] = args.name
    CLIENT_CONFIG["server_url"] = args.server
    CLIENT_CONFIG["port"] = args.port
    
    init_client(args.censuscode, args.server)
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
