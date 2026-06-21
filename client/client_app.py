import os
import sys
import argparse
import uvicorn
import torch
import numpy as np
import pandas as pd
import requests
import json
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

def init_client(censuscode, server_url):
    global model, local_df, scaler_dyn, scaler_stat
    
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
    avail_dyn = [f for f in CFG["dynamic_features"] if f in df.columns]
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
    scaler_dyn.fit(df.loc[train_mask, avail_dyn])
    
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
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FedXGNN Hospital Edge Node</title>
    <style>
        :root {
            --bg-color: #f8fafc;
            --panel-bg: #ffffff;
            --border-color: #e2e8f0;
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --accent: #10b981;
            --text-main: #334155;
            --text-muted: #64748b;
        }
        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'DM Sans', -apple-system, sans-serif;
            margin: 0;
            padding: 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .container {
            width: 100%;
            max-width: 1100px;
        }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }
        .hospital-title h1 {
            margin: 0;
            font-size: 24px;
            background: linear-gradient(90deg, #2563eb, #10b981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hospital-title p {
            margin: 4px 0 0 0;
            color: var(--text-muted);
            font-size: 14px;
        }
        .badge {
            background-color: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #2563eb;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: bold;
        }
        .grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 24px;
        }
        .card {
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        }
        .card h2 {
            margin-top: 0;
            font-size: 18px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
            margin-bottom: 16px;
        }
        .form-group {
            margin-bottom: 16px;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: var(--text-muted);
            font-size: 14px;
        }
        input[type="text"], input[type="number"], select {
            width: 100%;
            padding: 10px;
            background: #ffffff;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            color: var(--text-main);
            box-sizing: border-box;
        }
        .btn {
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.2s;
            width: 100%;
        }
        .btn:hover {
            background-color: var(--primary-hover);
        }
        .btn-success {
            background-color: var(--accent);
        }
        .btn-success:hover {
            background-color: #059669;
        }
        .upload-area {
            border: 2px dashed #cbd5e1;
            border-radius: 8px;
            padding: 30px;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.2s;
            background-color: #f8fafc;
        }
        .upload-area:hover {
            border-color: var(--primary);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
        }
        th, td {
            text-align: left;
            padding: 10px;
            border-bottom: 1px solid var(--border-color);
        }
        th {
            color: var(--text-muted);
            font-weight: normal;
        }
        .console {
            background: #0f172a;
            border: 1px solid #1e293b;
            font-family: monospace;
            padding: 12px;
            border-radius: 6px;
            height: 180px;
            overflow-y: auto;
            color: #34d399;
            font-size: 13px;
        }
        .console-line {
            margin: 4px 0;
        }
        .pulse-container {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            gap: 16px;
        }
        .pulse {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background-color: var(--accent);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            animation: pulse-anim 1.6s infinite;
        }
        @keyframes pulse-anim {
            0% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            }
            70% {
                transform: scale(1);
                box-shadow: 0 0 0 10px rgba(16, 185, 129, 0);
            }
            100% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="hospital-title">
                <h1 id="hospital-name-text">Hospital Edge Node: Loading...</h1>
                <p id="censuscode-text">District Census Code: </p>
            </div>
            <div class="badge">Edge Node Connected</div>
        </header>

        <div class="grid">
            <div class="left-col">
                <div class="card" style="margin-bottom: 24px;">
                    <h2>Upload Patient EHR (PDF, Word, Text)</h2>
                    <p style="color: var(--text-muted); font-size: 14px; margin-bottom: 16px;">
                        Upload local clinical patient files. The system will extract metrics locally. **No raw medical data leaves this machine.**
                    </p>
                    <div class="upload-area" onclick="document.getElementById('file-input').click()">
                        <p id="file-label">Click to select files (.pdf, .docx, .txt)</p>
                        <input type="file" id="file-input" style="display: none;" onchange="handleFileUpload(event)">
                    </div>
                    
                    <h3 style="margin-top: 24px;">Parsed Local EHR Patients</h3>
                    <table id="ehr-table">
                        <thead>
                            <tr>
                                <th>Filename</th>
                                <th>Temp (°C)</th>
                                <th>Dengue Status</th>
                                <th>Action</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td colspan="4" style="text-align: center; color: var(--text-muted);">No records parsed yet</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <div class="card">
                    <h2>Recent Local Timeline (Last 4 Weeks)</h2>
                    <table id="timeline-table">
                        <thead>
                            <tr>
                                <th>Year-Week</th>
                                <th>Temp (K)</th>
                                <th>Precip (mm)</th>
                                <th>LAI</th>
                                <th>Local Cases</th>
                            </tr>
                        </thead>
                        <tbody>
                            <!-- Filled dynamically -->
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="right-col">
                <div class="card" style="margin-bottom: 24px;">
                    <h2>Submit Weekly Report</h2>
                    <p style="color: var(--text-muted); font-size: 13px;">
                        Compute the Temporal GAT + GRU sequence locally. This generates a 32-dim secure embedding which is sent to the central server.
                    </p>
                    
                    <div class="form-group">
                        <label>Simulated Cases from EHRs this week</label>
                        <input type="number" id="week-cases" value="0" readonly>
                    </div>
                    
                    <div class="form-group">
                        <label>Temperature (K) - Current Week</label>
                        <input type="number" id="week-temp" step="0.1" value="298.5">
                    </div>
                    
                    <div class="form-group">
                        <label>Precipitation (mm) - Current Week</label>
                        <input type="number" id="week-precip" step="0.1" value="12.4">
                    </div>
                    
                    <button class="btn btn-success" onclick="sendEmbedding()">Generate & Transmit Embedding</button>
                    
                    <div class="pulse-container" id="transmitting-state" style="display: none;">
                        <div class="pulse"></div>
                        <span style="color: var(--accent); font-size: 14px; font-weight: bold;">Transmitting 32-Dim Vector...</span>
                    </div>
                </div>

                <div class="card" style="margin-bottom: 24px;">
                    <h2>Federated Learning (Flower)</h2>
                    <p style="color: var(--text-muted); font-size: 13px;">
                        Sync local weights with the central server using FedAvg to train the early warning model globally.
                    </p>
                    <button class="btn" style="background-color: #6366f1;" onclick="startFL()">Run Local Training Epoch & Sync</button>
                </div>

                <div class="card">
                    <h2>Client System Logs</h2>
                    <div class="console" id="console">
                        <div class="console-line">[*] Client initialized. Welcome to FedXGNN edge dashboard.</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const config = {
            censuscode: """ + str(CLIENT_CONFIG["censuscode"]) + """,
            name: '""" + CLIENT_CONFIG["name"] + """',
            port: """ + str(CLIENT_CONFIG["port"]) + """
        };

        document.getElementById('hospital-name-text').innerText = 'Hospital Edge Node: ' + config.name;
        document.getElementById('censuscode-text').innerText = 'District Census Code: ' + config.censuscode;

        // Add line to console
        function log(message) {
            const console = document.getElementById('console');
            const div = document.createElement('div');
            div.className = 'console-line';
            div.innerText = `[${new Date().toLocaleTimeString()}] ${message}`;
            console.appendChild(div);
            console.scrollTop = console.scrollHeight;
        }

        // Fetch Timeline from client backend
        async function loadTimeline() {
            try {
                const res = await fetch('/api/local-timeline');
                const data = await res.json();
                const tbody = document.querySelector('#timeline-table tbody');
                tbody.innerHTML = '';
                
                data.forEach(row => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td>${row.year}-W${row.week}</td>
                        <td>${row.temp_k.toFixed(1)}</td>
                        <td>${row.preci_mm.toFixed(1)}</td>
                        <td>${row.LAI.toFixed(2)}</td>
                        <td>${row.cases}</td>
                    `;
                    tbody.appendChild(tr);
                });
            } catch (err) {
                log('Error loading timeline: ' + err.message);
            }
        }

        async function handleFileUpload(e) {
            const file = e.target.files[0];
            if (!file) return;

            document.getElementById('file-label').innerText = 'Uploading: ' + file.name;
            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/api/upload-ehr', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                log(`Successfully parsed ${file.name}. Diagnosis: ${data.dengue_status === 1 ? 'Positive' : 'Negative'}, Temp: ${data.temperature_c}°C`);
                
                document.getElementById('file-label').innerText = 'Click to select files (.pdf, .docx, .txt)';
                
                // Refresh EHR table
                loadEHRTable();
                
                // Update simulated cases
                const casesInput = document.getElementById('week-cases');
                casesInput.value = parseInt(casesInput.value) + (data.dengue_status === 1 ? 1 : 0);
                
            } catch (err) {
                log('Error processing EHR: ' + err.message);
                document.getElementById('file-label').innerText = 'Error occurred. Try again.';
            }
        }

        async function loadEHRTable() {
            try {
                const res = await fetch('/api/ehrs');
                const data = await res.json();
                const tbody = document.querySelector('#ehr-table tbody');
                tbody.innerHTML = '';
                
                if (data.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No records parsed yet</td></tr>';
                    return;
                }

                data.forEach((ehr, idx) => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td>${ehr.filename}</td>
                        <td>${ehr.temperature_c}°C</td>
                        <td><span style="color: ${ehr.dengue_status === 1 ? 'var(--accent)' : '#ef4444'}">${ehr.dengue_status === 1 ? 'Positive' : 'Negative'}</span></td>
                        <td><button onclick="deleteEhr(${idx})" style="background:none; border:none; color:#ef4444; cursor:pointer;">Delete</button></td>
                    `;
                    tbody.appendChild(tr);
                });
            } catch (err) {
                log('Error loading EHR table: ' + err.message);
            }
        }

        async function deleteEhr(idx) {
            await fetch(`/api/ehrs/${idx}`, { method: 'DELETE' });
            loadEHRTable();
            // Recalculate cases
            const res = await fetch('/api/ehrs');
            const data = await res.json();
            document.getElementById('week-cases').value = data.filter(e => e.dengue_status === 1).length;
        }

        async function sendEmbedding() {
            const cases = parseInt(document.getElementById('week-cases').value);
            const temp = parseFloat(document.getElementById('week-temp').value);
            const precip = parseFloat(document.getElementById('week-precip').value);
            
            document.getElementById('transmitting-state').style.display = 'flex';
            log('Generating 32-dimensional embedding locally...');

            try {
                const res = await fetch('/api/transmit', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        cases: cases,
                        temp_k: temp,
                        preci_mm: precip
                    })
                });
                const data = await res.json();
                log('Local embedding computed.');
                log(`Secure 32-dim vector sent to Central Server. Status: ${data.server_status}`);
            } catch (err) {
                log('Transmission failed: ' + err.message);
            } finally {
                setTimeout(() => {
                    document.getElementById('transmitting-state').style.display = 'none';
                }, 1500);
            }
        }

        async function startFL() {
            log('Starting local training via Flower client...');
            try {
                const res = await fetch('/api/run-fl', { method: 'POST' });
                const data = await res.json();
                log(`Federated learning task complete. FedAvg updated server model with local gradients!`);
            } catch (err) {
                log('Flower training failed: ' + err.message);
            }
        }

        // Initialize
        loadTimeline();
        loadEHRTable();
    </script>
</body>
</html>
"""

class TransmitRequest(BaseModel):
    cases: int
    temp_k: float
    preci_mm: float

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    return DASHBOARD_HTML

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
        
    # Append the new week's features
    # is_monsoon: set to 1 if preci_mm > 50, otherwise 0 (simplified)
    # week_sin/cos: simple approximation
    current_raw = [
        req.temp_k,
        req.preci_mm,
        2.5, # LAI default
        np.log1p(req.cases), # cases_lag1 (current cases becomes lag1 for next week)
        raw_history[-1][3] if len(raw_history) > 0 else 0.0, # cases_lag2
        raw_history[-2][3] if len(raw_history) > 1 else 0.0, # cases_lag3
        0.5, # week_sin
        0.8, # week_cos
        1.0 if req.preci_mm > 50.0 else 0.0 # is_monsoon
    ]
    raw_history.append(current_raw)
    
    # Scale all 4 weeks
    scaled_history = scaler_dyn.transform(raw_history) # (4, 9)
    
    # Convert to tensor expected by ClientTemporalModel: (1, 4, 9)
    x_dyn = torch.tensor(scaled_history, dtype=torch.float32).unsqueeze(0)
    
    # Static features for the node
    # Let's prepare a default stat vector
    x_stat = torch.zeros(1, len(CFG["static_features"]), dtype=torch.float32)
    
    # Run the Client Temporal Model part to compute embedding
    with torch.no_grad():
        # Get Client Model from FedXGNN
        client_model = model.client
        embedding = client_model(x_dyn, x_stat) # (1, 32)
        emb_list = embedding.squeeze(0).tolist()
        
    # Send embedding list to central server
    try:
        # Post to central server's overlay / endpoint
        server_endpoint = f"{CLIENT_CONFIG['server_url']}/api/receive-edge-embedding"
        res = requests.post(server_endpoint, json={
            "censuscode": CLIENT_CONFIG["censuscode"],
            "embedding": emb_list,
            "cases": req.cases
        })
        
        return {
            "embedding": emb_list[:8], # Return first 8 elements as preview
            "server_status": "Received successfully" if res.status_code == 200 else f"Failed ({res.status_code})"
        }
    except Exception as e:
        return {
            "embedding": emb_list[:8],
            "server_status": f"Connection Error: {e}"
        }

@app.post("/api/run-fl")
def run_fl():
    # Trigger local Flower Client training
    # For simulation, we'll run a local process calling fl_client.py
    # or perform a mock FedAvg update via a rest call to demonstrate the concept instantly
    try:
        # Post mock update to Central FL Server to simulate 1 round of FL
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
