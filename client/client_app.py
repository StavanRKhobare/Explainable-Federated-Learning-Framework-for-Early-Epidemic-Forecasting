import os
import sys
import argparse
import uvicorn
import torch
import numpy as np
import pandas as pd
import requests
import json
import sqlite3
import datetime
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
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

def get_db_path(censuscode):
    return os.path.join(PROJECT_ROOT, "client", f"client_{censuscode}.db")

def init_client_db(censuscode, local_df):
    db_path = get_db_path(censuscode)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS local_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER,
            week INTEGER,
            temp_k REAL,
            preci_mm REAL,
            LAI REAL,
            cases INTEGER,
            UNIQUE(year, week)
        )
    """)
    conn.commit()
    
    # Check if empty, and pre-populate from local_df
    cursor.execute("SELECT COUNT(*) FROM local_history")
    count = cursor.fetchone()[0]
    if count == 0:
        print(f"[*] Pre-populating local database client_{censuscode}.db with historical CSV data...")
        records = local_df.to_dict(orient="records")
        for r in records:
            c_val = int(np.expm1(r["cases"])) if "cases" in r else 0
            cursor.execute("""
                INSERT OR IGNORE INTO local_history (year, week, temp_k, preci_mm, LAI, cases)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (int(r["iso_year"]), int(r["iso_week"]), float(r["temp_k"]), float(r["preci_mm"]), float(r["LAI"] or 2.5), c_val))
        conn.commit()
    conn.close()

def log_local_history(censuscode, year, week, temp_k, preci_mm, LAI, cases):
    db_path = get_db_path(censuscode)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO local_history (year, week, temp_k, preci_mm, LAI, cases)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(year, week) DO UPDATE SET
            temp_k = excluded.temp_k,
            preci_mm = excluded.preci_mm,
            LAI = excluded.LAI,
            cases = excluded.cases
    """, (year, week, temp_k, preci_mm, LAI, cases))
    conn.commit()
    conn.close()

def fetch_local_history(censuscode, limit=20):
    db_path = get_db_path(censuscode)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM local_history ORDER BY year ASC, week ASC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows][-limit:]

def get_analytics_db_path(censuscode):
    return os.path.join(PROJECT_ROOT, "client", f"analytics_{censuscode}.db")

SYMPTOMS_LIST = ["fever", "rash", "joint pain", "muscle pain", "headache", "retro-orbital pain", "nausea", "vomiting", "fatigue"]

def extract_symptoms_from_text(text: str):
    text_lower = text.lower()
    found = []
    for s in SYMPTOMS_LIST:
        if s in text_lower:
            found.append(s)
        elif s == "muscle pain" and "myalgia" in text_lower:
            found.append("muscle pain")
        elif s == "joint pain" and "arthralgia" in text_lower:
            found.append("joint pain")
    return found

def init_analytics_db(censuscode):
    db_path = get_analytics_db_path(censuscode)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            temperature_c REAL,
            dengue_status INTEGER,
            symptoms TEXT, -- JSON string array
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    
    # Check if empty, and pre-populate with mock patient records
    cursor.execute("SELECT COUNT(*) FROM patient_records")
    count = cursor.fetchone()[0]
    if count == 0:
        print(f"[*] Pre-populating analytics database analytics_{censuscode}.db with mock patient records...")
        import random
        # Seed dynamically to keep dataset distinct per city node
        random.seed(censuscode)
        
        for i in range(1, 31):
            filename = f"EHR_Patient_{100 + i}.pdf"
            # Positive, negative, unknown rolls
            if censuscode in (572, 94):
                status_roll = random.random()
                dengue_status = 1 if status_roll < 0.35 else (0 if status_roll < 0.9 else -1)
            else:
                status_roll = random.random()
                dengue_status = 1 if status_roll < 0.2 else (0 if status_roll < 0.9 else -1)
                
            # Formulate clinical profiles
            if dengue_status == 1:
                temperature_c = round(random.uniform(38.0, 40.2), 2)
                symptom_pool = ["fever", "rash", "joint pain", "muscle pain", "headache", "retro-orbital pain"]
                k = random.randint(2, 4)
                symptoms = list(set(["fever"] + random.sample(symptom_pool, k)))
            elif dengue_status == 0:
                temperature_c = round(random.uniform(36.4, 37.7), 2)
                symptom_pool = ["fatigue", "nausea", "headache"]
                if random.random() < 0.4:
                    symptoms = random.sample(symptom_pool, random.randint(1, 2))
                else:
                    symptoms = []
            else:
                temperature_c = round(random.uniform(37.2, 38.9), 2)
                symptom_pool = ["fever", "fatigue", "nausea", "vomiting"]
                symptoms = list(set(["fever"] + random.sample(symptom_pool, random.randint(0, 2))))
                
            cursor.execute("""
                INSERT INTO patient_records (filename, temperature_c, dengue_status, symptoms)
                VALUES (?, ?, ?, ?)
            """, (filename, temperature_c, dengue_status, json.dumps(symptoms)))
            
        conn.commit()
    conn.close()

def load_uploaded_cases_from_db(censuscode):
    global uploaded_cases
    db_path = get_analytics_db_path(censuscode)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patient_records ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    
    uploaded_cases = []
    for r in rows:
        uploaded_cases.append({
            "id": r["id"],
            "filename": r["filename"],
            "temperature_c": r["temperature_c"],
            "dengue_status": r["dengue_status"],
            "symptoms": json.loads(r["symptoms"] or "[]")
        })

def build_raw_history_from_db(censuscode):
    db_path = get_db_path(censuscode)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM local_history ORDER BY year ASC, week ASC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()

    if len(rows) < 4:
        return None
        
    last_4_indices = range(len(rows) - 4, len(rows))
    raw_history = []
    
    for idx in last_4_indices:
        r = rows[idx]
        
        # Get cases for w-1, w-2, w-3
        c_lag1 = rows[idx - 1]["cases"] if idx - 1 >= 0 else 0
        c_lag2 = rows[idx - 2]["cases"] if idx - 2 >= 0 else 0
        c_lag3 = rows[idx - 3]["cases"] if idx - 3 >= 0 else 0
        
        log_lag1 = float(np.log1p(max(c_lag1, 0)))
        log_lag2 = float(np.log1p(max(c_lag2, 0)))
        log_lag3 = float(np.log1p(max(c_lag3, 0)))
        
        import math
        w_val = r["week"]
        week_sin = math.sin(2 * math.pi * w_val / 52.0)
        week_cos = math.cos(2 * math.pi * w_val / 52.0)
        
        is_monsoon = 1.0 if r["preci_mm"] > 50.0 else 0.0
        
        is_latest = (idx == len(rows) - 1)
        symptoms = float(sum(1 for e in uploaded_cases if e.get('dengue_status', -1) == 1)) if is_latest else 0.0
        diseases = float(len(uploaded_cases)) if is_latest else 0.0
        total_notes = float(len(uploaded_cases)) if is_latest else 0.0
        
        raw_row = [
            float(r["temp_k"]),
            float(r["preci_mm"]),
            float(r["LAI"] or 2.5),
            log_lag1,
            log_lag2,
            log_lag3,
            float(week_sin),
            float(week_cos),
            float(is_monsoon),
            symptoms,
            diseases,
            0.0,
            0.0,
            total_notes
        ]
        raw_history.append(raw_row)
        
    return raw_history

def init_client(censuscode, server_url):
    global model, local_df, scaler_dyn, scaler_stat, avail_dyn
    
    # 1. Load full dataset to fit scalers exactly like backend to avoid mismatch
    data_path = os.path.join(PROJECT_ROOT, "data", "training_dataset_with_ner.csv")
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
    init_client_db(censuscode, local_df)
    init_analytics_db(censuscode)
    load_uploaded_cases_from_db(censuscode)
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
    # Return the last 20 weeks of records from local SQLite DB for a better timeline plot
    try:
        records = fetch_local_history(CLIENT_CONFIG["censuscode"], limit=20)
        out = []
        for r in records:
            out.append({
                "year": int(r["year"]),
                "week": int(r["week"]),
                "temp_k": float(r["temp_k"]),
                "preci_mm": float(r["preci_mm"]),
                "LAI": float(r["LAI"]),
                "cases": int(r["cases"])
            })
        return out
    except Exception as e:
        print(f"[DB ERROR] fetch timeline: {e}")
        # Fallback to local_df if DB fails
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
    symptoms = extract_symptoms_from_text(text)
    
    # Remove file after parsing
    try:
        os.remove(file_path)
    except:
        pass
        
    # Save to DB
    db_path = get_analytics_db_path(CLIENT_CONFIG["censuscode"])
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO patient_records (filename, temperature_c, dengue_status, symptoms)
        VALUES (?, ?, ?, ?)
    """, (file.filename, parsed["temperature_c"], parsed["dengue_status"], json.dumps(symptoms)))
    conn.commit()
    conn.close()
    
    load_uploaded_cases_from_db(CLIENT_CONFIG["censuscode"])
    return {
        "filename": file.filename,
        "temperature_c": parsed["temperature_c"],
        "dengue_status": parsed["dengue_status"],
        "symptoms": symptoms
    }

@app.get("/api/ehrs")
def get_ehrs():
    return uploaded_cases

@app.delete("/api/ehrs/{idx}")
def delete_ehr(idx: int):
    global uploaded_cases
    if 0 <= idx < len(uploaded_cases):
        record = uploaded_cases[idx]
        record_id = record.get("id")
        
        db_path = get_analytics_db_path(CLIENT_CONFIG["censuscode"])
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM patient_records WHERE id = ?", (record_id,))
        conn.commit()
        conn.close()
        
        load_uploaded_cases_from_db(CLIENT_CONFIG["censuscode"])
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
        
        db_path = get_analytics_db_path(CLIENT_CONFIG["censuscode"])
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
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
                
            symptom_str = row.get("symptoms", row.get("symptom", "")).strip()
            symptom_list = []
            if symptom_str:
                symptom_list = [s.strip().lower() for s in symptom_str.split(",") if s.strip().lower() in SYMPTOMS_LIST]
            else:
                # Realistic clinical generation
                if dengue_status == 1:
                    symptom_list = ["fever", "joint pain"] if temp_c > 38.0 else ["fever"]
                elif dengue_status == -1:
                    symptom_list = ["fever"] if temp_c > 37.5 else []
                    
            cursor.execute("""
                INSERT INTO patient_records (filename, temperature_c, dengue_status, symptoms)
                VALUES (?, ?, ?, ?)
            """, (f"{file.filename}:row{total}", round(temp_c, 2), dengue_status, json.dumps(symptom_list)))
            
        conn.commit()
        conn.close()
        
        load_uploaded_cases_from_db(CLIENT_CONFIG["censuscode"])
        return {"status": "ok", "total": total, "positive": positive, "negative": negative}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/analytics-stats")
def get_analytics_stats():
    db_path = get_analytics_db_path(CLIENT_CONFIG["censuscode"])
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT dengue_status, temperature_c, symptoms FROM patient_records")
    rows = cursor.fetchall()
    conn.close()
    
    status_counts = {"positive": 0, "negative": 0, "unknown": 0}
    temp_counts = {"febrile": 0, "afebrile": 0}
    symptom_counts = {s: 0 for s in SYMPTOMS_LIST}
    
    for r in rows:
        status = r["dengue_status"]
        if status == 1: status_counts["positive"] += 1
        elif status == 0: status_counts["negative"] += 1
        else: status_counts["unknown"] += 1
        
        temp = r["temperature_c"]
        if temp >= 37.5: temp_counts["febrile"] += 1
        else: temp_counts["afebrile"] += 1
        
        try:
            syms = json.loads(r["symptoms"] or "[]")
            for s in syms:
                if s in symptom_counts:
                    symptom_counts[s] += 1
        except:
            pass
            
    return {
        "total_records": len(rows),
        "status_counts": status_counts,
        "temp_counts": temp_counts,
        "symptom_counts": symptom_counts
    }

class AISummaryRequest(BaseModel):
    api_key: Optional[str] = None

@app.post("/api/ai-summary")
def get_ai_summary(req: AISummaryRequest):
    stats = get_analytics_stats()
    total = stats["total_records"]
    pos = stats["status_counts"]["positive"]
    neg = stats["status_counts"]["negative"]
    unk = stats["status_counts"]["unknown"]
    feb = stats["temp_counts"]["febrile"]
    afe = stats["temp_counts"]["afebrile"]
    syms = stats["symptom_counts"]
    
    hospital_name = CLIENT_CONFIG["name"]
    district_code = CLIENT_CONFIG["censuscode"]
    
    sorted_syms = sorted(syms.items(), key=lambda x: x[1], reverse=True)
    symptom_text = ", ".join([f"{k} ({v} cases)" for k, v in sorted_syms if v > 0])
    
    prompt = (
        f"Analyze this local clinical intelligence report for {hospital_name} (District Code: {district_code}):\n"
        f"- Total Patients Processed: {total}\n"
        f"- Dengue Positive Cases: {pos} ({pos/max(total,1)*100:.1f}% positivity rate)\n"
        f"- Dengue Negative Cases: {neg}\n"
        f"- Suspected/Unknown Cases: {unk}\n"
        f"- Febrile Patients (Temp >= 37.5°C): {feb} ({feb/max(total,1)*100:.1f}%)\n"
        f"- Afebrile Patients (Temp < 37.5°C): {afe}\n"
        f"- Extracted Symptom Breakdown: {symptom_text}\n\n"
        "Write a brief, highly professional Epidemic Intelligence Summary in Markdown format. "
        "Include: 1. Clinical Status Assessment, 2. Symptom Trend Analysis, 3. Immediate Epidemiological Recommendations. "
        "Keep the tone scientific, brief, and structured for medical practitioners."
    )
    
    def generate_mock_report():
        pos_rate = (pos/max(total, 1)) * 100
        fever_rate = (feb/max(total, 1)) * 100
        alert_level = "ELEVATED" if pos_rate > 20 else "STABLE"
        if pos_rate > 40: alert_level = "CRITICAL OUTBREAK WARNING"
        
        return (
            f"### 📋 Clinician AI Intelligence Report — {hospital_name}\n"
            f"**District Census ID:** {district_code} | **Status:** `{alert_level}`\n\n"
            f"#### 1. Clinical Status Assessment\n"
            f"The node has processed a total of **{total} electronic health records (EHRs)**. "
            f"We detect **{pos} confirmed positive cases** of Dengue infection, representing a **{pos_rate:.1f}% positivity rate**. "
            f"In addition, **{unk} cases** present borderline symptoms requiring active monitoring. "
            f"The high proportion of febrile patients (**{fever_rate:.1f}%** presenting temperature &ge; 37.5°C) correlates strongly with local vector-borne transmission risk.\n\n"
            f"#### 2. Symptom Trend Analysis\n"
            f"A semantic analysis of the unstructured clinical notes reveals the following primary symptom distribution:\n"
            + "\n".join([f"- **{k.capitalize()}**: Detected in `{v}` patient notes." for k, v in sorted_syms[:4] if v > 0]) +
            f"\n\n*Clinical Note:* The co-occurrence of febrile temperatures with joint and muscle pain (classic 'breakbone' dengue markers) indicates an active local transmission cycle in the area.\n\n"
            f"#### 3. Immediate Epidemiological Recommendations\n"
            f"1. **Active Local Ingestion**: Increase surveillance frequency for incoming febrile patients presenting with retro-orbital pain or rashes.\n"
            f"2. **Mitigation Trigger**: Mobilize municipal vector-control teams for larvicidal spraying in adjacent clusters of District {district_code}.\n"
            f"3. **Federated Synchronization**: Transmit the encrypted 64-dim edge embedding vector to the central server immediately to update the spatial spillover predictions for neighboring districts.\n\n"
            f"*(Note: Enter a valid Groq API Key in the panel to generate a live customized report using Llama-3).* "
        )

    if not req.api_key or not req.api_key.strip():
        return {"summary": generate_mock_report()}
        
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {req.api_key.strip()}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3-8b-8192",
            "messages": [
                {
                    "role": "system", 
                    "content": (
                        "You are an expert AI clinical epidemiologist assistant. Analyze local hospital stats "
                        "and generate a brief, professional medical markdown intelligence report."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=8)
        if resp.status_code == 200:
            return {"summary": resp.json()["choices"][0]["message"]["content"]}
        else:
            return {"summary": generate_mock_report() + f"\n\n*(Error calling Groq API: HTTP {resp.status_code}. Loaded dynamic mock report instead.)*"}
    except Exception as e:
        return {"summary": generate_mock_report() + f"\n\n*(Connection error calling Groq: {e}. Loaded dynamic mock report instead.)*"}


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
    # Log the new weekly data to SQLite first to preserve local case history
    year = req.year
    week = req.week
    if not year or not week:
        try:
            clk = get_sim_clock()
            year = clk.get("year", 2024)
            week = clk.get("week", 1)
        except:
            year = 2024
            week = 1

    log_local_history(
        censuscode=CLIENT_CONFIG["censuscode"],
        year=year,
        week=week,
        temp_k=req.temp_k,
        preci_mm=req.preci_mm,
        LAI=2.5,
        cases=req.cases
    )

    # Prepare dynamic input features (4-week lookback) from SQLite DB
    raw_history = build_raw_history_from_db(CLIENT_CONFIG["censuscode"])
    
    # Fallback to local_df static data if database is empty/corrupted
    if not raw_history:
        print("[*] SQLite history incomplete. Falling back to local_df...")
        last_3 = local_df.tail(3).copy()
        raw_history = []
        for row in last_3.itertuples():
            raw_row = []
            for feat in CFG["dynamic_features"]:
                raw_row.append(getattr(row, feat, 0.0))
            raw_history.append(raw_row)
            
        is_monsoon = 1.0 if req.preci_mm > 50.0 else 0.0
        current_raw = [
            req.temp_k,
            req.preci_mm,
            2.5,
            float(np.log1p(max(req.cases, 0))),
            float(raw_history[-1][3]) if len(raw_history) > 0 else 0.0,
            float(raw_history[-2][3]) if len(raw_history) > 1 else 0.0,
            0.5,
            0.8,
            is_monsoon,
            float(sum(1 for e in uploaded_cases if e.get('dengue_status', -1) == 1)),
            float(len(uploaded_cases)),
            0.0,
            0.0,
            float(len(uploaded_cases)),
        ]
        raw_history.append(current_raw)

    # Verify we have exactly 14 features matching the model
    n_feats = len(CFG["dynamic_features"])
    for i, row in enumerate(raw_history):
        if len(row) < n_feats:
            raw_history[i] = row + [0.0] * (n_feats - len(row))
        raw_history[i] = raw_history[i][:n_feats]
    
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

    # Run Client Temporal Model -> produces 64-dim embedding
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
    """Generate and stream a premium PDF report using the AeroSmart/EpiGraph AI design."""
    from fastapi.responses import StreamingResponse
    from client.report_generator import generate_report

    try:
        ep  = f"{CLIENT_CONFIG['server_url']}/api/report-data/{CLIENT_CONFIG['censuscode']}"
        res = requests.get(ep, timeout=8)
        if res.status_code != 200:
            raise HTTPException(status_code=500, detail="Could not fetch report data")
        data = res.json()

        # Enrich data with hospital name for the PDF header
        data["hospital"] = CLIENT_CONFIG["name"]

        # Normalise uploaded_cases: convert int dengue_status to string for the renderer
        cases_for_pdf = [
            {
                "filename":      e.get("filename", ""),
                "temperature_c": e.get("temperature_c", 0),
                "dengue_status": (
                    "Positive" if e.get("dengue_status") == 1
                    else "Negative" if e.get("dengue_status") == 0
                    else "Unknown"
                ),
            }
            for e in uploaded_cases
        ]

        buf = generate_report(data, cases_for_pdf)

        district  = data.get("district", "District").replace(" ", "_").replace("/", "-")
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M")
        filename  = f"FedXGNN_EpiReport_{district}_{timestamp}.pdf"

        return StreamingResponse(
            buf,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")


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
