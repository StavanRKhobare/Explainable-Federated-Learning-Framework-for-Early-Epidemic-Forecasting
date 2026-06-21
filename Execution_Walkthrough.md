# Walkthrough - Explainable Federated Learning Framework

We have successfully implemented the full end-to-end architecture for the **Explainable Federated Learning Framework for Early Epidemic Forecasting**. All 6 phases have been completed, matching the visual styles, color palettes, and operational paradigms requested.

---

## 🚀 Key Accomplishments & Implementations

### 1. Edge EHR Extraction Module (`client/ehr_parser.py`)
- **Privacy-Preserving Parsing**: An offline rule-based NLP parser that reads PDF, DOCX, and TXT medical documents locally.
- **Entity Extraction**: Automatically extracts core patient metrics (Temperature, Symptoms, Diagnoses, Age, Gender, Date) using structured regex and pattern matching. 
- **Zero Data Leakage**: No raw medical text or files leave the local edge hospital node.

### 2. Hospital Edge Node Dashboard (`client/client_app.py`)
- **FastAPI Edge UI**: A complete web interface running at `http://localhost:8001` that lets hospital workers upload clinical files and view a timeline of past weather/case stats.
- **Cohesive Color Palette & Tone**: Re-styled to use the exact same light theme, colors (`var(--slate-50)` background, white cards with subtle borders), typography, and button styles as the central React dashboard.
- **Temporal GAT Encoding**: Processes the local 4-week lookback timeline, computes a 32-dimensional privacy-preserving embedding, and transmits it via a secure REST connection to the central server.

### 3. Federated Learning & Weight Synchronization (`client/fl_client.py` & `backend/fl_server.py`)
- **Flower FL Infrastructure**: Implemented NumPy Flower clients and aggregation server rules.
- **FedAvg Strategy**: Aggregates the local temporal weights of the hospitals without centralizing patient records, making the models smarter over time.
- **Sync Telemetry**: Exposed `/api/fl-sync` for instant web-based demo synchronization of local weights.

### 4. Central Spatial GNN Inference & Live Overrides (`backend/server.py`)
- **Live Transmission Buffer**: Receives edge embeddings and overrides default historical nodes in the spatial graph in real-time.
- **Interactive Graph Update**: Re-evaluates risk predictions dynamically for the entire graph when an edge hospital uploads new data.
- **XAI API exposure**: Added `/api/xai/temporal` and `/api/xai/spatial` endpoints.

### 5. Explainable AI & Frontend Visualization (`frontend/src/pages/`)
- **`LivePredict.jsx` (SHAP & GNNExplainer)**:
  - Added **Temporal SHAP Feature Importance**: Computes and renders a horizontal bar chart showing how features (Temp, Rainfall, Past Cases) contributed to the risk prediction.
  - Added **Spatial GNNExplainer Neighbor Influence**: Lists neighboring districts and their spatial weights, explaining GAT disease propagation pathways.
- **`FederatedDemo.jsx` (Live Client Status)**:
  - Added an **Active Edge Clients Panel** that polls connected hospitals in real-time, flashing a green pulse when a node (like Bangalore or Chennai) connects and transmits.

### 6. Containerization (`Dockerfile.server`, `Dockerfile.client`, & `docker-compose.yml`)
- **Optimized Python Images**: Uses `python:3.10-slim` with precompiled CPU PyTorch and PyTorch Geometric wheels to ensure builds complete under 2 minutes.
- **Multi-Node Orchestration**: Starts the Central Backend Server and two hospital edge nodes (Bangalore and Chennai) simultaneously.

---

## 🛠️ How to Run & Demo

### Option A: Running with Docker Compose (Recommended)
Launch the entire distributed system (1 backend + 2 hospital nodes) in one command:
```bash
docker-compose up --build
```
- **Central GNN Dashboard**: `http://localhost:3001` (if running frontend locally via npm)
- **Central Backend Server**: `http://localhost:8000`
- **Bangalore Hospital Edge**: `http://localhost:8001`
- **Chennai Hospital Edge**: `http://localhost:8002`

### Option B: Running Locally (Manual Terminal Setup)

1. **Activate the Environment & Install Dependencies**:
   ```bash
   source venv/bin/activate
   pip install python-multipart
   ```

2. **Start the GNN Backend Server**:
   ```bash
   python backend/server.py
   ```

3. **Start the Hospital Edge Node(s)**:
   - **Bangalore**:
     ```bash
     python client/client_app.py --port 8001 --censuscode 572 --name "Bangalore General Hospital"
     ```
   - **Chennai**:
     ```bash
     python client/client_app.py --port 8002 --censuscode 632 --name "Chennai Medical College"
     ```

4. **Start the React Frontend**:
   ```bash
   cd frontend
   npm run dev
   ```

5. **Generate Sample EHRs for Testing**:
   ```bash
   python client/generate_samples.py
   ```
   *This outputs rule-based `.txt` and `.docx` clinical files in the `ehr_samples/` directory which you can upload on the hospital dashboards.*
