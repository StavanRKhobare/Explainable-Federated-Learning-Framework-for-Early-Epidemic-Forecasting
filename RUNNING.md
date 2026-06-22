# FedXGNN Platform — Execution & Verification Guide

This guide provides step-by-step instructions to boot, verify, and interact with the FedXGNN split-federated pipeline, including the GNN server, the edge client apps, and the React visualization interface.

---

## Quick Start (Recommended Option)

Launch the central server, the three default edge client nodes (Bangalore, Coimbatore, New Delhi), and the React UI dashboard concurrently using the provided orchestration script:

```bash
cd "Explainable-Federated-Learning-Framework-for-Early-Epidemic-Forecasting"
# Give execution permissions if not set
chmod +x start.sh
# Start all microservices in parallel
./start.sh
```

Once running, access the central dashboard at **http://localhost:3000** and the client portals at:
*   **Bangalore General Hospital**: http://localhost:8001
*   **Coimbatore Medical College**: http://localhost:8002
*   **New Delhi Hospital**: http://localhost:8003

To stop all background tasks, press `Ctrl+C` in the terminal.

---

## Step-by-Step Manual Orchestration

If you prefer to launch and monitor each microservice individually, run these commands in separate terminal sessions:

### Step 0: Environment Provisioning (One-time)
Setup the virtual environment, install dependencies, download the spaCy NLP entity model, and compile the frontend packages:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
cd frontend
npm install
cd ..
```

### Step 1: Start Central GNN Server (Terminal 1)
Launch the server first to initialize the district graph state and load the model checkpoints:
```bash
source venv/bin/activate
python3 backend/server.py
```
*   **Server URL**: http://localhost:8000
*   *Verification*: Wait for the terminal to print `[BOOT] Ready — 284 districts, 728 timesteps`.

### Step 2: Start Bangalore Edge Client (Terminal 2)
```bash
source venv/bin/activate
python3 -m client.client_app --port 8001 --censuscode 572 --name "Bangalore General Hospital"
```
*   **Bangalore Portal**: http://localhost:8001

### Step 3: Start Coimbatore Edge Client (Terminal 3)
```bash
source venv/bin/activate
python3 -m client.client_app --port 8002 --censuscode 632 --name "Coimbatore Medical College"
```
*   **Coimbatore Portal**: http://localhost:8002

### Step 4: Start New Delhi Edge Client (Terminal 4)
```bash
source venv/bin/activate
python3 -m client.client_app --port 8003 --censuscode 94 --name "New Delhi Hospital"
```
*   **New Delhi Portal**: http://localhost:8003

### Step 5: Start Vite React Frontend (Terminal 5)
```bash
cd frontend
npm run dev
```
*   **Dashboard URL**: http://localhost:3000

---

## Verification and Interaction Workflows

### 1. Verification of Clinical EHR Parsing & NLP Extraction
*   Open the Bangalore Edge Portal (http://localhost:8001).
*   Locate the **EHR Intake Form Ingestion** card.
*   Drag and drop the sample patient file located at `ehr_samples/patient_dengue_pos_99F.txt` into the upload container.
*   Observe the terminal output logging the matched entity count and the calculated clinical symptom score.
*   Verify that the local timeline table updates to show the positive symptom count.

### 2. Client-to-Server Embedding Transmission
*   Click the **Transmit Embedding** button on the Client Portal.
*   Open the main dashboard at http://localhost:3000.
*   Confirm that the active client list displays the corresponding node port status as **Active** (marked in green).
*   Select the district on the India map to verify that the outbreak probability scales correctly and matches the values reported on the client node.

### 3. Verification of legibility in XAI Heatmap
*   On the main React Dashboard, select a district (e.g., Bangalore or Coimbatore).
*   Inspect the **Temporal SHAP Interpretability** heatmap.
*   Verify that cell colors map clearly to positive/negative values and that the feature importance numbers exhibit clear, readable scales ($>0.01$) due to historical background profiling.

---

## Troubleshooting Guide

### Port Conflicts
If a port is already bound, free it using `fuser`:
```bash
fuser -k 8000/tcp  # Central Server
fuser -k 8001/tcp  # Bangalore Node
fuser -k 8002/tcp  # Coimbatore Node
fuser -k 8003/tcp  # Delhi Node
fuser -k 3000/tcp  # React Frontend
```

### Logs Inspection
Review real-time log outputs:
```bash
tail -n 50 logs/server.log        # Server output
tail -n 50 logs/client_blr.log    # Bangalore client output
tail -n 50 logs/client_cbe.log    # Coimbatore client output
tail -n 50 logs/client_del.log    # Delhi client output
```
