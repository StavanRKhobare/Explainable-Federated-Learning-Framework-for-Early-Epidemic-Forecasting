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

---

## 📡 Multi-Laptop LAN Presentation Setup

To run a collaborative live demonstration using two laptops connected to the same Local Area Network (LAN):

### Laptop A (Main Server & Orchestration Node)
Laptop A runs the central database server, the React visualization interface, and two local edge clients (Bangalore, Coimbatore).

1. **Find Laptop A's LAN IP Address**:
   - Open a command prompt/terminal on Laptop A and run:
     - Windows: `ipconfig` (look for the IPv4 Address under your active Wi-Fi or Ethernet adapter, e.g., `192.168.1.10`)
     - Mac/Linux: `ifconfig` or `ip a`
   - Let's call this `<LaptopA_IP>`.

2. **Start the GNN Server**:
   ```bash
   venv\Scripts\python backend/server.py
   ```
   *Note: Automatically binds to `0.0.0.0:8000`, accepting incoming traffic from the LAN.*

3. **Start the React Frontend**:
   ```bash
   cd frontend
   npm run dev
   ```
   *Note: Automatically runs on `0.0.0.0:3000` (or `3001` if port 3000 is occupied) and is fully accessible across the LAN.*

4. **Start Laptop A's Local Client Nodes**:
   ```bash
   venv\Scripts\python client/client_app.py --port 8001 --censuscode 572 --name "Bangalore Hospital" --server http://localhost:8000
   venv\Scripts\python client/client_app.py --port 8002 --censuscode 632 --name "Coimbatore Hospital" --server http://localhost:8000
   ```

---

### Laptop B (Distributed Edge Node)
Laptop B runs the remaining edge client nodes (Delhi, Mysore) and connects remotely to Laptop A's server.

1. **Start the Edge Client Nodes**:
   Run the edge client applications, pointing the `--server` parameter to Laptop A's LAN IP address:
   ```bash
   venv\Scripts\python client/client_app.py --port 8003 --censuscode 94 --name "Delhi Hospital" --server http://<LaptopA_IP>:8000
   venv\Scripts\python client/client_app.py --port 8004 --censuscode 577 --name "Mysore Hospital" --server http://<LaptopA_IP>:8000
   ```

2. **Access the Central Interface from Laptop B**:
   Open a browser on Laptop B and navigate to `http://<LaptopA_IP>:3000` (or the corresponding Vite port, e.g., `3001`) to view the live dashboard.

3. **Inbound Port Access & Firewalls**:
   - Ensure both laptops are connected to the same network profile, and set the connection profile to **Private** (Windows blocks incoming local traffic on Public network profiles).
   - If Laptop B gets a connection timeout connecting to `http://<LaptopA_IP>:8000`, open **Windows Defender Firewall** on Laptop A and add an **Inbound Rule** to allow TCP traffic on ports `8000` and `3000`.
