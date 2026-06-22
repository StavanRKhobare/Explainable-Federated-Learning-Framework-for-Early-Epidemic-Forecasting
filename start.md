# FedXGNN Epidemic Platform — Quick Start Guide

This guide provides instructions on how to start the central server, client hospital dashboards, and the central frontend web interface.

## Option 1: Automatic Startup (Recommended)

### Windows
1. Double-click the [start.bat](file:///c:/4th%20sem%20el/code/Explainable-Federated-Learning-Framework-for-Early-Epidemic-Forecasting/start.bat) file in the root directory.
   *OR*
2. Open Command Prompt or PowerShell in the root directory and run:
   ```cmd
   start.bat
   ```

### macOS / Linux
1. Open Terminal in the root directory and make the script executable:
   ```bash
   chmod +x start.sh
   ```
2. Run the script:
   ```bash
   ./start.sh
   ```

These scripts automate the process of:
* Creating a python virtual environment (`venv`).
* Installing requirements from [requirements.txt](file:///c:/4th%20sem%20el/code/Explainable-Federated-Learning-Framework-for-Early-Epidemic-Forecasting/requirements.txt).
* Downloading the required SpaCy model (`en_core_web_sm`).
* Spawning the Central Server, Bangalore Client, Coimbatore Client, Delhi Client, and Vite Frontend.

Once started, open your browser and navigate to:
* **Frontend Web Dashboard:** `http://localhost:5173` (or `http://localhost:3000`)

---

## Option 2: Manual Startup (Step-by-Step)

If you prefer to run services in separate terminal tabs manually:

### Step 0: Environment Setup (Once)
```bash
# Create and activate virtual environment
python -m venv venv
# On Windows:
call venv\Scripts\activate.bat
# On macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Set up frontend
cd frontend
npm install
cd ..
```

### Step 1: Start Central Server
```bash
# Windows: call venv\Scripts\activate.bat
# macOS/Linux: source venv/bin/activate
python backend/server.py
```
* **Endpoint:** `http://localhost:8000` (Wait ~5s for the boot sequence).

### Step 2: Start Clients
Open three separate terminal tabs, activate the `venv`, and run:
* **Bangalore Edge Client:**
  ```bash
  python client/client_app.py --port 8001 --censuscode 572 --name Bangalore
  ```
* **Coimbatore Edge Client:**
  ```bash
  python client/client_app.py --port 8002 --censuscode 632 --name Coimbatore
  ```
* **Delhi Edge Client:**
  ```bash
  python client/client_app.py --port 8003 --censuscode 94 --name Delhi
  ```

### Step 3: Start Frontend
```bash
cd frontend
npm run dev
```
* **Endpoint:** `http://localhost:5173`
