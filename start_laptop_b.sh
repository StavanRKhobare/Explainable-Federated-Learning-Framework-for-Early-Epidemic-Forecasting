#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# FedXGNN Platform — Laptop B (Linux Client) Orchestration Script
# ─────────────────────────────────────────────────────────────────────────────

# Default server host
SERVER_HOST="Rohith_lappy.local"

# If a custom IP/host is provided as the first argument, use it
if [ ! -z "$1" ]; then
    SERVER_HOST="$1"
fi

SERVER_URL="http://${SERVER_HOST}:8000"

echo "════════════════════════════════════════════════════════"
echo "  FedXGNN Platform — Starting Laptop B Edge Clients"
echo "  Targeting Server: ${SERVER_URL}"
echo "════════════════════════════════════════════════════════"

# 1. Activate venv / check environment
if [ -f "venv/bin/activate" ]; then
    echo "[*] Activating virtual environment..."
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    echo "[*] Activating virtual environment (.venv)..."
    source .venv/bin/activate
else
    echo "[WARNING] Virtual environment 'venv' not found. Trying global python3."
fi

# Make sure logs directory exists
mkdir -p logs

# 2. Start Coimbatore Edge Client (Port 8002)
echo "[*] Launching Coimbatore Hospital Client on port 8002..."
python3 client/client_app.py --port 8002 --censuscode 632 --name "Coimbatore Hospital" --server "${SERVER_URL}" > logs/client_cbe.log 2>&1 &
CBE_PID=$!

# 3. Start Mysore Edge Client (Port 8004)
echo "[*] Launching Mysore Hospital Client on port 8004..."
python3 client/client_app.py --port 8004 --censuscode 577 --name "Mysore Hospital" --server "${SERVER_URL}" > logs/client_mys.log 2>&1 &
MYS_PID=$!

echo "════════════════════════════════════════════════════════"
echo "  [SUCCESS] Laptop B edge clients started in background!"
echo "  - Coimbatore (PID: $CBE_PID) -> Logs: logs/client_cbe.log"
echo "  - Mysore     (PID: $MYS_PID) -> Logs: logs/client_mys.log"
echo "════════════════════════════════════════════════════════"
echo "  To stop these clients, run: ./stop_laptop_b.sh"
echo "  Or run: kill $CBE_PID $MYS_PID"
echo "════════════════════════════════════════════════════════"
