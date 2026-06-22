#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# FedXGNN Epidemic Platform — Linux / macOS Startup Script
# Usage: bash start.sh   OR   ./start.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs
VENV_DIR="$SCRIPT_DIR/venv"
PYTHON="$VENV_DIR/bin/python3"
PIP="$VENV_DIR/bin/pip"

# 1. Create venv if missing
if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# 2. Install / upgrade dependencies
echo "[*] Installing requirements..."
"$PIP" install --upgrade pip -q
"$PIP" install -r requirements.txt -q

# Download spaCy model if not present
"$PYTHON" -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null || \
    "$PYTHON" -m spacy download en_core_web_sm -q

echo ""
echo "════════════════════════════════════════════════════════"
echo "  FedXGNN Platform  —  Starting Services"
echo "════════════════════════════════════════════════════════"
echo "  Central Server   → http://localhost:8000"
echo "  Bangalore Client → http://localhost:8001"
echo "  Coimbatore Client→ http://localhost:8002"
echo "  New Delhi Client → http://localhost:8003"
echo "  Mysore Client    → http://localhost:8004"
echo "  Frontend (Vite)  → http://localhost:3000"
echo "════════════════════════════════════════════════════════"
echo ""

# Kill any existing instances to avoid port conflicts
pkill -f "client_app.py" 2>/dev/null || true
pkill -f "backend/server.py" 2>/dev/null || true
sleep 1

# 3. Launch central server (use explicit venv python so subshells keep the env)
"$PYTHON" backend/server.py > logs/server.log 2>&1 &
SERVER_PID=$!
echo "[*] Central server started (PID $SERVER_PID) — waiting for boot..."
sleep 6

# Verify server came up
if ! curl -s http://localhost:8000/api/model-info > /dev/null 2>&1; then
    echo "[!] WARNING: Server may still be loading. Continuing..."
fi

# 4. Launch edge clients (explicit venv python — avoids subshell venv loss)
"$PYTHON" -m client.client_app --port 8001 --censuscode 572 --name "Bangalore General Hospital" > logs/client_blr.log 2>&1 &
echo "[*] Bangalore client started (PID $!)"

sleep 2

"$PYTHON" -m client.client_app --port 8002 --censuscode 632 --name "Coimbatore Medical College" > logs/client_cbe.log 2>&1 &
echo "[*] Coimbatore client started (PID $!)"

sleep 2

"$PYTHON" -m client.client_app --port 8003 --censuscode 94 --name "New Delhi Hospital" > logs/client_del.log 2>&1 &
echo "[*] Delhi client started (PID $!)"

sleep 2

"$PYTHON" -m client.client_app --port 8004 --censuscode 577 --name "Mysore District Hospital" > logs/client_mys.log 2>&1 &
echo "[*] Mysore client started (PID $!)"

# 5. Start frontend
cd "$SCRIPT_DIR/frontend"
npm install -q --prefer-offline 2>/dev/null || npm install -q
npm run dev > "$SCRIPT_DIR/logs/frontend.log" 2>&1 &
echo "[*] Frontend started (PID $!)"
cd "$SCRIPT_DIR"

echo ""
echo "[✓] All services started. Open http://localhost:3000"
echo "    Logs: logs/server.log  logs/client_*.log  logs/frontend.log"
echo ""
echo "Press Ctrl+C to stop all services."

# Wait and keep script alive
wait
