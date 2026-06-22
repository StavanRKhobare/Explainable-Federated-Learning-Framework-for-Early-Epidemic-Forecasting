#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# FedXGNN Platform — Stop Laptop B Edge Clients
# ─────────────────────────────────────────────────────────────────────────────

echo "[*] Stopping Coimbatore and Mysore edge clients..."
fuser -k 8002/tcp 8004/tcp 2>/dev/null || true
pkill -f "client_app.py --port 8002" || true
pkill -f "client_app.py --port 8004" || true
echo "[✓] Edge clients stopped."
