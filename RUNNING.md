# FedXGNN Platform — How to Run

> Open each section in a **separate terminal tab**, in order.

---

## Option A — One Command (Recommended)

```bash
cd "Explainable-Federated-Learning-Framework-for-Early-Epidemic-Forecasting"
./start.sh
```

Then open **http://localhost:3000** in your browser.

---

## Option B — Run Each Service Manually

### Step 0 — Initial Setup (only once)

```bash
cd "/home/stavan-khobare/Desktop/4th sem 3rd phase/We Ball Antigravity/Explainable-Federated-Learning-Framework-for-Early-Epidemic-Forecasting"
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python3 -m spacy download en_core_web_sm
cd frontend && npm install && cd ..
```

---

### Terminal 1 — Central Server *(start this FIRST, wait ~5s)*

```bash
cd "/home/stavan-khobare/Desktop/4th sem 3rd phase/We Ball Antigravity/Explainable-Federated-Learning-Framework-for-Early-Epidemic-Forecasting"
venv/bin/python3 backend/server.py
```

✅ **http://localhost:8000** — wait for `[BOOT] Ready — X districts`

---

### Terminal 2 — Bangalore Edge Client

```bash
cd "/home/stavan-khobare/Desktop/4th sem 3rd phase/We Ball Antigravity/Explainable-Federated-Learning-Framework-for-Early-Epidemic-Forecasting"
venv/bin/python3 -m client.client_app --port 8001 --censuscode 572 --name "Bangalore General Hospital"
```

✅ **http://localhost:8001**

---

### Terminal 3 — Coimbatore Edge Client

```bash
cd "/home/stavan-khobare/Desktop/4th sem 3rd phase/We Ball Antigravity/Explainable-Federated-Learning-Framework-for-Early-Epidemic-Forecasting"
venv/bin/python3 -m client.client_app --port 8002 --censuscode 632 --name "Coimbatore Medical College"
```

✅ **http://localhost:8002**

---

### Terminal 4 — Delhi Edge Client

```bash
cd "/home/stavan-khobare/Desktop/4th sem 3rd phase/We Ball Antigravity/Explainable-Federated-Learning-Framework-for-Early-Epidemic-Forecasting"
venv/bin/python3 -m client.client_app --port 8003 --censuscode 94 --name "New Delhi Hospital"
```

✅ **http://localhost:8003**

---

### Terminal 5 — Frontend (React/Vite)

```bash
cd "/home/stavan-khobare/Desktop/4th sem 3rd phase/We Ball Antigravity/Explainable-Federated-Learning-Framework-for-Early-Epidemic-Forecasting/frontend"
npm run dev
```

✅ **http://localhost:3000** — Main Dashboard

---

## Checking Logs (if something crashes)

```bash
tail -30 logs/server.log       # Central server
tail -30 logs/client_blr.log   # Bangalore
tail -30 logs/client_cbe.log   # Coimbatore
tail -30 logs/client_del.log   # Delhi
tail -30 logs/frontend.log     # Vite frontend
```

## If a port is already in use

```bash
fuser -k 8001/tcp   # Kill whatever is on port 8001
fuser -k 8002/tcp
fuser -k 8003/tcp
fuser -k 8000/tcp
```

## Common Errors

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: pypdf` | `venv/bin/pip install pypdf python-docx` |
| `No module named 'uvicorn'` | Use `venv/bin/python3` — NOT system `python3` |
| `Address already in use :800X` | `fuser -k 800X/tcp` then restart |
| Clients show `This site can't be reached` | Check server (8000) started OK first |
| Frontend blank page | Check `npm run dev` running, visit http://localhost:3000 |
