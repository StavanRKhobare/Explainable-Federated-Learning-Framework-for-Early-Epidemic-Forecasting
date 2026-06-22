# FedXGNN — Multi-Laptop LAN Presentation Setup Guide

This guide describes the exact steps to launch a collaborative, live two-laptop demonstration on your local network (LAN) with the central database server, frontend dashboard, and Bangalore client running on **Laptop A (`Rohith_lappy`)**, and Coimbatore & Mysore clients running on **Laptop B**.

---

## 📡 Prerequisites
1. **Connect both laptops to the same Wi-Fi/LAN network**.
2. **Set your network profile to Private** on both systems (Windows Defender Firewall blocks local ports `3000`/`8000` on Public Wi-Fi profiles).

---

## 💻 LAPTOP A — `Rohith_lappy` (Main Server & UI)
Open separate terminal/command prompt sessions on your Windows system (`Rohith_lappy`) and execute:

### Step 1: Start Central GNN Server (Terminal 1)
```cmd
venv\Scripts\python backend/server.py
```
*Note: This binds to `0.0.0.0:8000` to automatically accept connections from Laptop B.*

### Step 2: Start the React Frontend (Terminal 2)
```cmd
cd frontend
npm run dev
```
*Note: This will run on `0.0.0.0:3000` (or `3001` if port 3000 is occupied).*

### Step 3: Start Bangalore Edge Client Node (Terminal 3)
```cmd
venv\Scripts\python client/client_app.py --port 8001 --censuscode 572 --name "Bangalore Hospital" --server http://localhost:8000
```

---

## 💻 LAPTOP B — (Coimbatore & Mysore Edge Nodes)
Open separate terminals on Laptop B and run:

### Step 1: Start Coimbatore Edge Client (Terminal 1)
*   **Windows**:
    ```cmd
    venv\Scripts\python client/client_app.py --port 8002 --censuscode 632 --name "Coimbatore Hospital" --server http://Rohith_lappy.local:8000
    ```
*   **Linux/Mac**:
    ```bash
    python3 client/client_app.py --port 8002 --censuscode 632 --name "Coimbatore Hospital" --server http://Rohith_lappy.local:8000
    ```

### Step 2: Start Mysore Edge Client (Terminal 2)
*   **Windows**:
    ```cmd
    venv\Scripts\python client/client_app.py --port 8004 --censuscode 577 --name "Mysore Hospital" --server http://Rohith_lappy.local:8000
    ```
*   **Linux/Mac**:
    ```bash
    python3 client/client_app.py --port 8004 --censuscode 577 --name "Mysore Hospital" --server http://Rohith_lappy.local:8000
    ```

### Step 3: Access the Visual Dashboard
Open a browser on Laptop B and navigate to:
```
http://Rohith_lappy.local:3000
```
*(or port `3001` depending on Vite's console output).*

---

## ⚠️ Troubleshooting & Hostname Fallbacks

Because the hostname `Rohith_lappy` contains a standard-violating **underscore (`_`)**, standard DNS lookups on some networks or routers might fail.

If Laptop B displays a "Connection Refused", "Server not found", or "DNS Error" when using `Rohith_lappy.local`:

1.  **Find your LAN IP address on Laptop A**:
    - Open command prompt on `Rohith_lappy` and run:
      ```cmd
      ipconfig
      ```
    - Locate your IPv4 Address (e.g. `192.168.1.15`). Let's call this `<LaptopA_IP>`.
2.  **Run Client Nodes on Laptop B using the IP**:
    ```bash
    python3 client/client_app.py --port 8002 --censuscode 632 --name "Coimbatore Hospital" --server http://<LaptopA_IP>:8000
    ```
3.  **Open browser on Laptop B using the IP**:
    Open a web browser and load:
    ```
    http://<LaptopA_IP>:3000
    ```
