@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM FedXGNN Platform — Laptop A (Rohith_lappy) Orchestration Script
REM ─────────────────────────────────────────────────────────────────────────────

SET SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo.
echo ════════════════════════════════════════════════════════
echo   FedXGNN Platform  —  Starting Laptop A (Host)
echo ════════════════════════════════════════════════════════
echo   1. GNN Central Server    ^> http://localhost:8000
echo   2. Bangalore Client Node  ^> http://localhost:8001
echo   3. React Frontend UI      ^> http://localhost:3000
echo ════════════════════════════════════════════════════════
echo.

REM 1. Activate venv / check environment
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Python virtual environment (venv) not found.
    echo Please make sure you have set up the virtual environment first.
    pause
    exit /b 1
)

REM 2. Start Central Server
echo [*] Launching Central GNN Server...
start "FedXGNN Central Server" cmd /k "call venv\Scripts\activate.bat && python backend/server.py"

REM Wait 4 seconds for backend to bind and load dataset
timeout /t 4 /nobreak >NUL

REM 3. Start Bangalore Client
echo [*] Launching Bangalore Hospital Client...
start "Bangalore Edge Client" cmd /k "call venv\Scripts\activate.bat && python client/client_app.py --port 8001 --censuscode 572 --name \"Bangalore Hospital\" --server http://localhost:8000"

REM 4. Start React Frontend
echo [*] Launching React Dashboard...
cd frontend
start "FedXGNN Frontend UI" cmd /k "npm run dev"
cd ..

echo.
echo [SUCCESS] Laptop A components have been launched in separate windows:
echo - Central Server (Port 8000)
echo - Bangalore Hospital Client (Port 8001)
echo - Frontend Dashboard (Port 3000/3001)
echo.
pause
