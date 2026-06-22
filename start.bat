@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM FedXGNN Epidemic Platform — Windows Startup Script
REM Usage: Double-click start.bat OR run from cmd: start.bat
REM ─────────────────────────────────────────────────────────────────────────────

SET SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

IF NOT EXIST "logs" mkdir logs

REM 1. Create venv if missing
IF NOT EXIST "venv\Scripts\activate.bat" (
    echo [*] Creating virtual environment...
    python -m venv venv
)

REM 2. Activate
call venv\Scripts\activate.bat

REM 3. Install dependencies
echo [*] Installing requirements...
pip install --upgrade pip -q
pip install -r requirements.txt -q
python -c "import spacy; spacy.load('en_core_web_sm')" 2>NUL || python -m spacy download en_core_web_sm -q

echo.
echo ════════════════════════════════════════════════════════
echo   FedXGNN Platform  —  Starting Services
echo ════════════════════════════════════════════════════════
echo   Central Server   ^> http://localhost:8000
echo   Bangalore Client ^> http://localhost:8001
echo   Coimbatore Client^> http://localhost:8002
echo   New Delhi Client ^> http://localhost:8003
echo   Frontend (Vite)  ^> http://localhost:5173
echo ════════════════════════════════════════════════════════
echo.

REM 4. Launch in separate windows
start "Central Server" cmd /k "call venv\Scripts\activate.bat && python backend/server.py"
timeout /t 4 /nobreak >NUL

start "Bangalore Client" cmd /k "call venv\Scripts\activate.bat && python client/client_app.py --port 8001 --censuscode 572 --name Bangalore"
start "Coimbatore Client" cmd /k "call venv\Scripts\activate.bat && python client/client_app.py --port 8002 --censuscode 632 --name Coimbatore"
start "Delhi Client" cmd /k "call venv\Scripts\activate.bat && python client/client_app.py --port 8003 --censuscode 94 --name Delhi"

REM 5. Frontend
cd frontend
call npm install -q
start "Frontend" cmd /k "npm run dev"
cd ..

echo [OK] All services started in separate windows.
echo Open http://localhost:5173 in your browser.
pause
