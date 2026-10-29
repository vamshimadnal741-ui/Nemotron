@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  Starting Nemotron Router for Nebius x NVIDIA AI Hackathon
echo ============================================================

cd /d "%~dp0"

:: 1. Check for Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found on PATH. Please install Python 3.10+.
    pause
    exit /b 1
)

:: 2. Create virtual environment if missing
if not exist ".venv" (
    echo [*] Creating virtual environment (.venv)...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: 3. Activate virtual environment
call .venv\Scripts\activate.bat

:: 4. Install / verify requirements
echo [*] Installing requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

:: 5. Create .env from .env.example if missing
if not exist ".env" (
    echo [*] Creating .env from .env.example...
    copy .env.example .env
    echo [!] NOTE: .env has been initialized with placeholder values.
    echo     Please edit .env to insert your NEBIUS_API_KEY when ready.
)

:: 6. Run check.py
echo [*] Running pre-flight checks (check.py)...
python check.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo [!] Pre-flight check encountered an error or missing key.
    echo     Starting server anyway so you can view the dashboard and configure settings.
    echo.
)

:: 7. Launch browser and start uvicorn
echo [*] Opening http://localhost:8000 in your browser...
start "" "http://localhost:8000"

echo [*] Starting FastAPI Uvicorn Server on http://0.0.0.0:8000...
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

pause
