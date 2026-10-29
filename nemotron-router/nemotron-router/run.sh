#!/usr/bin/env bash
set -e

echo "============================================================"
echo " Starting Nemotron Router for Nebius x NVIDIA AI Hackathon"
echo "============================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Check python3
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3 not found. Please install Python 3.10+."
    exit 1
fi

# 2. Create venv
if [ ! -d ".venv" ]; then
    echo "[*] Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

# 3. Activate venv
source .venv/bin/activate

# 4. Install requirements
echo "[*] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Copy .env if missing
if [ ! -f ".env" ]; then
    echo "[*] Creating .env from .env.example..."
    cp .env.example .env
    echo "[!] NOTE: .env created from .env.example. Update NEBIUS_API_KEY when ready."
fi

# 6. Run check.py
echo "[*] Running pre-flight checks (check.py)..."
python check.py || echo "[!] Pre-flight check noted warning; continuing to start server."

# 7. Start server and open browser
if command -v xdg-open &>/dev/null; then
    xdg-open "http://localhost:8000" &
elif command -v open &>/dev/null; then
    open "http://localhost:8000" &
fi

echo "[*] Starting FastAPI Server on http://0.0.0.0:8000..."
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
