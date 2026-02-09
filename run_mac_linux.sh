#!/bin/bash
set -e

# ----------------------------
# Find Python
# ----------------------------
if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "❌ Python not found. Please install Python 3."
  exit 1
fi

# ----------------------------
# Create venv if missing
# ----------------------------
if [ ! -d "venv" ]; then
  echo "🔧 Creating virtual environment..."
  $PYTHON -m venv venv
fi

# ----------------------------
# Install deps
# ----------------------------
echo "📦 Installing dependencies..."
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.txt

# ----------------------------
# Start app
# ----------------------------
echo "🚀 Starting application..."
venv/bin/python scripts/run_dev.py &

# ----------------------------
# Open browser
# ----------------------------
sleep 3
if command -v xdg-open >/dev/null; then
  xdg-open http://127.0.0.1:8050/
elif command -v open >/dev/null; then
  open http://127.0.0.1:8050/
fi