@echo off
setlocal


REM ----------------------------
REM Find Python
REM ----------------------------
where python >nul 2>nul
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3 and try again.
    pause
    exit /b 1
)

REM ----------------------------
REM Create venv if missing
REM ----------------------------
if not exist venv (
    echo 🔧 Creating virtual environment...
    python -m venv venv
)

REM ----------------------------
REM Upgrade pip & install deps
REM ----------------------------
echo 📦 Installing dependencies...
call venv\Scripts\python -m pip install --upgrade pip
call venv\Scripts\python -m pip install -r requirements.txt

REM ----------------------------
REM Start app
REM ----------------------------
echo 🚀 Starting application...
start cmd /k venv\Scripts\python scripts\run_dev.py

REM ----------------------------
REM Open browser
REM ----------------------------
timeout /t 3 >nul
start http://127.0.0.1:8050/

endlocal