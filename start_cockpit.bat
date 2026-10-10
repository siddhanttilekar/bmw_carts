@echo off
title BMW PM Intelligence Co-Pilot Launcher
color 0B

echo ===============================================================================
echo        BMW PRODUCT MANAGEMENT CO-PILOT -- NEUE KLASSE G0/G1 COCKPIT
echo ===============================================================================
echo.
echo [1/3] Checking environment & dependencies...
python -c "import fastapi, uvicorn, openpyxl, pandas, pydantic; print('All core libraries verified.')" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing required packages from requirements.txt...
    pip install -r requirements.txt
) else (
    echo Dependencies verified.
)

echo.
echo [2/3] Launching FastAPI Backend & Cockpit Server on http://127.0.0.1:8000...
start "" http://127.0.0.1:8000/

echo.
echo [3/3] Server running. Press Ctrl+C in this terminal to stop the application.
echo ===============================================================================
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
pause
