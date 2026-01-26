@echo off
echo Starting DiffRhythm API Server...
cd /d "%~dp0"

REM Check if WSL is available
wsl --version >nul 2>&1
if %errorlevel% neq 0 (
    echo WSL not found. Please install WSL first.
    pause
    exit /b 1
)

echo Activating environment and running API server...
echo Access the API at http://localhost:8000
echo Swagger UI documentation at http://localhost:8000/docs
echo.

REM Uses wslpath to dynamically get the path of the current folder in WSL format
wsl bash -c "cd \"$(wslpath -u '%~dp0')\" && source .venv/bin/activate && uvicorn api:app --host 0.0.0.0 --port 8000"

pause
