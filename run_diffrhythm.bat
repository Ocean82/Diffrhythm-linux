@echo off
echo Starting DiffRhythm...
cd /d "%~dp0"

REM Check if WSL is available
wsl --version >nul 2>&1
if %errorlevel% neq 0 (
    echo WSL not found. Please install WSL first.
    pause
    exit /b 1
)

echo Activating environment and running inference...
wsl bash -c "cd /mnt/d/EMBERS-BANK/DiffRhythm-Linux && source .venv/bin/activate && bash scripts/infer_prompt_ref.sh"

echo.
echo Check the output folder: infer/example/output/
pause