@echo off
echo Checking DiffRhythm Setup...
cd /d "%~dp0"

wsl bash -c "cd /mnt/d/EMBERS-BANK/DiffRhythm-Linux && python3 check_aws_readiness.py"

pause