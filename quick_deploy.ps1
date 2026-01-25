# Quick Deploy Script for Windows PowerShell
# Deploys backend code to server

$SERVER_USER = "ubuntu"
$SERVER_HOST = "burntbeats.com"  # Update if different
$SERVER_PATH = "/home/ubuntu/app"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Deploying Implementation to Server" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if backend directory exists
if (-not (Test-Path "backend")) {
    Write-Host "Error: backend/ directory not found" -ForegroundColor Red
    exit 1
}

Write-Host "[1] Deploying backend code..." -ForegroundColor Yellow
scp -r backend/ ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/
Write-Host "  Backend code deployed" -ForegroundColor Green

Write-Host ""
Write-Host "[2] Deploying test scripts..." -ForegroundColor Yellow
scp test_server_implementation.py ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/
scp test_payment_flow.py ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/
Write-Host "  Test scripts deployed" -ForegroundColor Green

Write-Host ""
Write-Host "[3] Restarting service..." -ForegroundColor Yellow
ssh ${SERVER_USER}@${SERVER_HOST} "sudo systemctl restart burntbeats-api"
Write-Host "  Service restart command sent" -ForegroundColor Green

Write-Host ""
Write-Host "[4] Checking service status..." -ForegroundColor Yellow
Start-Sleep -Seconds 3
ssh ${SERVER_USER}@${SERVER_HOST} "sudo systemctl status burntbeats-api --no-pager | head -n 10"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. SSH to server: ssh ${SERVER_USER}@${SERVER_HOST}"
Write-Host "2. Run tests: cd ${SERVER_PATH} && python3 test_server_implementation.py"
Write-Host "3. Check logs: sudo journalctl -u burntbeats-api -n 50"
Write-Host ""
Write-Host "See DEPLOYMENT_COMMANDS.md for detailed instructions" -ForegroundColor Cyan
