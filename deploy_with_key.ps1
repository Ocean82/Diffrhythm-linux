# Deploy Implementation to Server (PowerShell)
# Uses SSH key: ~/.ssh/server_saver_key

$SSH_KEY = "$env:USERPROFILE\.ssh\server_saver_key"
$SERVER = "ubuntu@52.0.207.242"
$SERVER_PATH = "/home/ubuntu/app"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Deploying Implementation to Server" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check SSH key exists
if (-not (Test-Path $SSH_KEY)) {
    Write-Host "Error: SSH key not found: $SSH_KEY" -ForegroundColor Red
    Write-Host "Trying alternative path..." -ForegroundColor Yellow
    $SSH_KEY = "~/.ssh/server_saver_key"
}

Write-Host "[1] Testing SSH connection..." -ForegroundColor Yellow
$testResult = ssh -i $SSH_KEY -o ConnectTimeout=10 $SERVER "echo 'SSH connection successful'" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  SSH connection successful" -ForegroundColor Green
} else {
    Write-Host "  SSH connection failed" -ForegroundColor Red
    Write-Host "  Please verify SSH key and server access" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "[2] Creating backup..." -ForegroundColor Yellow
ssh -i $SSH_KEY $SERVER "cd $SERVER_PATH && if [ -d backend ]; then cp -r backend backend.backup.`$(date +%Y%m%d_%H%M%S); echo 'Backup created'; fi"

Write-Host ""
Write-Host "[3] Deploying backend code..." -ForegroundColor Yellow
scp -i $SSH_KEY -r backend/ ${SERVER}:${SERVER_PATH}/
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Backend code deployed" -ForegroundColor Green
} else {
    Write-Host "  Deployment failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[4] Deploying test scripts..." -ForegroundColor Yellow
scp -i $SSH_KEY test_server_implementation.py ${SERVER}:${SERVER_PATH}/
scp -i $SSH_KEY test_payment_flow.py ${SERVER}:${SERVER_PATH}/
Write-Host "  Test scripts deployed" -ForegroundColor Green

Write-Host ""
Write-Host "[5] Verifying deployment..." -ForegroundColor Yellow
ssh -i $SSH_KEY $SERVER "cd $SERVER_PATH && ls -la backend/api.py backend/payment_verification.py backend/config.py"
if ($LASTEXITCODE -eq 0) {
    Write-Host "  All files verified" -ForegroundColor Green
}

Write-Host ""
Write-Host "[6] Restarting service..." -ForegroundColor Yellow
ssh -i $SSH_KEY $SERVER "sudo systemctl restart burntbeats-api"
Write-Host "  Service restart command sent" -ForegroundColor Green

Write-Host ""
Write-Host "[7] Checking service status..." -ForegroundColor Yellow
Start-Sleep -Seconds 3
ssh -i $SSH_KEY $SERVER "sudo systemctl status burntbeats-api --no-pager | head -n 15"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. SSH to server: ssh -i $SSH_KEY $SERVER"
Write-Host "2. Run tests: cd $SERVER_PATH && python3 test_server_implementation.py"
Write-Host "3. Check logs: sudo journalctl -u burntbeats-api -n 50"
