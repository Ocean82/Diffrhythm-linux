#!/bin/bash
# Deploy Implementation to Server
# This script deploys the updated backend code to the server

set -e

# Configuration - UPDATE THESE VALUES
SERVER_USER="ubuntu"
SERVER_HOST="burntbeats.com"  # Update with your server IP or hostname
SERVER_PATH="/home/ubuntu/app"
SSH_KEY=""  # Optional: path to SSH key if needed

echo "=========================================="
echo "Deploying Implementation to Server"
echo "=========================================="
echo ""

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "❌ Error: backend/ directory not found"
    exit 1
fi

# Build SSH command
SSH_CMD="ssh"
if [ -n "$SSH_KEY" ]; then
    SSH_CMD="ssh -i $SSH_KEY"
fi

SSH_TARGET="${SERVER_USER}@${SERVER_HOST}"

echo "[1] Testing SSH connection..."
if $SSH_CMD -o ConnectTimeout=5 $SSH_TARGET "echo 'SSH connection successful'" > /dev/null 2>&1; then
    echo "  ✅ SSH connection successful"
else
    echo "  ❌ SSH connection failed"
    echo "  Please verify:"
    echo "    - Server hostname/IP: $SERVER_HOST"
    echo "    - Username: $SERVER_USER"
    echo "    - SSH keys are configured"
    exit 1
fi

echo ""
echo "[2] Creating backup of existing backend..."
$SSH_CMD $SSH_TARGET "cd $SERVER_PATH && if [ -d backend ]; then cp -r backend backend.backup.\$(date +%Y%m%d_%H%M%S); echo '  ✅ Backup created'; else echo '  ⚠️  No existing backend to backup'; fi"

echo ""
echo "[3] Deploying backend code..."
scp -r backend/ ${SSH_TARGET}:${SERVER_PATH}/
echo "  ✅ Backend code deployed"

echo ""
echo "[4] Deploying test scripts..."
scp test_server_implementation.py ${SSH_TARGET}:${SERVER_PATH}/
scp test_payment_flow.py ${SSH_TARGET}:${SERVER_PATH}/
echo "  ✅ Test scripts deployed"

echo ""
echo "[5] Verifying deployment..."
$SSH_CMD $SSH_TARGET "cd $SERVER_PATH && ls -la backend/api.py backend/payment_verification.py backend/config.py" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "  ✅ All files verified"
else
    echo "  ⚠️  Some files may be missing"
fi

echo ""
echo "[6] Restarting service..."
$SSH_CMD $SSH_TARGET "sudo systemctl restart burntbeats-api"
echo "  ✅ Service restart command sent"

echo ""
echo "[7] Checking service status..."
sleep 3
$SSH_CMD $SSH_TARGET "sudo systemctl status burntbeats-api --no-pager | head -n 10"

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. SSH to server: ssh $SSH_TARGET"
echo "2. Run tests: cd $SERVER_PATH && python3 test_server_implementation.py"
echo "3. Check logs: sudo journalctl -u burntbeats-api -n 50"
echo "4. Configure Stripe webhook in Dashboard"
echo ""
echo "See SERVER_TESTING_GUIDE.md for detailed testing instructions"
