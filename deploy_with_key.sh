#!/bin/bash
# Deploy Implementation to Server
# Uses SSH key: ~/.ssh/server_saver_key

set -e

SSH_KEY="$HOME/.ssh/server_saver_key"
SERVER="ubuntu@52.0.207.242"
SERVER_PATH="/home/ubuntu/app"

echo "=========================================="
echo "Deploying Implementation to Server"
echo "=========================================="
echo ""

# Check SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo "❌ SSH key not found: $SSH_KEY"
    exit 1
fi

echo "[1] Testing SSH connection..."
if ssh -i "$SSH_KEY" -o ConnectTimeout=10 "$SERVER" "echo 'SSH connection successful'" > /dev/null 2>&1; then
    echo "  ✅ SSH connection successful"
else
    echo "  ❌ SSH connection failed"
    echo "  Please verify:"
    echo "    - SSH key: $SSH_KEY"
    echo "    - Server: $SERVER"
    exit 1
fi

echo ""
echo "[2] Creating backup of existing backend..."
ssh -i "$SSH_KEY" "$SERVER" "cd $SERVER_PATH && if [ -d backend ]; then cp -r backend backend.backup.\$(date +%Y%m%d_%H%M%S); echo '  ✅ Backup created'; else echo '  ⚠️  No existing backend to backup'; fi"

echo ""
echo "[3] Deploying backend code..."
scp -i "$SSH_KEY" -r backend/ ${SERVER}:${SERVER_PATH}/
echo "  ✅ Backend code deployed"

echo ""
echo "[4] Deploying test scripts..."
scp -i "$SSH_KEY" test_server_implementation.py ${SERVER}:${SERVER_PATH}/
scp -i "$SSH_KEY" test_payment_flow.py ${SERVER}:${SERVER_PATH}/
echo "  ✅ Test scripts deployed"

echo ""
echo "[5] Verifying deployment..."
ssh -i "$SSH_KEY" "$SERVER" "cd $SERVER_PATH && ls -la backend/api.py backend/payment_verification.py backend/config.py" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "  ✅ All files verified"
else
    echo "  ⚠️  Some files may be missing"
fi

echo ""
echo "[6] Restarting service..."
ssh -i "$SSH_KEY" "$SERVER" "sudo systemctl restart burntbeats-api"
echo "  ✅ Service restart command sent"

echo ""
echo "[7] Checking service status..."
sleep 3
ssh -i "$SSH_KEY" "$SERVER" "sudo systemctl status burntbeats-api --no-pager | head -n 15"

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. SSH to server: ssh -i $SSH_KEY $SERVER"
echo "2. Run tests: cd $SERVER_PATH && python3 test_server_implementation.py"
echo "3. Check logs: sudo journalctl -u burntbeats-api -n 50"
echo "4. Test health: curl http://127.0.0.1:8001/api/v1/health"
