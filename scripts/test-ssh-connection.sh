#!/bin/bash
# Test SSH connection to diagnose server connectivity issues

EC2_HOST="52.0.207.242"
EC2_USER="ubuntu"
SSH_KEY_ORIG="C:/Users/sammy/.ssh/server_saver_key"

echo "=========================================="
echo "SSH Connection Diagnostic"
echo "=========================================="
echo "Server: $EC2_USER@$EC2_HOST"
echo "Original key path: $SSH_KEY_ORIG"
echo ""

# Test 1: Check if WSL path exists
echo "[Test 1] Checking WSL path..."
SSH_KEY_WSL="/mnt/c/Users/sammy/.ssh/server_saver_key"
if [ -f "$SSH_KEY_WSL" ]; then
    echo "✓ Key found at: $SSH_KEY_WSL"
    ls -lh "$SSH_KEY_WSL"
else
    echo "✗ Key NOT found at: $SSH_KEY_WSL"
    echo "  Checking directory..."
    ls -la /mnt/c/Users/sammy/.ssh/ 2>&1 | head -10
fi
echo ""

# Test 2: Try path conversion from deploy script
echo "[Test 2] Testing deploy script path conversion..."
CONVERTED=$(echo "$SSH_KEY_ORIG" | sed 's|C:|/mnt/c|' | sed 's|\\|/|g' | tr '[:upper:]' '[:lower:]')
echo "  Original: $SSH_KEY_ORIG"
echo "  Converted (with lowercase): $CONVERTED"
if [ -f "$CONVERTED" ]; then
    echo "  ✓ Converted path works"
else
    echo "  ✗ Converted path does NOT work"
    # Try without lowercase conversion
    CONVERTED_NO_LOWER=$(echo "$SSH_KEY_ORIG" | sed 's|C:|/mnt/c|' | sed 's|\\|/|g')
    echo "  Trying without lowercase: $CONVERTED_NO_LOWER"
    if [ -f "$CONVERTED_NO_LOWER" ]; then
        echo "  ✓ Path without lowercase conversion works!"
    else
        echo "  ✗ Path without lowercase also fails"
    fi
fi
echo ""

# Test 3: Copy key and test SSH
echo "[Test 3] Testing SSH connection..."
if [ -f "$SSH_KEY_WSL" ]; then
    TEST_KEY="$HOME/server_saver_key_test_$$"
    cp "$SSH_KEY_WSL" "$TEST_KEY"
    chmod 600 "$TEST_KEY"
    echo "  Copied key to: $TEST_KEY"
    
    echo "  Attempting SSH connection..."
    if ssh -i "$TEST_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$EC2_USER@$EC2_HOST" "echo 'Connection successful'" 2>&1; then
        echo "  ✓ SSH connection successful!"
    else
        SSH_EXIT=$?
        echo "  ✗ SSH connection failed (exit code: $SSH_EXIT)"
        echo "  Trying with verbose output..."
        ssh -v -i "$TEST_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$EC2_USER@$EC2_HOST" "echo 'test'" 2>&1 | tail -20
    fi
    
    rm -f "$TEST_KEY"
else
    echo "  ✗ Cannot test SSH - key file not found"
fi
echo ""

# Test 4: Network connectivity
echo "[Test 4] Testing network connectivity..."
if ping -c 1 -W 2 "$EC2_HOST" > /dev/null 2>&1; then
    echo "  ✓ Server is reachable (ping)"
else
    echo "  ✗ Server is NOT reachable (ping failed)"
fi

if nc -z -w 2 "$EC2_HOST" 22 > /dev/null 2>&1; then
    echo "  ✓ Port 22 is open"
else
    echo "  ✗ Port 22 is NOT accessible"
fi
echo ""

echo "=========================================="
echo "Diagnostic Complete"
echo "=========================================="
