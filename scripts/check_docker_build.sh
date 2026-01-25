#!/bin/bash
# Check Docker build status
# Usage: bash scripts/check_docker_build.sh

EC2_HOST="52.0.207.242"
EC2_USER="ubuntu"
SSH_KEY="$HOME/server_saver_key"

echo "=========================================="
echo "Docker Build Status Check"
echo "=========================================="
echo ""

ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" << 'EOF'
echo "1. Build Process Status:"
echo "------------------------"
BUILD_PROCESS=$(ps aux | grep "docker build" | grep -v grep)
if [ -n "$BUILD_PROCESS" ]; then
    echo "✓ Build is running"
    echo "$BUILD_PROCESS" | head -1
else
    echo "✗ No build process found"
fi
echo ""

echo "2. Docker Images:"
echo "------------------------"
sudo docker images | head -5
echo ""

echo "3. Build Log (last 20 lines):"
echo "------------------------"
if [ -f /tmp/docker_build.log ]; then
    tail -20 /tmp/docker_build.log
else
    echo "No build log found"
fi
echo ""

echo "4. Disk Space:"
echo "------------------------"
df -h / | tail -1
echo ""

echo "5. Docker System Usage:"
echo "------------------------"
sudo docker system df
echo ""

echo "6. Check for diffrhythm image:"
echo "------------------------"
if sudo docker images | grep -q diffrhythm; then
    echo "✓ diffrhythm image found:"
    sudo docker images | grep diffrhythm
else
    echo "✗ diffrhythm image not found yet"
fi
EOF

echo ""
echo "=========================================="
