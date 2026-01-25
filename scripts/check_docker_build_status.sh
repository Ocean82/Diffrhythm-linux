#!/bin/bash
# Check Docker build status on remote server
# Usage: bash scripts/check_docker_build_status.sh

EC2_HOST="52.0.207.242"
EC2_USER="ubuntu"
SSH_KEY="$HOME/server_saver_key"

echo "Checking Docker build status..."
ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" << 'EOF'
echo "=== Docker Processes ==="
ps aux | grep -E "(docker|pip|python)" | grep -v grep | head -10

echo ""
echo "=== Disk Space ==="
df -h /

echo ""
echo "=== Docker Images ==="
sudo docker images

echo ""
echo "=== Docker Containers ==="
sudo docker ps -a

echo ""
echo "=== Recent Docker Build Log (last 50 lines) ==="
if [ -f /tmp/docker_build.log ]; then
    tail -50 /tmp/docker_build.log
else
    echo "No build log found"
fi
EOF
