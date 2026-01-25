#!/bin/bash
# Monitor Docker build progress on remote server

set -e

EC2_HOST="52.0.207.242"
EC2_USER="ubuntu"
SSH_KEY="$HOME/server_saver_key"

echo "=========================================="
echo "Docker Build Progress Monitor"
echo "=========================================="
echo ""

# Check if build is running
BUILD_RUNNING=$(ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "ps aux | grep 'docker build' | grep -v grep | wc -l" 2>/dev/null || echo "0")

if [ "$BUILD_RUNNING" -eq 0 ]; then
    echo "Checking build status..."
    ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" << 'EOF'
        if [ -f /tmp/docker_build.log ]; then
            echo "Build log found. Last 30 lines:"
            tail -30 /tmp/docker_build.log
            echo ""
            echo "Checking for errors..."
            grep -i "error\|failed\|success" /tmp/docker_build.log | tail -10
        else
            echo "No build log found. Build may not have started."
        fi
EOF
else
    echo "Build is running. Showing progress..."
    ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" << 'EOF'
        echo "Current build step:"
        tail -20 /tmp/docker_build.log 2>/dev/null | grep -E "^#|Step|RUN|COPY" | tail -5
        echo ""
        echo "Disk usage:"
        df -h / | tail -1
        echo ""
        echo "Docker system usage:"
        sudo docker system df 2>/dev/null | head -5
EOF
fi

echo ""
echo "To view full log:"
echo "  ssh -i $SSH_KEY $EC2_USER@$EC2_HOST 'tail -f /tmp/docker_build.log'"
