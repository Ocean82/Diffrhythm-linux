#!/bin/bash
# Monitor Docker build progress
# Usage: bash scripts/monitor_docker_build.sh

EC2_HOST="52.0.207.242"
EC2_USER="ubuntu"
SSH_KEY="$HOME/server_saver_key"

echo "Monitoring Docker build..."
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    echo "=== $(date) ==="
    ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" << 'EOF'
echo "Disk space:"
df -h / | tail -1
echo ""
echo "Docker processes:"
ps aux | grep -E "(docker|pip)" | grep -v grep | head -5
echo ""
echo "Build log (last 10 lines):"
tail -10 /tmp/docker_build.log 2>/dev/null || echo "No log file yet"
echo ""
EOF
    sleep 30
done
