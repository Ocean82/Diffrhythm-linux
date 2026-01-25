#!/bin/bash
# Comprehensive server and build status check

set -e

EC2_HOST="52.0.207.242"
EC2_USER="ubuntu"
SSH_KEY="$HOME/server_saver_key"

echo "=========================================="
echo "Server & Build Status Investigation"
echo "=========================================="
echo ""

ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" << 'EOF'
    echo "=== BUILD LOG (Last 15 lines) ==="
    tail -15 /tmp/docker_build.log 2>/dev/null || echo "No build log found"
    echo ""
    
    echo "=== BUILD PROCESS STATUS ==="
    BUILD_RUNNING=$(ps aux | grep "[d]ocker build" | wc -l)
    if [ "$BUILD_RUNNING" -gt 0 ]; then
        echo "Build is RUNNING"
        ps aux | grep "[d]ocker build" | head -2
    else
        echo "Build is NOT running"
    fi
    echo ""
    
    echo "=== BUILD SUCCESS CHECK ==="
    if grep -qi "Successfully built\|Successfully tagged" /tmp/docker_build.log 2>/dev/null; then
        echo "✓ BUILD COMPLETED SUCCESSFULLY"
        grep -i "Successfully" /tmp/docker_build.log | tail -2
    else
        echo "Build not yet complete or failed"
    fi
    echo ""
    
    echo "=== BUILD ERRORS ==="
    ERROR_COUNT=$(grep -i "ERROR\|error\|failed" /tmp/docker_build.log 2>/dev/null | wc -l || echo "0")
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo "Found $ERROR_COUNT potential errors:"
        grep -i "ERROR\|error\|failed" /tmp/docker_build.log | tail -5
    else
        echo "No errors found in build log"
    fi
    echo ""
    
    echo "=== DOCKER IMAGES ==="
    sudo docker images | grep diffrhythm || echo "No diffrhythm image found"
    echo ""
    
    echo "=== DOCKER CONTAINERS ==="
    sudo docker ps -a | head -6
    echo ""
    
    echo "=== DISK SPACE ==="
    df -h / | tail -1
    echo ""
    
    echo "=== MEMORY USAGE ==="
    free -h | head -2
    echo ""
    
    echo "=== DOCKER SYSTEM STATUS ==="
    sudo docker system df
    echo ""
    
    echo "=== CONTAINER STATUS (docker-compose) ==="
    cd /opt/diffrhythm 2>/dev/null && sudo docker-compose -f docker-compose.prod.yml ps 2>/dev/null || echo "No containers running or docker-compose not found"
    echo ""
    
    echo "=== API HEALTH CHECK ==="
    curl -s -f http://localhost:8000/api/v1/health 2>/dev/null | head -5 || echo "API not responding"
    echo ""
    
    echo "=== BUILD LOG SIZE ==="
    if [ -f /tmp/docker_build.log ]; then
        LINES=$(wc -l < /tmp/docker_build.log)
        SIZE=$(du -h /tmp/docker_build.log | cut -f1)
        echo "Lines: $LINES, Size: $SIZE"
    else
        echo "Build log not found"
    fi
EOF

echo ""
echo "=========================================="
echo "Status Check Complete"
echo "=========================================="
