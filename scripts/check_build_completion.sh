#!/bin/bash
# Check if Docker build completed successfully

EC2_HOST="52.0.207.242"
EC2_USER="ubuntu"
SSH_KEY="$HOME/server_saver_key"

ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" << 'EOF'
    echo "=== Checking Build Completion ==="
    
    # Check if build process is still running
    BUILD_PROCESS=$(ps aux | grep "[d]ocker build" | wc -l)
    if [ "$BUILD_PROCESS" -gt 0 ]; then
        echo "Build is still RUNNING"
    else
        echo "Build process is NOT running"
    fi
    echo ""
    
    # Check for success message
    echo "=== Success Messages ==="
    if grep -q "Successfully built\|Successfully tagged" /tmp/docker_build.log 2>/dev/null; then
        echo "✓ BUILD SUCCEEDED"
        grep "Successfully" /tmp/docker_build.log | tail -3
    else
        echo "✗ No success message found"
    fi
    echo ""
    
    # Check for jieba error specifically
    echo "=== Jieba Installation Status ==="
    if grep -q "ERROR.*jieba" /tmp/docker_build.log 2>/dev/null; then
        echo "⚠ Jieba error found:"
        grep "jieba" /tmp/docker_build.log | grep -i error | tail -3
        echo ""
        echo "Checking if fallback worked:"
        if grep -q "Installing.*jieba\|Successfully installed.*jieba" /tmp/docker_build.log 2>/dev/null; then
            echo "✓ Jieba was installed (fallback worked)"
        else
            echo "✗ Jieba installation failed"
        fi
    else
        echo "✓ No jieba errors found"
    fi
    echo ""
    
    # Check if image exists
    echo "=== Docker Image Check ==="
    IMAGE_EXISTS=$(sudo docker images | grep "diffrhythm.*prod" | wc -l)
    if [ "$IMAGE_EXISTS" -gt 0 ]; then
        echo "✓ Image EXISTS:"
        sudo docker images | grep diffrhythm
    else
        echo "✗ Image NOT found"
    fi
    echo ""
    
    # Show last 10 lines of build log
    echo "=== Last 10 Lines of Build Log ==="
    tail -10 /tmp/docker_build.log
EOF
