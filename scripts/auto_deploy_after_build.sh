#!/bin/bash
# Automatically deploy after Docker build completes
# Monitors build progress and proceeds with deployment

set -e

EC2_HOST="52.0.207.242"
EC2_USER="ubuntu"
SSH_KEY="$HOME/server_saver_key"
PROJECT_DIR="/opt/diffrhythm"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=========================================="
echo "Auto-Deploy After Build Complete"
echo "=========================================="
echo ""

# Function to check build status
check_build_status() {
    ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" << 'EOF'
        # Check if build process is running
        BUILD_RUNNING=$(ps aux | grep "docker build" | grep -v grep | wc -l)
        
        if [ "$BUILD_RUNNING" -eq 0 ]; then
            # Build finished, check for success
            if [ -f /tmp/docker_build.log ]; then
                if grep -qi "Successfully built\|Successfully tagged" /tmp/docker_build.log; then
                    echo "SUCCESS"
                elif grep -qi "ERROR\|error\|failed" /tmp/docker_build.log | tail -1 | grep -v "grep"; then
                    echo "FAILED"
                else
                    echo "UNKNOWN"
                fi
            else
                echo "NO_LOG"
            fi
        else
            echo "RUNNING"
        fi
EOF
}

# Function to wait for build
wait_for_build() {
    echo -e "${YELLOW}Waiting for Docker build to complete...${NC}"
    echo ""
    
    MAX_WAIT=3600  # 1 hour max
    ELAPSED=0
    CHECK_INTERVAL=30  # Check every 30 seconds
    
    while [ $ELAPSED -lt $MAX_WAIT ]; do
        STATUS=$(check_build_status)
        
        case "$STATUS" in
            "SUCCESS")
                echo -e "${GREEN}✓ Docker build completed successfully!${NC}"
                return 0
                ;;
            "FAILED")
                echo -e "${RED}✗ Docker build failed!${NC}"
                echo "Showing last 50 lines of build log:"
                ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "tail -50 /tmp/docker_build.log"
                return 1
                ;;
            "RUNNING")
                # Show progress
                PROGRESS=$(ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "tail -3 /tmp/docker_build.log 2>/dev/null | grep -E 'Step|RUN|COPY' | tail -1" || echo "Building...")
                echo -e "${YELLOW}[$ELAPSED s] Build in progress... $PROGRESS${NC}"
                ;;
            *)
                echo -e "${YELLOW}[$ELAPSED s] Checking build status...${NC}"
                ;;
        esac
        
        sleep $CHECK_INTERVAL
        ELAPSED=$((ELAPSED + CHECK_INTERVAL))
    done
    
    echo -e "${RED}Build timeout after $MAX_WAIT seconds${NC}"
    return 1
}

# Step 1: Wait for build to complete
if ! wait_for_build; then
    echo -e "${RED}Build did not complete successfully. Exiting.${NC}"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 1: Verifying Docker Image"
echo "=========================================="

# Verify image exists
IMAGE_EXISTS=$(ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "sudo docker images | grep diffrhythm:prod | wc -l")
if [ "$IMAGE_EXISTS" -gt 0 ]; then
    echo -e "${GREEN}✓ Docker image 'diffrhythm:prod' found${NC}"
    ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "sudo docker images | grep diffrhythm"
else
    echo -e "${RED}✗ Docker image 'diffrhythm:prod' not found${NC}"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Stopping Existing Containers"
echo "=========================================="

# Stop any existing containers
ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" << 'EOF'
    cd /opt/diffrhythm
    if [ -f docker-compose.prod.yml ]; then
        sudo docker-compose -f docker-compose.prod.yml down 2>&1 || true
        echo "Existing containers stopped"
    else
        echo "No docker-compose file found, skipping"
    fi
EOF

echo ""
echo "=========================================="
echo "Step 3: Starting Docker Containers"
echo "=========================================="

# Start containers
ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" << 'EOF'
    cd /opt/diffrhythm
    echo "Starting services with docker-compose..."
    sudo docker-compose -f docker-compose.prod.yml up -d
    echo "Waiting for services to initialize..."
    sleep 20
EOF

echo -e "${GREEN}✓ Docker containers started${NC}"

echo ""
echo "=========================================="
echo "Step 4: Checking Container Status"
echo "=========================================="

# Check container status
ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "cd $PROJECT_DIR && sudo docker-compose -f docker-compose.prod.yml ps"

echo ""
echo "=========================================="
echo "Step 5: Checking Container Logs"
echo "=========================================="

# Check logs
ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" << 'EOF'
    cd /opt/diffrhythm
    echo "Last 40 lines of diffrhythm-api logs:"
    sudo docker-compose -f docker-compose.prod.yml logs --tail=40 diffrhythm-api
EOF

echo ""
echo "=========================================="
echo "Step 6: Health Check"
echo "=========================================="

# Health check with retries
HEALTH_OK=false
for i in {1..6}; do
    echo "Attempt $i/6: Checking health endpoint..."
    HEALTH_RESPONSE=$(ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "curl -s -f http://localhost:8000/api/v1/health 2>&1" || echo "FAILED")
    
    if [[ "$HEALTH_RESPONSE" == *"status"* ]] || [[ "$HEALTH_RESPONSE" == *"healthy"* ]]; then
        echo -e "${GREEN}✓ Health check passed!${NC}"
        echo "$HEALTH_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$HEALTH_RESPONSE"
        HEALTH_OK=true
        break
    else
        echo -e "${YELLOW}⚠ Health check returned: ${HEALTH_RESPONSE:0:100}${NC}"
        if [ $i -lt 6 ]; then
            echo "Waiting 10 seconds before retry..."
            sleep 10
        fi
    fi
done

if [ "$HEALTH_OK" = false ]; then
    echo -e "${YELLOW}⚠ Health check did not pass, but continuing...${NC}"
    echo "Service may still be starting. Check logs manually."
fi

echo ""
echo "=========================================="
echo "Step 7: Testing API Endpoints"
echo "=========================================="

# Test root endpoint
ROOT_RESPONSE=$(ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "curl -s http://localhost:8000/ 2>&1" || echo "FAILED")
if [[ "$ROOT_RESPONSE" == *"DiffRhythm"* ]] || [[ "$ROOT_RESPONSE" == *"API"* ]]; then
    echo -e "${GREEN}✓ Root endpoint responding${NC}"
    echo "$ROOT_RESPONSE" | head -3
else
    echo -e "${YELLOW}⚠ Root endpoint: ${ROOT_RESPONSE:0:100}${NC}"
fi

# Test metrics endpoint
METRICS_RESPONSE=$(ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "curl -s http://localhost:8000/api/v1/metrics 2>&1" || echo "FAILED")
if [[ "$METRICS_RESPONSE" == *"diffrhythm"* ]] || [[ "$METRICS_RESPONSE" == *"# TYPE"* ]]; then
    echo -e "${GREEN}✓ Metrics endpoint responding${NC}"
else
    echo -e "${YELLOW}⚠ Metrics endpoint: ${METRICS_RESPONSE:0:100}${NC}"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}Auto-Deployment Complete!${NC}"
echo "=========================================="
echo ""
echo "API Endpoint: http://$EC2_HOST:8000"
echo "Health Check: http://$EC2_HOST:8000/api/v1/health"
echo "API Docs: http://$EC2_HOST:8000/docs"
echo ""
echo "Useful commands:"
echo "  View logs: ssh -i $SSH_KEY $EC2_USER@$EC2_HOST 'cd $PROJECT_DIR && sudo docker-compose -f docker-compose.prod.yml logs -f diffrhythm-api'"
echo "  Restart: ssh -i $SSH_KEY $EC2_USER@$EC2_HOST 'cd $PROJECT_DIR && sudo docker-compose -f docker-compose.prod.yml restart diffrhythm-api'"
echo "  Stop: ssh -i $SSH_KEY $EC2_USER@$EC2_HOST 'cd $PROJECT_DIR && sudo docker-compose -f docker-compose.prod.yml down'"
echo ""
