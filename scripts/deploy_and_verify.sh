#!/bin/bash
# Complete Deployment and Verification Script
# Builds, deploys, and verifies the DiffRhythm backend

set -e

# Configuration
SERVER_IP="${SERVER_IP:-52.0.207.242}"
SERVER_USER="${SERVER_USER:-ubuntu}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/server_saver_key}"
PROJECT_DIR="/opt/diffrhythm"
IMAGE_NAME="diffrhythm:prod"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

SSH_CMD="ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no ${SERVER_USER}@${SERVER_IP}"

echo "=========================================="
echo "DiffRhythm Complete Deployment Script"
echo "=========================================="
echo ""

# Step 1: Check prerequisites
echo -e "${BLUE}Step 1: Checking prerequisites...${NC}"
$SSH_CMD "cd ${PROJECT_DIR} && test -f Dockerfile.prod" || {
    echo -e "${RED}Error: Dockerfile.prod not found on server${NC}"
    exit 1
}
echo -e "${GREEN}✓ Prerequisites OK${NC}"

# Step 2: Check disk space
echo ""
echo -e "${BLUE}Step 2: Checking disk space...${NC}"
DISK_FREE=$($SSH_CMD "df -h / | tail -1 | awk '{print \$4}' | sed 's/G//'")
DISK_PERCENT=$($SSH_CMD "df -h / | tail -1 | awk '{print \$5}' | sed 's/%//'")
echo "  Free space: ${DISK_FREE}GB"
echo "  Usage: ${DISK_PERCENT}%"

if [ "$DISK_PERCENT" -gt 90 ]; then
    echo -e "${RED}Warning: Disk usage is ${DISK_PERCENT}% - build may fail${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 3: Build Docker image
echo ""
echo -e "${BLUE}Step 3: Building Docker image...${NC}"
echo "  This may take 25-45 minutes..."
echo "  Monitoring build progress..."

$SSH_CMD "cd ${PROJECT_DIR} && \
    docker build -f Dockerfile.prod -t ${IMAGE_NAME} . \
    > /tmp/docker_build.log 2>&1 & \
    echo \$!" > /tmp/build_pid.txt

BUILD_PID=$(cat /tmp/build_pid.txt)
echo "  Build process started (PID: $BUILD_PID)"

# Monitor build
echo "  Monitoring build (press Ctrl+C to stop monitoring, build will continue)..."
while $SSH_CMD "ps -p $BUILD_PID > /dev/null 2>&1"; do
    sleep 10
    LAST_LINE=$($SSH_CMD "tail -1 /tmp/docker_build.log 2>/dev/null" || echo "")
    if [ -n "$LAST_LINE" ]; then
        echo "  ... $LAST_LINE"
    fi
done

# Check build result
BUILD_SUCCESS=$($SSH_CMD "docker images ${IMAGE_NAME} --format '{{.Repository}}:{{.Tag}}' 2>/dev/null" || echo "")
if [ -n "$BUILD_SUCCESS" ]; then
    echo -e "${GREEN}✓ Docker image built successfully${NC}"
else
    echo -e "${RED}✗ Docker build failed${NC}"
    echo "  Check logs: $SSH_CMD 'tail -50 /tmp/docker_build.log'"
    exit 1
fi

# Step 4: Stop existing containers
echo ""
echo -e "${BLUE}Step 4: Stopping existing containers...${NC}"
$SSH_CMD "cd ${PROJECT_DIR} && docker-compose -f docker-compose.prod.yml down 2>/dev/null || true"
echo -e "${GREEN}✓ Containers stopped${NC}"

# Step 5: Start services
echo ""
echo -e "${BLUE}Step 5: Starting services...${NC}"
$SSH_CMD "cd ${PROJECT_DIR} && docker-compose -f docker-compose.prod.yml up -d"
sleep 5
echo -e "${GREEN}✓ Services started${NC}"

# Step 6: Wait for health check
echo ""
echo -e "${BLUE}Step 6: Waiting for health check...${NC}"
echo "  Waiting up to 5 minutes for models to load..."

MAX_WAIT=300
ELAPSED=0
HEALTHY=false

while [ $ELAPSED -lt $MAX_WAIT ]; do
    HEALTH_RESPONSE=$($SSH_CMD "curl -s -f http://localhost:8000/api/v1/health 2>/dev/null" || echo "")
    
    if [ -n "$HEALTH_RESPONSE" ]; then
        MODELS_LOADED=$(echo "$HEALTH_RESPONSE" | grep -o '"models_loaded":\s*true' || echo "")
        if [ -n "$MODELS_LOADED" ]; then
            echo -e "${GREEN}✓ Health check passed - models loaded${NC}"
            HEALTHY=true
            break
        else
            echo "  Models still loading... (${ELAPSED}s / ${MAX_WAIT}s)"
        fi
    else
        echo "  Waiting for API to start... (${ELAPSED}s)"
    fi
    
    sleep 10
    ELAPSED=$((ELAPSED + 10))
done

if [ "$HEALTHY" = false ]; then
    echo -e "${YELLOW}⚠ Health check timeout - check logs manually${NC}"
    echo "  Run: $SSH_CMD 'docker logs diffrhythm-api'"
fi

# Step 7: Run verification
echo ""
echo -e "${BLUE}Step 7: Running deployment verification...${NC}"
bash scripts/verify_server_deployment.sh

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "API URL: http://${SERVER_IP}:8000"
echo "Health: http://${SERVER_IP}:8000/api/v1/health"
echo "Docs: http://${SERVER_IP}:8000/docs"
echo ""
echo "To check logs:"
echo "  ssh -i ${SSH_KEY} ${SERVER_USER}@${SERVER_IP} 'docker logs -f diffrhythm-api'"
