#!/bin/bash
# Server Deployment Verification Script
# Uses SSH to verify deployment status on remote server

set -e

# Configuration
SERVER_IP="${SERVER_IP:-52.0.207.242}"
SERVER_USER="${SERVER_USER:-ubuntu}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/server_saver_key}"
PROJECT_DIR="/opt/diffrhythm"
CONTAINER_NAME="diffrhythm-api"
IMAGE_NAME="diffrhythm:prod"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# SSH command helper
SSH_CMD="ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no ${SERVER_USER}@${SERVER_IP}"

echo "=========================================="
echo "DiffRhythm Server Deployment Verification"
echo "=========================================="
echo "Server: ${SERVER_USER}@${SERVER_IP}"
echo "Project: ${PROJECT_DIR}"
echo ""

# Function to run command on server
run_remote() {
    $SSH_CMD "$1"
}

# Function to check if command succeeded
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${RED}✗${NC} $1"
        return 1
    fi
}

# Function to print section header
print_section() {
    echo ""
    echo -e "${BLUE}=== $1 ===${NC}"
    echo ""
}

# Track overall status
OVERALL_STATUS=0

# Phase 1: Server Connectivity
print_section "Phase 1: Server Connectivity"
if run_remote "echo 'Connected'" > /dev/null 2>&1; then
    check_status "SSH connection successful"
else
    echo -e "${RED}✗${NC} SSH connection failed"
    echo "Please check:"
    echo "  - SSH key: ${SSH_KEY}"
    echo "  - Server IP: ${SERVER_IP}"
    echo "  - Server user: ${SERVER_USER}"
    exit 1
fi

# Phase 2: Docker Installation
print_section "Phase 2: Docker Installation"
DOCKER_VERSION=$(run_remote "docker --version 2>/dev/null" || echo "")
if [ -n "$DOCKER_VERSION" ]; then
    check_status "Docker installed: $DOCKER_VERSION"
else
    check_status "Docker not installed"
    OVERALL_STATUS=1
fi

DOCKER_COMPOSE_VERSION=$(run_remote "docker-compose --version 2>/dev/null || docker compose version 2>/dev/null" || echo "")
if [ -n "$DOCKER_COMPOSE_VERSION" ]; then
    check_status "Docker Compose available: $DOCKER_COMPOSE_VERSION"
else
    check_status "Docker Compose not available"
    OVERALL_STATUS=1
fi

# Phase 3: Project Directory
print_section "Phase 3: Project Directory"
if run_remote "test -d ${PROJECT_DIR}" > /dev/null 2>&1; then
    check_status "Project directory exists: ${PROJECT_DIR}"
    DIR_SIZE=$(run_remote "du -sh ${PROJECT_DIR} 2>/dev/null | cut -f1" || echo "unknown")
    echo "  Directory size: $DIR_SIZE"
else
    check_status "Project directory not found: ${PROJECT_DIR}"
    OVERALL_STATUS=1
fi

# Phase 4: Docker Image
print_section "Phase 4: Docker Image"
IMAGE_EXISTS=$(run_remote "docker images ${IMAGE_NAME} --format '{{.Repository}}:{{.Tag}}' 2>/dev/null" || echo "")
if [ -n "$IMAGE_EXISTS" ]; then
    check_status "Docker image exists: ${IMAGE_NAME}"
    IMAGE_SIZE=$(run_remote "docker images ${IMAGE_NAME} --format '{{.Size}}' 2>/dev/null" || echo "unknown")
    echo "  Image size: $IMAGE_SIZE"
else
    check_status "Docker image not found: ${IMAGE_NAME}"
    echo "  Run: docker build -f Dockerfile.prod -t ${IMAGE_NAME} ."
    OVERALL_STATUS=1
fi

# Phase 5: Docker Container
print_section "Phase 5: Docker Container"
CONTAINER_STATUS=$(run_remote "docker ps -a --filter name=${CONTAINER_NAME} --format '{{.Status}}' 2>/dev/null" || echo "")
if [ -n "$CONTAINER_STATUS" ]; then
    check_status "Container exists: ${CONTAINER_NAME}"
    echo "  Status: $CONTAINER_STATUS"
    
    # Check if running
    IS_RUNNING=$(run_remote "docker ps --filter name=${CONTAINER_NAME} --format '{{.Names}}' 2>/dev/null" || echo "")
    if [ -n "$IS_RUNNING" ]; then
        check_status "Container is running"
    else
        check_status "Container is not running"
        echo "  Start with: docker-compose -f docker-compose.prod.yml up -d"
        OVERALL_STATUS=1
    fi
else
    check_status "Container not found: ${CONTAINER_NAME}"
    OVERALL_STATUS=1
fi

# Phase 6: Container Health
print_section "Phase 6: Container Health"
if [ -n "$IS_RUNNING" ]; then
    HEALTH_STATUS=$(run_remote "docker inspect ${CONTAINER_NAME} --format '{{.State.Health.Status}}' 2>/dev/null" || echo "unknown")
    if [ "$HEALTH_STATUS" = "healthy" ]; then
        check_status "Container health: healthy"
    elif [ "$HEALTH_STATUS" = "starting" ]; then
        echo -e "${YELLOW}⚠${NC} Container health: starting (may need more time)"
    else
        check_status "Container health: $HEALTH_STATUS"
        OVERALL_STATUS=1
    fi
    
    # Check container logs for errors
    echo ""
    echo "Recent container logs:"
    run_remote "docker logs ${CONTAINER_NAME} --tail 20 2>&1" | head -20
fi

# Phase 7: API Health Check
print_section "Phase 7: API Health Check"
if [ -n "$IS_RUNNING" ]; then
    HEALTH_RESPONSE=$(run_remote "curl -s -f http://localhost:8000/api/v1/health 2>/dev/null" || echo "")
    if [ -n "$HEALTH_RESPONSE" ]; then
        check_status "API health endpoint responding"
        echo "  Response:"
        echo "$HEALTH_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$HEALTH_RESPONSE"
        
        # Check models_loaded status
        MODELS_LOADED=$(echo "$HEALTH_RESPONSE" | grep -o '"models_loaded":\s*true' || echo "")
        if [ -n "$MODELS_LOADED" ]; then
            check_status "Models loaded successfully"
        else
            check_status "Models not loaded yet"
            echo "  Models may still be loading (takes 2-5 minutes)"
        fi
    else
        check_status "API health endpoint not responding"
        OVERALL_STATUS=1
    fi
else
    echo -e "${YELLOW}⚠${NC} Skipping API check (container not running)"
    OVERALL_STATUS=1
fi

# Phase 8: Disk Space
print_section "Phase 8: Disk Space"
DISK_USAGE=$(run_remote "df -h / | tail -1 | awk '{print \$5}' | sed 's/%//'" || echo "unknown")
if [ "$DISK_USAGE" != "unknown" ]; then
    if [ "$DISK_USAGE" -lt 80 ]; then
        check_status "Disk usage: ${DISK_USAGE}% (OK)"
    elif [ "$DISK_USAGE" -lt 90 ]; then
        echo -e "${YELLOW}⚠${NC} Disk usage: ${DISK_USAGE}% (Warning)"
    else
        check_status "Disk usage: ${DISK_USAGE}% (Critical)"
        OVERALL_STATUS=1
    fi
    DISK_FREE=$(run_remote "df -h / | tail -1 | awk '{print \$4}'" || echo "unknown")
    echo "  Free space: $DISK_FREE"
else
    check_status "Could not check disk usage"
fi

# Phase 9: Port Accessibility
print_section "Phase 9: Port Accessibility"
PORT_8000=$(run_remote "netstat -tuln 2>/dev/null | grep ':8000' || ss -tuln 2>/dev/null | grep ':8000'" || echo "")
if [ -n "$PORT_8000" ]; then
    check_status "Port 8000 is listening"
    echo "  $PORT_8000"
else
    check_status "Port 8000 is not listening"
    OVERALL_STATUS=1
fi

# Phase 10: Test Generation Endpoint
print_section "Phase 10: Test Generation Endpoint"
if [ -n "$IS_RUNNING" ] && [ -n "$HEALTH_RESPONSE" ]; then
    # Check if models are loaded before testing
    if [ -n "$MODELS_LOADED" ]; then
        echo "Testing generation endpoint (this will create a test job)..."
        TEST_RESPONSE=$(run_remote "curl -s -X POST http://localhost:8000/api/v1/generate \
            -H 'Content-Type: application/json' \
            -d '{\"lyrics\":\"[00:00.00]Test\n[00:05.00]Song\",\"style_prompt\":\"pop\",\"audio_length\":95}' 2>/dev/null" || echo "")
        
        if [ -n "$TEST_RESPONSE" ]; then
            echo "$TEST_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$TEST_RESPONSE"
            JOB_ID=$(echo "$TEST_RESPONSE" | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4 || echo "")
            if [ -n "$JOB_ID" ]; then
                check_status "Generation endpoint working (job_id: ${JOB_ID:0:8}...)"
            else
                check_status "Generation endpoint responded but no job_id found"
            fi
        else
            check_status "Generation endpoint not responding"
            OVERALL_STATUS=1
        fi
    else
        echo -e "${YELLOW}⚠${NC} Skipping generation test (models not loaded)"
    fi
else
    echo -e "${YELLOW}⚠${NC} Skipping generation test (API not available)"
fi

# Summary
echo ""
echo "=========================================="
echo "Deployment Verification Summary"
echo "=========================================="
echo ""

if [ $OVERALL_STATUS -eq 0 ]; then
    echo -e "${GREEN}✓ Deployment appears successful!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Test from frontend"
    echo "  2. Monitor logs: docker logs -f ${CONTAINER_NAME}"
    echo "  3. Check metrics: curl http://${SERVER_IP}:8000/api/v1/metrics"
else
    echo -e "${RED}✗ Some issues detected${NC}"
    echo ""
    echo "Please review the errors above and:"
    echo "  1. Check container logs: docker logs ${CONTAINER_NAME}"
    echo "  2. Verify Docker image: docker images ${IMAGE_NAME}"
    echo "  3. Check disk space: df -h /"
    echo "  4. Restart services: docker-compose -f docker-compose.prod.yml restart"
fi

echo ""
exit $OVERALL_STATUS
