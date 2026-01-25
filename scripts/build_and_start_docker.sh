#!/bin/bash
# Build and start Docker containers for DiffRhythm on the server
# Usage: bash scripts/build_and_start_docker.sh

set -e

# Configuration
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
echo "Building and Starting DiffRhythm Docker"
echo "=========================================="
echo "Server: $EC2_USER@$EC2_HOST"
echo ""

# Check SSH key
if [ ! -f "$SSH_KEY" ]; then
    echo -e "${RED}Error: SSH key not found at $SSH_KEY${NC}"
    exit 1
fi

# Step 1: Build Docker image
echo -e "${YELLOW}Step 1: Building Docker image...${NC}"
ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" << 'EOF'
cd /opt/diffrhythm
echo "Cleaning up old images..."
sudo docker builder prune -f
echo "Building new image..."
sudo docker build -f Dockerfile.prod -t diffrhythm:prod . 2>&1 | tail -20
EOF
echo -e "${GREEN}✓ Docker image built${NC}"

# Step 2: Verify image exists
echo ""
echo -e "${YELLOW}Step 2: Verifying Docker image...${NC}"
ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "sudo docker images | grep diffrhythm || echo 'Image not found'"

# Step 3: Stop any existing containers
echo ""
echo -e "${YELLOW}Step 3: Stopping existing containers...${NC}"
ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "cd /opt/diffrhythm && sudo docker-compose -f docker-compose.prod.yml down 2>&1 || true"

# Step 4: Start Docker containers
echo ""
echo -e "${YELLOW}Step 4: Starting Docker containers...${NC}"
ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" << 'EOF'
cd /opt/diffrhythm
echo "Starting services with docker-compose..."
sudo docker-compose -f docker-compose.prod.yml up -d
echo "Waiting for services to initialize..."
sleep 15
EOF
echo -e "${GREEN}✓ Docker containers started${NC}"

# Step 5: Check container status
echo ""
echo -e "${YELLOW}Step 5: Checking container status...${NC}"
ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "cd /opt/diffrhythm && sudo docker-compose -f docker-compose.prod.yml ps"

# Step 6: Check container logs
echo ""
echo -e "${YELLOW}Step 6: Checking container logs (last 30 lines)...${NC}"
ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "cd /opt/diffrhythm && sudo docker-compose -f docker-compose.prod.yml logs --tail=30 diffrhythm-api"

# Step 7: Health check
echo ""
echo -e "${YELLOW}Step 7: Performing health check...${NC}"
sleep 5
HEALTH_CHECK=$(ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "curl -s http://localhost:8000/api/v1/health 2>&1 || echo 'FAILED'")
if [[ "$HEALTH_CHECK" == *"status"* ]] || [[ "$HEALTH_CHECK" == *"healthy"* ]]; then
    echo -e "${GREEN}✓ Health check passed${NC}"
    echo "$HEALTH_CHECK" | python3 -m json.tool 2>/dev/null || echo "$HEALTH_CHECK"
else
    echo -e "${YELLOW}⚠ Health check returned: $HEALTH_CHECK${NC}"
    echo -e "${YELLOW}Service may still be starting. Check logs with:${NC}"
    echo "ssh -i $SSH_KEY $EC2_USER@$EC2_HOST 'cd $PROJECT_DIR && sudo docker-compose -f docker-compose.prod.yml logs -f diffrhythm-api'"
fi

# Step 8: Verify backend is running
echo ""
echo -e "${YELLOW}Step 8: Verifying backend endpoint...${NC}"
API_RESPONSE=$(ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "curl -s http://localhost:8000/ 2>&1 || echo 'FAILED'")
if [[ "$API_RESPONSE" == *"DiffRhythm"* ]] || [[ "$API_RESPONSE" == *"API"* ]]; then
    echo -e "${GREEN}✓ Backend is responding${NC}"
    echo "$API_RESPONSE" | head -5
else
    echo -e "${YELLOW}⚠ Backend response: $API_RESPONSE${NC}"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}Docker Build and Start Complete!${NC}"
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
