#!/bin/bash
# Deploy DiffRhythm to remote EC2 server from local machine
# Usage: bash scripts/deploy-to-server.sh

set -e

# Configuration
EC2_HOST="52.0.207.242"
EC2_USER="ubuntu"
SSH_KEY="C:/Users/sammy/.ssh/server_saver_key"
PROJECT_DIR="/opt/diffrhythm"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=========================================="
echo "DiffRhythm Remote Deployment"
echo "=========================================="
echo "Server: $EC2_USER@$EC2_HOST"
echo "Key: $SSH_KEY"
echo ""

# Convert Windows path to WSL/Linux path if needed
SSH_KEY_ORIG="$SSH_KEY"
if [[ "$SSH_KEY" == *"C:"* ]] || [[ "$SSH_KEY" == *"C:/"* ]]; then
    # Running in WSL - convert Windows path
    SSH_KEY_WSL=$(echo "$SSH_KEY" | sed 's|C:|/mnt/c|' | sed 's|\\|/|g' | tr '[:upper:]' '[:lower:]')
    if [ -f "$SSH_KEY_WSL" ]; then
        # Copy to home directory with proper permissions (WSL can't chmod Windows files)
        SSH_KEY="$HOME/server_saver_key_deploy"
        cp "$SSH_KEY_WSL" "$SSH_KEY"
        chmod 600 "$SSH_KEY"
    elif [ -f "/mnt/c/Users/sammy/.ssh/server_saver_key" ]; then
        SSH_KEY="$HOME/server_saver_key_deploy"
        cp "/mnt/c/Users/sammy/.ssh/server_saver_key" "$SSH_KEY"
        chmod 600 "$SSH_KEY"
    fi
fi

# Check SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo -e "${RED}Error: SSH key not found at $SSH_KEY_ORIG${NC}"
    exit 1
fi

# Test SSH connection
echo -e "${YELLOW}Testing SSH connection...${NC}"
if ! ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$EC2_USER@$EC2_HOST" "echo 'Connection successful'" 2>/dev/null; then
    echo -e "${RED}Error: Cannot connect to server${NC}"
    exit 1
fi
echo -e "${GREEN}✓ SSH connection successful${NC}"

# Step 1: Run initial setup on server
echo ""
echo -e "${YELLOW}Step 1: Running initial server setup...${NC}"
ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "sudo bash -s" < scripts/ec2-setup.sh
echo -e "${GREEN}✓ Server setup complete${NC}"

# Step 2: Create project directory
echo ""
echo -e "${YELLOW}Step 2: Creating project directory...${NC}"
ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "sudo mkdir -p $PROJECT_DIR && sudo chown $EC2_USER:$EC2_USER $PROJECT_DIR"
echo -e "${GREEN}✓ Project directory created${NC}"

# Step 3: Copy project files (excluding unnecessary files)
echo ""
echo -e "${YELLOW}Step 3: Copying project files to server...${NC}"
rsync -avz --progress \
    -e "ssh -i $SSH_KEY" \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude 'venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '*.log' \
    --exclude 'output/' \
    --exclude 'temp/' \
    --exclude '.pytest_cache' \
    --exclude 'node_modules' \
    --exclude '.idea' \
    --exclude '.vscode' \
    ./ "$EC2_USER@$EC2_HOST:$PROJECT_DIR/"
echo -e "${GREEN}✓ Files copied${NC}"

# Step 4: Set up environment file
echo ""
echo -e "${YELLOW}Step 4: Setting up environment configuration...${NC}"
ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" << EOF
cd $PROJECT_DIR
if [ ! -f .env ]; then
    cp config/ec2-config.env .env
    echo "✓ Created .env file"
else
    echo "✓ .env file already exists"
fi
EOF
echo -e "${GREEN}✓ Environment configured${NC}"

# Step 5: Build and start Docker containers
echo ""
echo -e "${YELLOW}Step 5: Building and starting Docker containers...${NC}"
ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" << EOF
cd $PROJECT_DIR
echo "Building Docker image..."
sudo docker build -f Dockerfile.prod -t diffrhythm:prod .
echo "Starting services..."
sudo docker-compose -f docker-compose.prod.yml up -d
echo "Waiting for services to start..."
sleep 10
EOF
echo -e "${GREEN}✓ Docker containers started${NC}"

# Step 6: Verify deployment
echo ""
echo -e "${YELLOW}Step 6: Verifying deployment...${NC}"
sleep 5
HEALTH_CHECK=$(ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "curl -s http://localhost:8000/api/v1/health || echo 'FAILED'")
if [[ "$HEALTH_CHECK" == *"status"* ]]; then
    echo -e "${GREEN}✓ Health check passed${NC}"
    echo "$HEALTH_CHECK" | python3 -m json.tool 2>/dev/null || echo "$HEALTH_CHECK"
else
    echo -e "${YELLOW}⚠ Health check returned: $HEALTH_CHECK${NC}"
    echo -e "${YELLOW}Service may still be starting. Check logs with:${NC}"
    echo "ssh -i $SSH_KEY $EC2_USER@$EC2_HOST 'cd $PROJECT_DIR && sudo docker-compose -f docker-compose.prod.yml logs'"
fi

# Step 7: Show status
echo ""
echo -e "${YELLOW}Step 7: Service status...${NC}"
ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "cd $PROJECT_DIR && sudo docker-compose -f docker-compose.prod.yml ps"

echo ""
echo "=========================================="
echo -e "${GREEN}Deployment Complete!${NC}"
echo "=========================================="
echo ""
echo "API Endpoint: http://$EC2_HOST:8000"
echo "Health Check: http://$EC2_HOST:8000/api/v1/health"
echo "API Docs: http://$EC2_HOST:8000/docs"
echo ""
echo "Useful commands:"
echo "  View logs: ssh -i $SSH_KEY $EC2_USER@$EC2_HOST 'cd $PROJECT_DIR && sudo docker-compose -f docker-compose.prod.yml logs -f'"
echo "  Restart: ssh -i $SSH_KEY $EC2_USER@$EC2_HOST 'cd $PROJECT_DIR && sudo docker-compose -f docker-compose.prod.yml restart'"
echo "  Stop: ssh -i $SSH_KEY $EC2_USER@$EC2_HOST 'cd $PROJECT_DIR && sudo docker-compose -f docker-compose.prod.yml down'"
echo ""
