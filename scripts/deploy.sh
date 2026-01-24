#!/bin/bash
# DiffRhythm Production Deployment Script
# For AWS EC2 deployment

set -e

echo "=========================================="
echo "DiffRhythm Production Deployment"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/opt/diffrhythm"
SERVICE_NAME="diffrhythm-api"
USER="diffrhythm"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root${NC}"
    exit 1
fi

# Create user if doesn't exist
if ! id "$USER" &>/dev/null; then
    echo -e "${YELLOW}Creating user: $USER${NC}"
    useradd -m -s /bin/bash "$USER"
fi

# Create project directory
echo -e "${YELLOW}Creating project directory...${NC}"
mkdir -p "$PROJECT_DIR"
chown "$USER:$USER" "$PROJECT_DIR"

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Installing Docker...${NC}"
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    usermod -aG docker "$USER"
    systemctl enable docker
    systemctl start docker
    rm get-docker.sh
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}Installing Docker Compose...${NC}"
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

# Copy systemd service file
echo -e "${YELLOW}Setting up systemd service...${NC}"
cp "$PROJECT_DIR/config/systemd/diffrhythm.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

# Create directories
echo -e "${YELLOW}Creating required directories...${NC}"
mkdir -p "$PROJECT_DIR/output" "$PROJECT_DIR/pretrained" "$PROJECT_DIR/temp"
chown -R "$USER:$USER" "$PROJECT_DIR"

# Set up environment file
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cp "$PROJECT_DIR/config/ec2-config.env" "$PROJECT_DIR/.env"
    chown "$USER:$USER" "$PROJECT_DIR/.env"
    echo -e "${YELLOW}Please edit $PROJECT_DIR/.env with your configuration${NC}"
fi

echo -e "${GREEN}Deployment setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Copy your project files to $PROJECT_DIR"
echo "2. Edit $PROJECT_DIR/.env with your settings"
echo "3. Run: sudo systemctl start $SERVICE_NAME"
echo "4. Check status: sudo systemctl status $SERVICE_NAME"
