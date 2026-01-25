#!/bin/bash
# Stripe Payment System Investigation Script
# Run this on the server to investigate Stripe payment implementation

set -e

echo "=========================================="
echo "Stripe Payment System Investigation"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

BACKEND_DIR="/home/ubuntu/app/backend"

echo -e "${YELLOW}1. Checking Stripe Configuration${NC}"
echo "----------------------------------------"
if [ -f "$BACKEND_DIR/.env" ]; then
    echo "Stripe keys in .env:"
    grep -i stripe "$BACKEND_DIR/.env" || echo -e "${RED}No Stripe keys found in .env${NC}"
else
    echo -e "${RED}.env file not found${NC}"
fi
echo ""

echo -e "${YELLOW}2. Finding Payment-Related Files${NC}"
echo "----------------------------------------"
echo "Payment files:"
find "$BACKEND_DIR" -name '*payment*' -o -name '*stripe*' 2>/dev/null | head -20
echo ""

echo -e "${YELLOW}3. Checking Payment Code${NC}"
echo "----------------------------------------"
if [ -f "$BACKEND_DIR/src/api/payments.py" ]; then
    echo "Found: src/api/payments.py"
    echo "First 50 lines:"
    head -50 "$BACKEND_DIR/src/api/payments.py"
else
    echo -e "${RED}src/api/payments.py not found${NC}"
fi
echo ""

if [ -f "$BACKEND_DIR/src/api/v1/payments.py" ]; then
    echo "Found: src/api/v1/payments.py"
    echo "First 50 lines:"
    head -50 "$BACKEND_DIR/src/api/v1/payments.py"
else
    echo -e "${RED}src/api/v1/payments.py not found${NC}"
fi
echo ""

echo -e "${YELLOW}4. Searching for Pricing Logic${NC}"
echo "----------------------------------------"
echo "Pricing calculation code:"
grep -r 'calculate.*price\|price.*duration\|charge.*song\|stripe.*amount' "$BACKEND_DIR/src" --include='*.py' 2>/dev/null | head -20 || echo "No pricing logic found"
echo ""

echo -e "${YELLOW}5. Checking Generation Endpoint for Payment Integration${NC}"
echo "----------------------------------------"
if [ -f "$BACKEND_DIR/src/api/v1/generation.py" ]; then
    echo "Checking generation endpoint for payment calls:"
    grep -A 10 -B 5 'payment\|stripe\|charge' "$BACKEND_DIR/src/api/v1/generation.py" || echo "No payment integration in generation endpoint"
fi
echo ""

echo -e "${YELLOW}6. Checking Service Status${NC}"
echo "----------------------------------------"
systemctl status burntbeats-api --no-pager | head -15
echo ""

echo -e "${YELLOW}7. Checking Service Logs for Stripe Errors${NC}"
echo "----------------------------------------"
sudo journalctl -u burntbeats-api --no-pager -n 100 | grep -i stripe || echo "No Stripe-related logs found"
echo ""

echo -e "${YELLOW}8. Testing Payment Endpoints${NC}"
echo "----------------------------------------"
echo "Testing /api/v1/payments/checkout:"
curl -s http://127.0.0.1:8001/api/v1/payments/checkout 2>&1 | head -20 || echo "Endpoint not accessible"
echo ""

echo "Testing /api/v1/payments/webhook:"
curl -s http://127.0.0.1:8001/api/v1/payments/webhook 2>&1 | head -20 || echo "Endpoint not accessible"
echo ""

echo -e "${YELLOW}9. Checking Python Dependencies${NC}"
echo "----------------------------------------"
if [ -f "$BACKEND_DIR/requirements.txt" ]; then
    echo "Stripe in requirements:"
    grep -i stripe "$BACKEND_DIR/requirements.txt" || echo -e "${RED}Stripe not in requirements.txt${NC}"
fi
echo ""

echo -e "${YELLOW}10. Checking Installed Packages${NC}"
echo "----------------------------------------"
python3 -c "import stripe; print(f'Stripe version: {stripe.__version__}')" 2>&1 || echo -e "${RED}Stripe package not installed${NC}"
echo ""

echo "=========================================="
echo "Investigation Complete"
echo "=========================================="
