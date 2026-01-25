#!/bin/bash
# E2E check: health + payment-required behaviour.
# Run after deploy when API is reachable.
# Usage: BASE_URL=http://52.0.207.242:8000 bash scripts/e2e-payment-generation-check.sh

set -e

BASE_URL="${BASE_URL:-http://52.0.207.242:8000}"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "E2E: Health + Payment → Generation"
echo "=========================================="
echo "API: $BASE_URL"
echo ""

# 1. Health check
echo -e "${YELLOW}[1] Health check...${NC}"
HEALTH=$(curl -s -w "\n%{http_code}" --connect-timeout 15 "$BASE_URL/api/v1/health" || true)
HTTP_CODE=$(echo "$HEALTH" | tail -n1)
BODY=$(echo "$HEALTH" | sed '$d')

if [ "$HTTP_CODE" = "200" ] && echo "$BODY" | grep -q "status"; then
    echo -e "${GREEN}✓ Health OK${NC}"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
else
    echo -e "${RED}✗ Health failed (HTTP $HTTP_CODE)${NC}"
    echo "$BODY"
    echo ""
    echo "Ensure API is running and reachable, then re-run."
    exit 1
fi

# 2. Generate without payment_intent_id (expect 402 if payment required)
echo ""
echo -e "${YELLOW}[2] Generate without payment_intent_id (expect 402 if REQUIRE_PAYMENT=true)...${NC}"
GEN_RESP=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/api/v1/generate" \
    -H "Content-Type: application/json" \
    -d '{"lyrics":"[00:00.00]Test\n[00:05.00]Line two","style_prompt":"pop","audio_length":95}' \
    --connect-timeout 15 2>/dev/null || true)
GEN_CODE=$(echo "$GEN_RESP" | tail -n1)
GEN_BODY=$(echo "$GEN_RESP" | sed '$d')

if [ "$GEN_CODE" = "402" ]; then
    echo -e "${GREEN}✓ 402 Payment Required (payment check active)${NC}"
elif [ "$GEN_CODE" = "200" ] || [ "$GEN_CODE" = "201" ]; then
    echo -e "${YELLOW}⚠ Generate accepted without payment_intent_id (REQUIRE_PAYMENT=false or no auth)${NC}"
else
    echo "  HTTP $GEN_CODE: $GEN_BODY"
fi

echo ""
echo "=========================================="
echo "Full E2E (payment → generation):"
echo "  1. Create payment: Stripe Dashboard or \`stripe payment_intents create --amount=200 --currency=usd --confirm\`"
echo "  2. Use PaymentIntent ID (pi_...) in:"
echo "     curl -X POST $BASE_URL/api/v1/generate \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"lyrics\":\"[00:00.00]Line one\\n[00:05.00]Line two\",\"style_prompt\":\"pop\",\"audio_length\":95,\"payment_intent_id\":\"pi_XXX\"}'"
echo "  3. Poll /api/v1/status/{job_id} then download output."
echo "=========================================="
