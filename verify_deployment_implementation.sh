#!/bin/bash
# Verify Deployment Implementation
# Tests that all code changes are correctly implemented

set -e

echo "=========================================="
echo "Deployment Implementation Verification"
echo "=========================================="
echo ""

# Check if files exist
echo "[1] Checking implementation files..."
FILES=(
    "backend/api.py"
    "backend/config.py"
    "backend/payment_verification.py"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file exists"
    else
        echo "  ❌ $file missing"
        exit 1
    fi
done

# Check Python syntax
echo ""
echo "[2] Checking Python syntax..."
python3 -m py_compile backend/api.py && echo "  ✅ backend/api.py syntax OK" || exit 1
python3 -m py_compile backend/config.py && echo "  ✅ backend/config.py syntax OK" || exit 1
python3 -m py_compile backend/payment_verification.py && echo "  ✅ backend/payment_verification.py syntax OK" || exit 1

# Check for required imports
echo ""
echo "[3] Checking imports..."
if grep -q "from backend.payment_verification import" backend/api.py; then
    echo "  ✅ Payment verification imported"
else
    echo "  ❌ Payment verification not imported"
    exit 1
fi

if grep -q "payment_intent_id" backend/api.py; then
    echo "  ✅ payment_intent_id field found"
else
    echo "  ❌ payment_intent_id field missing"
    exit 1
fi

if grep -q "/api/webhooks/stripe" backend/api.py; then
    echo "  ✅ Webhook endpoint found"
else
    echo "  ❌ Webhook endpoint missing"
    exit 1
fi

# Check quality defaults
echo ""
echo "[4] Checking quality defaults..."
if grep -q 'preset.*=.*"high"' backend/api.py || grep -q "preset.*Field.*high" backend/api.py; then
    echo "  ✅ Default preset set to 'high'"
else
    echo "  ⚠️  Default preset may not be set to 'high'"
fi

if grep -q "auto_master.*=.*True" backend/api.py || grep -q "auto_master.*Field.*True" backend/api.py; then
    echo "  ✅ Auto-mastering enabled by default"
else
    echo "  ⚠️  Auto-mastering may not be enabled by default"
fi

# Check config variables
echo ""
echo "[5] Checking configuration variables..."
if grep -q "STRIPE_SECRET_KEY" backend/config.py; then
    echo "  ✅ STRIPE_SECRET_KEY configured"
else
    echo "  ❌ STRIPE_SECRET_KEY missing"
    exit 1
fi

if grep -q "STRIPE_WEBHOOK_SECRET" backend/config.py; then
    echo "  ✅ STRIPE_WEBHOOK_SECRET configured"
else
    echo "  ❌ STRIPE_WEBHOOK_SECRET missing"
    exit 1
fi

if grep -q "REQUIRE_PAYMENT_FOR_GENERATION" backend/config.py; then
    echo "  ✅ REQUIRE_PAYMENT_FOR_GENERATION configured"
else
    echo "  ❌ REQUIRE_PAYMENT_FOR_GENERATION missing"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ All implementation checks passed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Deploy to server"
echo "2. Restart service: sudo systemctl restart burntbeats-api"
echo "3. Run: python3 test_server_implementation.py"
echo "4. Configure Stripe webhook in Dashboard"
