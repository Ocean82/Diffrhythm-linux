#!/bin/bash
# Helper script to update Stripe keys in .env file
# Usage: ./update_stripe_keys.sh

ENV_FILE="/home/ubuntu/app/backend/.env"

echo "=========================================="
echo "Stripe Keys Update Script"
echo "=========================================="
echo ""

# Check if .env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Error: .env file not found at $ENV_FILE"
    exit 1
fi

echo "Current Stripe configuration:"
grep -E "STRIPE|REQUIRE_PAYMENT" "$ENV_FILE" || echo "No Stripe keys found"
echo ""

# Prompt for keys
read -p "Enter Stripe Secret Key (sk_test_... or sk_live_...): " SECRET_KEY
read -p "Enter Stripe Publishable Key (pk_test_... or pk_live_...): " PUBLISHABLE_KEY
read -p "Enter Stripe Webhook Secret (whsec_...): " WEBHOOK_SECRET
read -p "Require payment for generation? (true/false) [default: false]: " REQUIRE_PAYMENT

REQUIRE_PAYMENT=${REQUIRE_PAYMENT:-false}

# Validate keys
if [[ ! "$SECRET_KEY" =~ ^sk_(test|live)_ ]]; then
    echo "❌ Error: Invalid secret key format. Must start with sk_test_ or sk_live_"
    exit 1
fi

if [[ ! "$PUBLISHABLE_KEY" =~ ^pk_(test|live)_ ]]; then
    echo "❌ Error: Invalid publishable key format. Must start with pk_test_ or pk_live_"
    exit 1
fi

if [[ ! "$WEBHOOK_SECRET" =~ ^whsec_ ]]; then
    echo "❌ Error: Invalid webhook secret format. Must start with whsec_"
    exit 1
fi

# Backup .env file
cp "$ENV_FILE" "${ENV_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
echo "✅ Backup created: ${ENV_FILE}.backup.$(date +%Y%m%d_%H%M%S)"

# Update or add Stripe keys
if grep -q "STRIPE_SECRET_KEY=" "$ENV_FILE"; then
    # Update existing keys
    sed -i "s|STRIPE_SECRET_KEY=.*|STRIPE_SECRET_KEY=$SECRET_KEY|" "$ENV_FILE"
    sed -i "s|STRIPE_PUBLISHABLE_KEY=.*|STRIPE_PUBLISHABLE_KEY=$PUBLISHABLE_KEY|" "$ENV_FILE"
    sed -i "s|STRIPE_WEBHOOK_SECRET=.*|STRIPE_WEBHOOK_SECRET=$WEBHOOK_SECRET|" "$ENV_FILE"
    if grep -q "REQUIRE_PAYMENT_FOR_GENERATION=" "$ENV_FILE"; then
        sed -i "s|REQUIRE_PAYMENT_FOR_GENERATION=.*|REQUIRE_PAYMENT_FOR_GENERATION=$REQUIRE_PAYMENT|" "$ENV_FILE"
    else
        echo "REQUIRE_PAYMENT_FOR_GENERATION=$REQUIRE_PAYMENT" >> "$ENV_FILE"
    fi
else
    # Add new keys
    echo "" >> "$ENV_FILE"
    echo "# Stripe Configuration" >> "$ENV_FILE"
    echo "STRIPE_SECRET_KEY=$SECRET_KEY" >> "$ENV_FILE"
    echo "STRIPE_PUBLISHABLE_KEY=$PUBLISHABLE_KEY" >> "$ENV_FILE"
    echo "STRIPE_WEBHOOK_SECRET=$WEBHOOK_SECRET" >> "$ENV_FILE"
    echo "REQUIRE_PAYMENT_FOR_GENERATION=$REQUIRE_PAYMENT" >> "$ENV_FILE"
fi

echo ""
echo "✅ Stripe keys updated successfully!"
echo ""
echo "Updated configuration:"
grep -E "STRIPE|REQUIRE_PAYMENT" "$ENV_FILE"
echo ""
echo "Next steps:"
echo "1. Restart the service: sudo systemctl restart burntbeats-api"
echo "2. Verify configuration: curl http://127.0.0.1:8001/api/v1/payments/calculate-price?duration=120"
echo ""
