# Stripe Payment System Investigation

**Date:** January 23, 2026  
**Server:** ubuntu@52.0.207.242  
**Service:** burntbeats-api (port 8001)

## Investigation Plan

### 1. Check Stripe Configuration

```bash
# Check environment variables
cat /home/ubuntu/app/backend/.env | grep -i stripe

# Check if Stripe keys are set
echo $STRIPE_SECRET_KEY
echo $STRIPE_PUBLISHABLE_KEY
echo $STRIPE_WEBHOOK_SECRET
```

### 2. Locate Payment Code

```bash
# Find payment-related files
find /home/ubuntu/app/backend -name '*payment*' -o -name '*stripe*'

# Check payment endpoints
grep -r 'stripe\|STRIPE\|payment' /home/ubuntu/app/backend/src --include='*.py'

# Check payment routes
cat /home/ubuntu/app/backend/src/api/payments.py
cat /home/ubuntu/app/backend/src/api/v1/payments.py
```

### 3. Verify Pricing Logic

```bash
# Check pricing calculation based on duration
grep -r 'calculate.*price\|price.*duration\|charge.*song' /home/ubuntu/app/backend/src --include='*.py'

# Check generation endpoint for pricing integration
grep -A 20 'def.*generate' /home/ubuntu/app/backend/src/api/v1/generation.py
```

### 4. Test Payment Endpoints

```bash
# Check if payment endpoints are accessible
curl http://127.0.0.1:8001/api/v1/payments/checkout
curl http://127.0.0.1:8001/api/v1/payments/webhook

# Check service logs for Stripe errors
sudo journalctl -u burntbeats-api | grep -i stripe
```

## Expected Findings

Based on service logs, we saw:
- `WARNING:src.api.payments:⚠️  Stripe not configured`
- `WARNING:src.api.v1.payments:⚠️  Stripe not configured`

This suggests:
1. Stripe code exists but is not configured
2. Payment endpoints may be disabled
3. Need to verify pricing logic is implemented

## Key Files to Check

1. `/home/ubuntu/app/backend/src/api/payments.py`
2. `/home/ubuntu/app/backend/src/api/v1/payments.py`
3. `/home/ubuntu/app/backend/src/api/v1/generation.py` (for pricing integration)
4. `/home/ubuntu/app/backend/.env` (for Stripe keys)
5. `/home/ubuntu/app/backend/src/services/payment_service.py` (if exists)

## Pricing Model to Verify

Based on song duration:
- **95 seconds (base)**: $X.XX
- **96-285 seconds (extended)**: $Y.YY per second or tiered pricing

## Next Steps

1. SSH to server and run investigation commands
2. Review payment code implementation
3. Verify pricing calculation logic
4. Check Stripe webhook configuration
5. Test payment flow end-to-end
6. Document findings and recommendations
