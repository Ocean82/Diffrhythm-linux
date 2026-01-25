# Stripe Webhook and Payment System - Final Setup Guide

**Date:** January 23, 2026  
**Status:** ✅ **IMPLEMENTATION COMPLETE, READY FOR STRIPE DASHBOARD CONFIGURATION**

## System Status

### ✅ Implementation Complete

All payment system components are implemented and configured:

1. **Pricing System** ✅
   - Duration-based pricing implemented
   - All pricing tiers supported
   - Tested and verified

2. **Payment Endpoints** ✅
   - Calculate price: Working
   - Create payment intent: Ready
   - Verify payment: Ready

3. **Generation Integration** ✅
   - Payment verification before generation
   - Price validation
   - Payment metadata storage

4. **Webhook Handler** ✅
   - Endpoint accessible
   - Signature verification implemented
   - Event handling ready

5. **Stripe Configuration** ✅
   - Live keys configured
   - Webhook secret configured
   - Service running

## Stripe Dashboard Configuration

### Step 1: Create Webhook Endpoint

1. **Go to Stripe Dashboard**
   - URL: https://dashboard.stripe.com
   - Log in to your account
   - Switch to **Live mode**

2. **Navigate to Webhooks**
   - Click **Developers** → **Webhooks**
   - Click **Add endpoint**

3. **Configure Endpoint**
   - **Endpoint URL:** `https://burntbeats.com/api/webhooks/stripe`
   - Click **Add endpoint**

4. **Select Events**
   Check these events:
   - ✅ `payment_intent.succeeded`
   - ✅ `payment_intent.payment_failed`
   - ✅ `payment_intent.canceled`

5. **Get Signing Secret**
   - Click on your webhook endpoint
   - Click **Reveal** next to **Signing secret**
   - Copy the secret
   - **Verify it matches:** `whsec_nCaUM9ArPRjwqAa1lieItdDevmBasGTI`
   - If different, update `.env` file

### Step 2: Verify Configuration

- Endpoint status: Should be **Active**
- Endpoint URL: `https://burntbeats.com/api/webhooks/stripe`
- Events selected: `payment_intent.succeeded`, `payment_intent.payment_failed`
- Webhook secret matches `.env` file

## Payment Flow Testing

### Test 1: Calculate Price ✅

```bash
curl http://127.0.0.1:8001/api/v1/payments/calculate-price?duration=120
```

**Expected Result:**
```json
{
  "base_price_dollars": 2.0,
  "with_commercial_license_dollars": 17.0,
  "bulk_10_price_dollars": 18.0,
  "bulk_50_price_dollars": 80.0
}
```

**Status:** ✅ Working

### Test 2: Create Payment Intent

**Option A: Using Stripe CLI (for testing)**

```bash
# Login to Stripe CLI first
stripe login

# Create test payment intent
stripe payment_intents create \
  --amount=200 \
  --currency=usd \
  --confirm
```

**Option B: Using API Endpoint (requires auth token)**

```bash
curl -X POST http://127.0.0.1:8001/api/v1/payments/create-intent \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "duration_seconds": 120,
    "amount_cents": 200,
    "currency": "usd"
  }'
```

### Test 3: Verify Payment Intent

```bash
curl http://127.0.0.1:8001/api/v1/payments/verify-payment/pi_xxx \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Test 4: Generate Song (with Payment)

```bash
curl -X POST http://127.0.0.1:8001/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "text_prompt": "A happy pop song",
    "duration": 120,
    "payment_intent_id": "pi_xxx"
  }'
```

## Webhook Testing

### Method 1: Stripe CLI Listener (Recommended)

**Step 1: Start Webhook Listener**
```bash
stripe listen --forward-to http://127.0.0.1:8001/api/webhooks/stripe
```

**Output:**
```
> Ready! Your webhook signing secret is whsec_xxx
```

**Step 2: Trigger Test Events**

In another terminal:
```bash
# Test successful payment
stripe trigger payment_intent.succeeded

# Test failed payment
stripe trigger payment_intent.payment_failed

# Test canceled payment
stripe trigger payment_intent.canceled
```

**Step 3: Monitor Logs**
```bash
sudo journalctl -u burntbeats-api -f | grep -i webhook
```

### Method 2: Stripe Dashboard Test

1. Go to **Developers** → **Webhooks**
2. Click on your endpoint
3. Click **Send test webhook**
4. Select event: `payment_intent.succeeded`
5. Click **Send test webhook**
6. Check server logs

### Method 3: Real Payment Test

1. Create payment intent with small amount ($0.50)
2. Use test card: `4242 4242 4242 4242`
3. Complete payment
4. Webhook should fire automatically
5. Monitor logs for processing

## Test Script

**Location:** `/tmp/test_payment_flow.py`

**Run:**
```bash
cd /home/ubuntu/app/backend
python3 /tmp/test_payment_flow.py
```

## Current Configuration

### Server .env
```bash
STRIPE_SECRET_KEY=sk_live_51RbydHP38C54URjE...
STRIPE_PUBLISHABLE_KEY=pk_live_51RbydHP38C54URjE...
STRIPE_WEBHOOK_SECRET=whsec_nCaUM9ArPRjwqAa1lieItdDevmBasGTI
REQUIRE_PAYMENT_FOR_GENERATION=true
```

### Webhook Endpoint
- **URL:** `https://burntbeats.com/api/webhooks/stripe`
- **Method:** POST
- **Signature Verification:** ✅ Implemented
- **Events:** `payment_intent.succeeded`, `payment_intent.payment_failed`

## Verification Checklist

### Webhook Configuration
- [ ] Webhook endpoint created in Stripe Dashboard
- [ ] Endpoint URL: `https://burntbeats.com/api/webhooks/stripe`
- [ ] Events selected: `payment_intent.succeeded`, `payment_intent.payment_failed`
- [ ] Webhook secret matches `.env` file
- [ ] Endpoint status: Active

### Payment Flow
- [x] Price calculation endpoint working
- [ ] Payment intent creation tested
- [ ] Payment verification tested
- [ ] Generation with payment tested

### Webhook Handling
- [x] Webhook endpoint accessible
- [ ] Test events received
- [ ] Signature verification working
- [ ] Events processed correctly

## Summary

✅ **All code implemented and configured:**
- Duration-based pricing system
- Payment endpoints
- Webhook handler
- Generation integration
- Stripe keys configured

⚠️ **Remaining:**
- Configure webhook endpoint in Stripe Dashboard
- Test complete payment flow end-to-end

---

**Status:** ✅ **READY FOR STRIPE DASHBOARD CONFIGURATION**  
**Next:** Follow steps above to configure webhook endpoint in Stripe Dashboard
