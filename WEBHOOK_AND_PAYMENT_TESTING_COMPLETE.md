# Webhook Configuration and Payment Testing - Complete Guide

**Date:** January 23, 2026

## Summary

This guide provides complete instructions for configuring webhook endpoints in Stripe Dashboard and testing the payment flow.

## Webhook Endpoint Configuration

### Stripe Dashboard Setup

**Endpoint URL:** `https://burntbeats.com/api/webhooks/stripe`

**Required Events:**
- ✅ `payment_intent.succeeded`
- ✅ `payment_intent.payment_failed`
- ✅ `payment_intent.canceled`

**Webhook Secret:** `whsec_nCaUM9ArPRjwqAa1lieItdDevmBasGTI`

### Steps to Configure

1. **Access Stripe Dashboard**
   - Go to https://dashboard.stripe.com
   - Log in and switch to **Live mode**

2. **Create Webhook Endpoint**
   - Navigate to **Developers** → **Webhooks**
   - Click **Add endpoint**
   - URL: `https://burntbeats.com/api/webhooks/stripe`
   - Click **Add endpoint**

3. **Select Events**
   - Check: `payment_intent.succeeded`
   - Check: `payment_intent.payment_failed`
   - Check: `payment_intent.canceled`
   - Click **Add events**

4. **Get Signing Secret**
   - Click on your endpoint
   - Click **Reveal** next to **Signing secret**
   - Verify it matches: `whsec_nCaUM9ArPRjwqAa1lieItdDevmBasGTI`
   - If different, update `.env` file

## Payment Flow Testing

### Test Script Available

**Location:** `/tmp/test_payment_flow.py`

**Usage:**
```bash
cd /home/ubuntu/app/backend
python3 /tmp/test_payment_flow.py
```

### Manual Testing Steps

#### 1. Calculate Price
```bash
curl http://127.0.0.1:8001/api/v1/payments/calculate-price?duration=120
```

**Expected:** Returns pricing options including base price, commercial license, bulk packs.

#### 2. Create Payment Intent (Test with Stripe CLI)

```bash
stripe payment_intents create \
  --amount=200 \
  --currency=usd \
  --metadata[duration_seconds]=120 \
  --metadata[user_id]=test_user \
  --confirm
```

**Expected:** Creates payment intent and confirms payment (for testing).

#### 3. Verify Payment Intent

```bash
curl http://127.0.0.1:8001/api/v1/payments/verify-payment/pi_xxx \
  -H "Authorization: Bearer TOKEN"
```

**Expected:** Returns payment status and `ready_for_generation: true` if succeeded.

#### 4. Generate Song (with Payment)

```bash
curl -X POST http://127.0.0.1:8001/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer TOKEN" \
  -d '{
    "text_prompt": "A happy pop song",
    "duration": 120,
    "payment_intent_id": "pi_xxx"
  }'
```

**Expected:** Generation starts after payment verification.

## Webhook Testing

### Method 1: Stripe CLI Listener (Recommended)

**Start webhook listener:**
```bash
stripe listen --forward-to http://127.0.0.1:8001/api/webhooks/stripe
```

**In another terminal, trigger test events:**
```bash
# Test successful payment
stripe trigger payment_intent.succeeded

# Test failed payment
stripe trigger payment_intent.payment_failed

# Test canceled payment
stripe trigger payment_intent.canceled
```

**Monitor logs:**
```bash
sudo journalctl -u burntbeats-api -f | grep -i webhook
```

### Method 2: Stripe Dashboard Test

1. Go to **Developers** → **Webhooks**
2. Click on your endpoint
3. Click **Send test webhook**
4. Select event: `payment_intent.succeeded`
5. Click **Send test webhook**
6. Check server logs for receipt

### Method 3: Real Payment Test

1. Create a real payment intent (small amount)
2. Complete payment
3. Webhook should fire automatically
4. Monitor logs for webhook processing

## Verification Checklist

### Webhook Configuration
- [ ] Webhook endpoint created in Stripe Dashboard
- [ ] Endpoint URL: `https://burntbeats.com/api/webhooks/stripe`
- [ ] Events selected: `payment_intent.succeeded`, `payment_intent.payment_failed`
- [ ] Webhook secret matches `.env` file
- [ ] Endpoint status: Active

### Payment Flow
- [ ] Price calculation endpoint working
- [ ] Payment intent creation working
- [ ] Payment verification working
- [ ] Generation with payment working

### Webhook Handling
- [ ] Webhook endpoint accessible
- [ ] Test events received
- [ ] Signature verification working
- [ ] Events processed correctly

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
- **Events:** `payment_intent.succeeded`, `payment_intent.payment_failed`

## Testing Results

### Price Calculation
- ✅ Endpoint working
- ✅ Returns correct pricing
- ✅ All options included

### Payment Intent Creation
- ⚠️ Requires authentication token
- ⚠️ Requires Stripe CLI login for CLI testing
- ✅ Endpoint configured correctly

### Webhook Endpoint
- ✅ Endpoint accessible
- ⚠️ Requires Stripe Dashboard configuration
- ⚠️ Requires signature verification

## Next Steps

1. **Configure Webhook in Stripe Dashboard**
   - Create endpoint with URL: `https://burntbeats.com/api/webhooks/stripe`
   - Select required events
   - Verify webhook secret

2. **Test Webhook Delivery**
   - Use Stripe CLI listener
   - Trigger test events
   - Verify events received

3. **Test Complete Payment Flow**
   - Calculate price
   - Create payment intent
   - Verify payment
   - Generate song

---

**Status:** ⚠️ **REQUIRES STRIPE DASHBOARD CONFIGURATION**  
**Next:** Configure webhook endpoint in Stripe Dashboard
