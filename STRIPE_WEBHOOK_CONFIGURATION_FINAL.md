# Stripe Webhook Configuration - Final Instructions

**Date:** January 23, 2026

## Webhook Endpoint Configuration in Stripe Dashboard

### Step-by-Step Instructions

1. **Access Stripe Dashboard**
   - Go to https://dashboard.stripe.com
   - Log in to your Stripe account
   - Ensure you're in **Live mode** (top right toggle)

2. **Navigate to Webhooks**
   - Click **Developers** in the left sidebar
   - Click **Webhooks**

3. **Add Endpoint**
   - Click **Add endpoint** button
   - In the **Endpoint URL** field, enter:
     ```
     https://burntbeats.com/api/webhooks/stripe
     ```
   - Click **Add endpoint**

4. **Select Events**
   In the "Select events to listen to" section, check:
   - ✅ `payment_intent.succeeded`
   - ✅ `payment_intent.payment_failed`
   - ✅ `payment_intent.canceled` (optional but recommended)
   
   Click **Add events**

5. **Get Webhook Signing Secret**
   - After creating the endpoint, click on it
   - Find **Signing secret** section
   - Click **Reveal** button
   - Copy the secret (starts with `whsec_`)
   - **Verify it matches:** `whsec_nCaUM9ArPRjwqAa1lieItdDevmBasGTI`
   - If different, update `/home/ubuntu/app/backend/.env` file

6. **Verify Endpoint Status**
   - Check that endpoint status is **Active**
   - Verify the endpoint URL is correct
   - Confirm events are selected

## Testing Webhook Delivery

### Method 1: Stripe Dashboard Test (Easiest)

1. In Stripe Dashboard, go to your webhook endpoint
2. Click **Send test webhook** button
3. Select event: `payment_intent.succeeded`
4. Click **Send test webhook**
5. Check server logs:
   ```bash
   sudo journalctl -u burntbeats-api -n 50 | grep -i webhook
   ```

### Method 2: Stripe CLI Listener (For Local Testing)

**On Server:**
```bash
# Start webhook listener
stripe listen --forward-to http://127.0.0.1:8001/api/webhooks/stripe
```

**Output will show:**
```
> Ready! Your webhook signing secret is whsec_xxx
```

**In Another Terminal:**
```bash
# Trigger test events
stripe trigger payment_intent.succeeded
stripe trigger payment_intent.payment_failed
```

**Monitor Logs:**
```bash
sudo journalctl -u burntbeats-api -f | grep -i webhook
```

### Method 3: Real Payment Test

1. Create a test payment intent with small amount ($0.50)
2. Use test card: `4242 4242 4242 4242`
3. Complete payment
4. Webhook should fire automatically
5. Check logs for webhook processing

## Payment Flow Testing

### Complete Test Flow

**Step 1: Calculate Price** ✅
```bash
curl http://127.0.0.1:8001/api/v1/payments/calculate-price?duration=120
```
**Result:** Returns pricing options

**Step 2: Create Payment Intent**

**Using Stripe CLI (for testing):**
```bash
stripe login  # First time only
stripe payment_intents create --amount=200 --currency=usd --confirm
```

**Using API (requires auth token):**
```bash
curl -X POST http://127.0.0.1:8001/api/v1/payments/create-intent \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer TOKEN" \
  -d '{
    "duration_seconds": 120,
    "amount_cents": 200,
    "currency": "usd"
  }'
```

**Step 3: Verify Payment**
```bash
curl http://127.0.0.1:8001/api/v1/payments/verify-payment/pi_xxx \
  -H "Authorization: Bearer TOKEN"
```

**Step 4: Generate Song**
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
- **Status:** ✅ Accessible
- **Signature Verification:** ✅ Working
- **Events Handled:** `payment_intent.succeeded`, `payment_intent.payment_failed`

## Verification

### Endpoint Status
- ✅ Webhook endpoint accessible
- ✅ Signature verification working (correctly rejects invalid signatures)
- ✅ Event handling code ready

### Payment Endpoints
- ✅ Price calculation: Working
- ✅ Payment intent creation: Ready (requires auth)
- ✅ Payment verification: Ready (requires auth)

## Troubleshooting

### Webhook Not Receiving Events

1. **Check endpoint URL**
   - Must be: `https://burntbeats.com/api/webhooks/stripe`
   - Must be accessible from internet
   - Check Nginx configuration

2. **Verify webhook secret**
   - Check `.env` file has correct `STRIPE_WEBHOOK_SECRET`
   - Must match Stripe Dashboard signing secret
   - Restart service after updating

3. **Check endpoint status**
   - In Stripe Dashboard, verify endpoint is **Active**
   - Check for delivery failures
   - Review webhook event logs

### Signature Verification Errors

- Verify `STRIPE_WEBHOOK_SECRET` matches Stripe Dashboard
- Ensure webhook handler validates signatures
- Check webhook payload format

## Summary

✅ **Implementation:** Complete  
✅ **Configuration:** Stripe keys configured  
✅ **Endpoints:** Working  
⚠️ **Remaining:** Configure webhook endpoint in Stripe Dashboard

---

**Status:** ✅ **READY FOR STRIPE DASHBOARD CONFIGURATION**  
**Action:** Follow steps above to configure webhook endpoint in Stripe Dashboard
