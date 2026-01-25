# Webhook Configuration and Testing Guide

**Date:** January 23, 2026

## Webhook Endpoint Configuration

### Stripe Dashboard Setup

1. **Access Stripe Dashboard**
   - Go to https://dashboard.stripe.com
   - Log in to your account
   - Ensure you're in **Live mode** (for production)

2. **Create Webhook Endpoint**
   - Navigate to **Developers** → **Webhooks**
   - Click **Add endpoint**
   - Enter endpoint URL: `https://burntbeats.com/api/webhooks/stripe`
   - Click **Add endpoint**

3. **Select Events**
   Select the following events:
   - ✅ `payment_intent.succeeded`
   - ✅ `payment_intent.payment_failed`
   - ✅ `payment_intent.canceled`
   - ✅ `payment_intent.payment_method_attached` (optional)
   - ✅ `charge.succeeded` (optional)
   - ✅ `charge.failed` (optional)

4. **Get Webhook Signing Secret**
   - Click on your webhook endpoint
   - Click **Reveal** next to **Signing secret**
   - Copy the secret (starts with `whsec_`)
   - Verify it matches: `whsec_nCaUM9ArPRjwqAa1lieItdDevmBasGTI`

### Current Webhook Secret

**Configured in .env:**
```
STRIPE_WEBHOOK_SECRET=whsec_nCaUM9ArPRjwqAa1lieItdDevmBasGTI
```

**Verify in Stripe Dashboard:**
- The webhook signing secret in Stripe Dashboard should match this value
- If different, update `.env` file with the correct secret

## Testing Webhook Endpoint

### Method 1: Using Stripe CLI (Recommended for Testing)

**Step 1: Start Webhook Listener**
```bash
stripe listen --forward-to http://127.0.0.1:8001/api/webhooks/stripe
```

This will:
- Display webhook signing secret (verify it matches `.env`)
- Forward webhook events to your endpoint
- Show all events in real-time

**Step 2: Trigger Test Events**

In another terminal:
```bash
# Trigger payment_intent.succeeded
stripe trigger payment_intent.succeeded

# Trigger payment_intent.payment_failed
stripe trigger payment_intent.payment_failed

# Trigger payment_intent.canceled
stripe trigger payment_intent.canceled
```

**Step 3: Monitor Logs**

```bash
# Watch service logs
sudo journalctl -u burntbeats-api -f | grep -i webhook

# Or watch all logs
sudo journalctl -u burntbeats-api -f
```

### Method 2: Test with Real Payment Intent

**Step 1: Create Test Payment Intent**
```bash
stripe payment_intents create \
  --amount=200 \
  --currency=usd \
  --metadata[duration_seconds]=120 \
  --metadata[user_id]=test_user \
  --confirm
```

**Step 2: Monitor Webhook Events**

The webhook endpoint should receive `payment_intent.succeeded` event automatically.

### Method 3: Manual Webhook Test

**Test endpoint accessibility:**
```bash
curl -X POST http://127.0.0.1:8001/api/webhooks/stripe \
  -H "Content-Type: application/json" \
  -d '{"type": "test"}'
```

**Note:** This will fail signature verification, but confirms endpoint is accessible.

## Payment Flow Testing

### Complete Test Flow

1. **Calculate Price**
   ```bash
   curl http://127.0.0.1:8001/api/v1/payments/calculate-price?duration=120
   ```

2. **Create Payment Intent** (via Stripe CLI for testing)
   ```bash
   stripe payment_intents create \
     --amount=200 \
     --currency=usd \
     --metadata[duration_seconds]=120 \
     --metadata[user_id]=test_user \
     --confirm
   ```

3. **Verify Payment Intent**
   ```bash
   curl http://127.0.0.1:8001/api/v1/payments/verify-payment/pi_xxx \
     -H "Authorization: Bearer TOKEN"
   ```

4. **Generate Song** (with payment_intent_id)
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

## Webhook Endpoint Verification

### Check Endpoint is Registered

In Stripe Dashboard:
- Go to **Developers** → **Webhooks**
- Verify endpoint URL: `https://burntbeats.com/api/webhooks/stripe`
- Check endpoint status: Should be **Active**
- Verify events are selected

### Test Webhook Delivery

1. **Trigger test event in Stripe Dashboard**
   - Click on your webhook endpoint
   - Click **Send test webhook**
   - Select event: `payment_intent.succeeded`
   - Click **Send test webhook**

2. **Check server logs**
   ```bash
   sudo journalctl -u burntbeats-api -n 100 | grep -i webhook
   ```

3. **Verify webhook received**
   - Check logs for webhook processing
   - Verify no signature errors
   - Confirm event was handled

## Troubleshooting

### Webhook Not Receiving Events

1. **Check endpoint URL**
   - Verify URL is correct: `https://burntbeats.com/api/webhooks/stripe`
   - Ensure endpoint is accessible from internet
   - Check Nginx configuration

2. **Verify webhook secret**
   - Check `.env` file has correct `STRIPE_WEBHOOK_SECRET`
   - Verify it matches Stripe Dashboard signing secret
   - Restart service after updating

3. **Check endpoint status**
   - In Stripe Dashboard, verify endpoint is **Active**
   - Check for delivery failures
   - Review webhook event logs in Stripe Dashboard

### Signature Verification Errors

- Verify `STRIPE_WEBHOOK_SECRET` matches Stripe Dashboard
- Ensure webhook handler validates signatures
- Check webhook payload format

### Events Not Processing

- Verify events are selected in Stripe Dashboard
- Check webhook handler code for event types
- Review service logs for errors

## Production Checklist

- [ ] Webhook endpoint created in Stripe Dashboard
- [ ] Endpoint URL: `https://burntbeats.com/api/webhooks/stripe`
- [ ] Events selected: `payment_intent.succeeded`, `payment_intent.payment_failed`
- [ ] Webhook secret matches `.env` file
- [ ] Endpoint status: Active
- [ ] Test webhook delivery successful
- [ ] Service logs show webhook processing
- [ ] Payment flow tested end-to-end

---

**Status:** ⚠️ **REQUIRES STRIPE DASHBOARD CONFIGURATION**  
**Next:** Configure webhook endpoint in Stripe Dashboard and test delivery
