# Stripe Keys Setup Instructions

**Date:** January 23, 2026

## Overview

To enable payment processing, you need to add your Stripe API keys to the `.env` file on the server.

## Steps to Get Stripe Keys

### 1. Access Stripe Dashboard

1. Go to https://dashboard.stripe.com
2. Log in to your Stripe account
3. Make sure you're in the correct mode (Test or Live)

### 2. Get API Keys

#### Secret Key
1. Navigate to **Developers** → **API keys**
2. Find **Secret key** (starts with `sk_test_` for test mode or `sk_live_` for live mode)
3. Click **Reveal test key** or **Reveal live key**
4. Copy the key

#### Publishable Key
1. On the same page, find **Publishable key** (starts with `pk_test_` or `pk_live_`)
2. Copy the key

#### Webhook Secret
1. Navigate to **Developers** → **Webhooks**
2. Click on your webhook endpoint (or create one if it doesn't exist)
3. Click **Reveal** next to **Signing secret**
4. Copy the secret (starts with `whsec_`)

## Update .env File

**File:** `/home/ubuntu/app/backend/.env`

Replace the placeholder values with your actual Stripe keys:

```bash
# Stripe Configuration
STRIPE_SECRET_KEY=sk_live_YOUR_ACTUAL_SECRET_KEY_HERE
STRIPE_PUBLISHABLE_KEY=pk_live_YOUR_ACTUAL_PUBLISHABLE_KEY_HERE
STRIPE_WEBHOOK_SECRET=whsec_YOUR_ACTUAL_WEBHOOK_SECRET_HERE
REQUIRE_PAYMENT_FOR_GENERATION=true
```

### For Test Mode (Development)

```bash
STRIPE_SECRET_KEY=sk_test_YOUR_TEST_SECRET_KEY
STRIPE_PUBLISHABLE_KEY=pk_test_YOUR_TEST_PUBLISHABLE_KEY
STRIPE_WEBHOOK_SECRET=whsec_YOUR_TEST_WEBHOOK_SECRET
REQUIRE_PAYMENT_FOR_GENERATION=false  # Set to false for testing
```

### For Live Mode (Production)

```bash
STRIPE_SECRET_KEY=sk_live_YOUR_LIVE_SECRET_KEY
STRIPE_PUBLISHABLE_KEY=pk_live_YOUR_LIVE_PUBLISHABLE_KEY
STRIPE_WEBHOOK_SECRET=whsec_YOUR_LIVE_WEBHOOK_SECRET
REQUIRE_PAYMENT_FOR_GENERATION=true  # Set to true in production
```

## Webhook Configuration

### Create Webhook Endpoint

1. In Stripe Dashboard, go to **Developers** → **Webhooks**
2. Click **Add endpoint**
3. Enter endpoint URL: `https://burntbeats.com/api/webhooks/stripe`
4. Select events to listen to:
   - `payment_intent.succeeded`
   - `payment_intent.payment_failed`
   - `payment_intent.canceled`
5. Click **Add endpoint**
6. Copy the **Signing secret** and add it to `.env` as `STRIPE_WEBHOOK_SECRET`

## Restart Service

After updating `.env`, restart the API service:

```bash
sudo systemctl restart burntbeats-api
sudo systemctl status burntbeats-api
```

## Verify Configuration

Test that Stripe is configured:

```bash
curl http://127.0.0.1:8001/api/v1/payments/calculate-price?duration=120
```

If Stripe is properly configured, you should see pricing information without warnings.

## Security Notes

- ⚠️ **Never commit `.env` file to version control**
- ⚠️ **Keep secret keys secure**
- ⚠️ **Use test keys for development**
- ⚠️ **Use live keys only in production**
- ⚠️ **Rotate keys if compromised**

## Troubleshooting

### "Stripe not configured" warning
- Check that keys are in `.env` file
- Verify keys are correct (no extra spaces)
- Restart service after updating `.env`

### Payment intent creation fails
- Verify `STRIPE_SECRET_KEY` is correct
- Check Stripe Dashboard for API errors
- Ensure account is active

### Webhook not receiving events
- Verify `STRIPE_WEBHOOK_SECRET` matches webhook signing secret
- Check webhook endpoint URL is correct
- Ensure webhook endpoint is active in Stripe Dashboard

---

**Status:** ⚠️ **REQUIRES MANUAL CONFIGURATION**  
**Next:** Add your Stripe keys to `.env` file
