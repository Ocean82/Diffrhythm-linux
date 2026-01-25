# Adding Stripe Keys to .env File

**Date:** January 23, 2026

## Overview

To enable payment processing, you need to add your real Stripe API keys to the `.env` file. This guide provides multiple methods to obtain and add your keys.

## Method 1: Using Stripe Dashboard (Recommended)

### Step 1: Get Keys from Stripe Dashboard

1. **Go to Stripe Dashboard**
   - Visit https://dashboard.stripe.com
   - Log in to your Stripe account
   - Switch to **Test mode** for testing or **Live mode** for production

2. **Get Secret Key**
   - Navigate to **Developers** → **API keys**
   - Find **Secret key** (starts with `sk_test_` or `sk_live_`)
   - Click **Reveal test key** or **Reveal live key**
   - Copy the key

3. **Get Publishable Key**
   - On the same page, find **Publishable key** (starts with `pk_test_` or `pk_live_`)
   - Copy the key

4. **Get Webhook Secret**
   - Navigate to **Developers** → **Webhooks**
   - Click on your webhook endpoint (or create one)
   - Click **Reveal** next to **Signing secret**
   - Copy the secret (starts with `whsec_`)

### Step 2: Update .env File

**File:** `/home/ubuntu/app/backend/.env`

```bash
# Edit the file
nano /home/ubuntu/app/backend/.env

# Or use vi
vi /home/ubuntu/app/backend/.env
```

Replace the placeholder values:

```bash
# Stripe Configuration
STRIPE_SECRET_KEY=sk_live_YOUR_ACTUAL_SECRET_KEY_HERE
STRIPE_PUBLISHABLE_KEY=pk_live_YOUR_ACTUAL_PUBLISHABLE_KEY_HERE
STRIPE_WEBHOOK_SECRET=whsec_YOUR_ACTUAL_WEBHOOK_SECRET_HERE
REQUIRE_PAYMENT_FOR_GENERATION=true
```

### Step 3: Restart Service

```bash
sudo systemctl restart burntbeats-api
sudo systemctl status burntbeats-api
```

## Method 2: Using Stripe CLI

### Step 1: Login to Stripe CLI

```bash
stripe login
```

This will open a browser for authentication.

### Step 2: Get API Keys

```bash
# Get secret key (from Stripe Dashboard or CLI config)
stripe config --list

# Or retrieve from Stripe Dashboard after login
```

### Step 3: Create Webhook Endpoint

```bash
# Forward webhooks to get signing secret
stripe listen --forward-to http://127.0.0.1:8001/api/webhooks/stripe
```

The output will show:
```
> Ready! Your webhook signing secret is whsec_xxx
```

Copy this secret.

### Step 4: Update .env

Add the keys to `.env` file as shown in Method 1.

## Method 3: Direct SSH Edit

### Step 1: SSH to Server

```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242
```

### Step 2: Edit .env File

```bash
cd /home/ubuntu/app/backend
nano .env
```

### Step 3: Add/Update Stripe Keys

Find the Stripe section and update:

```bash
STRIPE_SECRET_KEY=sk_live_YOUR_KEY_HERE
STRIPE_PUBLISHABLE_KEY=pk_live_YOUR_KEY_HERE
STRIPE_WEBHOOK_SECRET=whsec_YOUR_SECRET_HERE
REQUIRE_PAYMENT_FOR_GENERATION=true
```

Save and exit (Ctrl+X, then Y, then Enter for nano).

### Step 4: Restart Service

```bash
sudo systemctl restart burntbeats-api
```

## Verification

### Test Stripe Configuration

```bash
# Check if Stripe is configured
curl http://127.0.0.1:8001/api/v1/payments/calculate-price?duration=120

# Should return pricing without warnings
```

### Check Service Logs

```bash
sudo journalctl -u burntbeats-api -n 50 | grep -i stripe
```

Should show no "Stripe not configured" warnings.

## Security Notes

- ⚠️ **Never commit `.env` file to version control**
- ⚠️ **Keep secret keys secure**
- ⚠️ **Use test keys for development**
- ⚠️ **Use live keys only in production**
- ⚠️ **Rotate keys if compromised**

## Troubleshooting

### "Stripe not configured" warning
- Check that keys are in `.env` file
- Verify keys are correct (no extra spaces, quotes, etc.)
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
**Next:** Add your Stripe keys to `.env` file using one of the methods above
