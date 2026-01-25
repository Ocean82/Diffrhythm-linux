# How to Add Stripe Keys to .env File

**Date:** January 23, 2026

## Quick Instructions

I cannot access your Stripe Dashboard to retrieve your keys. You need to add them manually. Here's how:

## Step-by-Step Guide

### 1. Get Your Stripe Keys

**Option A: From Stripe Dashboard (Recommended)**
1. Go to https://dashboard.stripe.com
2. Log in to your account
3. Navigate to **Developers** → **API keys**
4. Copy your **Secret key** (starts with `sk_test_` or `sk_live_`)
5. Copy your **Publishable key** (starts with `pk_test_` or `pk_live_`)
6. Navigate to **Developers** → **Webhooks**
7. Create or select your webhook endpoint
8. Copy the **Signing secret** (starts with `whsec_`)

**Option B: Using Stripe CLI**
```bash
# Login first
stripe login

# Then get keys from dashboard or use CLI commands
stripe config --list
```

### 2. Edit .env File on Server

**SSH to server:**
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242
```

**Edit .env file:**
```bash
cd /home/ubuntu/app/backend
nano .env
```

**Find the Stripe section and replace:**
```bash
# Replace these lines:
STRIPE_SECRET_KEY=sk_test_...          # Replace with your actual secret key
STRIPE_PUBLISHABLE_KEY=pk_test_...     # Replace with your actual publishable key
STRIPE_WEBHOOK_SECRET=whsec_...        # Replace with your actual webhook secret
REQUIRE_PAYMENT_FOR_GENERATION=true    # Set to true for production
```

**Save and exit:**
- Press `Ctrl+X`
- Press `Y` to confirm
- Press `Enter` to save

### 3. Restart Service

```bash
sudo systemctl restart burntbeats-api
sudo systemctl status burntbeats-api
```

### 4. Verify Configuration

```bash
# Test price calculation endpoint
curl http://127.0.0.1:8001/api/v1/payments/calculate-price?duration=120

# Check service logs for Stripe warnings
sudo journalctl -u burntbeats-api -n 50 | grep -i stripe
```

## Example .env Configuration

### For Test Mode (Development)
```bash
STRIPE_SECRET_KEY=sk_test_...  # Replace with your actual test secret key
STRIPE_PUBLISHABLE_KEY=pk_test_...  # Replace with your actual test publishable key
STRIPE_WEBHOOK_SECRET=whsec_...  # Replace with your actual test webhook secret
REQUIRE_PAYMENT_FOR_GENERATION=false
```

### For Live Mode (Production)
```bash
STRIPE_SECRET_KEY=sk_live_...  # Replace with your actual live secret key
STRIPE_PUBLISHABLE_KEY=pk_live_...  # Replace with your actual live publishable key
STRIPE_WEBHOOK_SECRET=whsec_...  # Replace with your actual live webhook secret
REQUIRE_PAYMENT_FOR_GENERATION=true
```

## Important Notes

- ⚠️ **Never share your secret keys publicly**
- ⚠️ **Never commit `.env` file to git**
- ⚠️ **Use test keys for development**
- ⚠️ **Use live keys only in production**
- ⚠️ **Keep keys secure and rotate if compromised**

## Troubleshooting

### Keys not working?
- Verify keys are correct (no extra spaces)
- Check you're using the right mode (test vs live)
- Ensure account is active in Stripe Dashboard

### Service not starting?
- Check `.env` file syntax (no quotes around values)
- Verify file permissions
- Check service logs: `sudo journalctl -u burntbeats-api -n 100`

---

**Status:** ⚠️ **REQUIRES YOUR STRIPE KEYS**  
**Action:** Add your Stripe keys from Stripe Dashboard to `.env` file
