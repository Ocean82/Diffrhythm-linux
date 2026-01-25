# Stripe Keys Setup - Final Instructions

**Date:** January 23, 2026

## Overview

I cannot access your Stripe Dashboard to retrieve your keys. You need to add them manually. I've created a helper script to make this easier.

## Quick Setup (3 Steps)

### Step 1: Get Your Stripe Keys

1. Go to https://dashboard.stripe.com
2. Log in to your account
3. Navigate to **Developers** → **API keys**
4. Copy:
   - **Secret key** (starts with `sk_test_` or `sk_live_`)
   - **Publishable key** (starts with `pk_test_` or `pk_live_`)
5. Navigate to **Developers** → **Webhooks**
6. Create or select webhook endpoint
7. Copy **Signing secret** (starts with `whsec_`)

### Step 2: Use Helper Script

**SSH to server:**
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242
```

**Run the helper script:**
```bash
/tmp/update_stripe_keys.sh
```

The script will:
- Prompt you for each key
- Validate the key formats
- Update the `.env` file
- Create a backup

### Step 3: Restart Service

```bash
sudo systemctl restart burntbeats-api
sudo systemctl status burntbeats-api
```

## Alternative: Manual Edit

If you prefer to edit manually:

```bash
# SSH to server
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242

# Edit .env file
cd /home/ubuntu/app/backend
nano .env

# Find and replace these lines:
STRIPE_SECRET_KEY=sk_test_...          # Replace with your actual key
STRIPE_PUBLISHABLE_KEY=pk_test_...     # Replace with your actual key
STRIPE_WEBHOOK_SECRET=whsec_...        # Replace with your actual secret
REQUIRE_PAYMENT_FOR_GENERATION=true    # Set to true for production

# Save and exit (Ctrl+X, Y, Enter)

# Restart service
sudo systemctl restart burntbeats-api
```

## Verification

After adding keys, verify configuration:

```bash
# Test price calculation endpoint
curl http://127.0.0.1:8001/api/v1/payments/calculate-price?duration=120

# Check service logs
sudo journalctl -u burntbeats-api -n 50 | grep -i stripe
```

Should show no "Stripe not configured" warnings.

## Current .env Status

**File:** `/home/ubuntu/app/backend/.env`

**Current placeholders:**
```bash
STRIPE_SECRET_KEY=sk_test_...
STRIPE_PUBLISHABLE_KEY=pk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
REQUIRE_PAYMENT_FOR_GENERATION=false
```

**Replace with your actual keys from Stripe Dashboard.**

## Helper Script Location

**Script:** `/tmp/update_stripe_keys.sh`

**Features:**
- ✅ Validates key formats
- ✅ Creates backup of .env
- ✅ Updates keys safely
- ✅ Shows updated configuration

## Security Reminders

- ⚠️ Never share your secret keys
- ⚠️ Never commit `.env` to git
- ⚠️ Use test keys for development
- ⚠️ Use live keys only in production

---

**Status:** ⚠️ **REQUIRES YOUR STRIPE KEYS**  
**Action:** Run `/tmp/update_stripe_keys.sh` on the server after getting keys from Stripe Dashboard
