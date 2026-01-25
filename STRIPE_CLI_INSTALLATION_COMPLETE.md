# Stripe CLI Installation - Complete

**Date:** January 23, 2026  
**Status:** ✅ **INSTALLATION SUCCESSFUL**

## Installation Summary

The Stripe CLI has been successfully installed on the server.

### Installation Details

- **Version:** 1.34.0
- **Location:** `/usr/local/bin/stripe`
- **Installation Method:** Direct download from GitHub releases
- **Status:** ✅ Installed and verified

## Verification

```bash
stripe --version
# Output: stripe version 1.34.0
```

## Next Steps

### 1. Login to Stripe

```bash
stripe login
```

This will:
- Open a browser for authentication
- Save your API key locally
- Enable CLI access to your Stripe account

### 2. Test Webhook Forwarding

```bash
# Forward webhooks to your local endpoint
stripe listen --forward-to http://127.0.0.1:8001/api/webhooks/stripe
```

This will:
- Display webhook signing secret (add to `.env`)
- Forward webhook events to your endpoint
- Show all events in real-time

### 3. Test Payment Intent Creation

```bash
# Create a test payment intent
stripe payment_intents create \
  --amount=200 \
  --currency=usd \
  --metadata[duration_seconds]=120 \
  --metadata[user_id]=test_user \
  --confirm
```

### 4. Trigger Test Events

```bash
# Trigger payment_intent.succeeded event
stripe trigger payment_intent.succeeded

# Trigger payment_intent.payment_failed event
stripe trigger payment_intent.payment_failed
```

## Common Commands

### List Payment Intents
```bash
stripe payment_intents list --limit=10
```

### Retrieve Payment Intent
```bash
stripe payment_intents retrieve pi_xxx
```

### View Recent Events
```bash
stripe events list --limit=10
```

### Check API Status
```bash
stripe status
```

## Integration with Backend Testing

The Stripe CLI can be used to:

1. **Test Payment Intent Creation**
   - Create payment intents directly
   - Verify metadata is set correctly
   - Confirm payments for testing

2. **Test Webhook Handling**
   - Forward webhooks to local endpoint
   - Trigger test events
   - Verify webhook signature validation

3. **Debug Payment Issues**
   - List payment intents
   - Retrieve payment details
   - View event logs

## Webhook Testing Setup

### Step 1: Start Webhook Listener

```bash
stripe listen --forward-to http://127.0.0.1:8001/api/webhooks/stripe
```

**Output will show:**
```
> Ready! Your webhook signing secret is whsec_xxx
```

### Step 2: Add Webhook Secret to .env

```bash
# Add to /home/ubuntu/app/backend/.env
STRIPE_WEBHOOK_SECRET=whsec_xxx
```

### Step 3: Restart Service

```bash
sudo systemctl restart burntbeats-api
```

### Step 4: Trigger Test Events

```bash
# In another terminal
stripe trigger payment_intent.succeeded
```

## Troubleshooting

### CLI Not Found
If `stripe` command is not found:
```bash
which stripe
# Should show: /usr/local/bin/stripe
```

### Authentication Issues
If login fails:
```bash
stripe logout
stripe login
```

### Webhook Forwarding Issues
- Check endpoint URL is correct
- Ensure endpoint is accessible
- Verify webhook secret matches

---

**Status:** ✅ **STRIPE CLI INSTALLED AND READY**  
**Next:** Run `stripe login` to authenticate
