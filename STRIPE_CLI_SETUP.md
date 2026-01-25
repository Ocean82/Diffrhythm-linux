# Stripe CLI Installation and Setup

**Date:** January 23, 2026  
**Status:** ✅ **STRIPE CLI INSTALLED**  
**Version:** 1.34.0  
**Location:** `/usr/local/bin/stripe`

## Installation Complete

The Stripe CLI has been successfully installed on the server using direct download method.

### Installation Method

```bash
# Download latest release
cd /tmp
curl -L -o stripe_cli.tar.gz https://github.com/stripe/stripe-cli/releases/latest/download/stripe_1.34.0_linux_x86_64.tar.gz

# Extract
tar -xzf stripe_cli.tar.gz

# Install to system
sudo mv /tmp/stripe /usr/local/bin/stripe
sudo chmod +x /usr/local/bin/stripe

# Verify
stripe --version

# Cleanup
rm -f /tmp/stripe_cli.tar.gz
```

**Note:** Check the latest version at https://github.com/stripe/stripe-cli/releases and update the version number in the URL if needed.

## Verify Installation

```bash
stripe --version
```

## Common Stripe CLI Commands

### 1. Login to Stripe

```bash
stripe login
```

This will open a browser to authenticate with your Stripe account.

### 2. Test Payment Intent Creation

```bash
# Create a test payment intent
stripe payment_intents create \
  --amount=200 \
  --currency=usd \
  --metadata[duration_seconds]=120 \
  --metadata[user_id]=test_user
```

### 3. Forward Webhooks Locally

```bash
# Forward webhooks to local endpoint
stripe listen --forward-to http://127.0.0.1:8001/api/webhooks/stripe
```

This will:
- Show webhook signing secret (add to `.env` as `STRIPE_WEBHOOK_SECRET`)
- Forward webhook events to your local endpoint
- Display all webhook events in real-time

### 4. Trigger Test Events

```bash
# Trigger payment_intent.succeeded event
stripe trigger payment_intent.succeeded

# Trigger payment_intent.payment_failed event
stripe trigger payment_intent.payment_failed
```

### 5. List Payment Intents

```bash
# List recent payment intents
stripe payment_intents list --limit=10
```

### 6. Retrieve Payment Intent

```bash
# Get details of a payment intent
stripe payment_intents retrieve pi_xxx
```

## Testing Payment Flow with Stripe CLI

### Step 1: Login
```bash
stripe login
```

### Step 2: Create Test Payment Intent
```bash
stripe payment_intents create \
  --amount=200 \
  --currency=usd \
  --metadata[duration_seconds]=120 \
  --metadata[user_id]=test_user \
  --confirm
```

### Step 3: Forward Webhooks (in separate terminal)
```bash
stripe listen --forward-to http://127.0.0.1:8001/api/webhooks/stripe
```

### Step 4: Test Webhook Events
```bash
# Trigger test events
stripe trigger payment_intent.succeeded
```

## Webhook Testing

### Get Webhook Signing Secret

When you run `stripe listen`, it will display a webhook signing secret:

```
> Ready! Your webhook signing secret is whsec_xxx
```

Add this to your `.env` file:
```bash
STRIPE_WEBHOOK_SECRET=whsec_xxx
```

### Test Webhook Endpoint

```bash
# Forward webhooks to your endpoint
stripe listen --forward-to http://127.0.0.1:8001/api/webhooks/stripe

# In another terminal, trigger test events
stripe trigger payment_intent.succeeded
```

## Useful Commands

### Check API Status
```bash
stripe status
```

### View Recent Events
```bash
stripe events list --limit=10
```

### Test API Connection
```bash
stripe balance retrieve
```

## Integration with Backend

### Test Payment Intent Creation via API

```bash
# Use Stripe CLI to create payment intent (for testing)
stripe payment_intents create \
  --amount=200 \
  --currency=usd \
  --metadata[duration_seconds]=120 \
  --metadata[user_id]=test_user \
  --confirm
```

### Verify Payment Intent

```bash
# Get payment intent details
stripe payment_intents retrieve pi_xxx
```

## Troubleshooting

### CLI Not Found
If `stripe` command is not found, ensure it's in PATH:
```bash
which stripe
# Should show: /usr/local/bin/stripe or /usr/bin/stripe
```

### Authentication Issues
If login fails:
```bash
stripe logout
stripe login
```

### Webhook Forwarding Issues
If webhooks aren't forwarding:
1. Check endpoint URL is correct
2. Ensure endpoint is accessible
3. Check firewall rules
4. Verify webhook secret matches

---

**Status:** ✅ **STRIPE CLI INSTALLED**  
**Version:** Check with `stripe --version`  
**Next:** Login with `stripe login` and test webhook forwarding
