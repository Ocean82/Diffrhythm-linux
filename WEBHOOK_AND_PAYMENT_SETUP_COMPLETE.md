# Webhook Configuration and Payment Testing - Complete Setup

**Date:** January 23, 2026  
**Status:** ✅ **WEBHOOK ENDPOINT READY, REQUIRES STRIPE DASHBOARD CONFIGURATION**

## Webhook Endpoint Status

### Endpoint Configuration

**URL:** `https://burntbeats.com/api/webhooks/stripe`  
**Method:** POST  
**Status:** ✅ Endpoint exists and is accessible  
**Signature Verification:** ✅ Implemented

### Current Implementation

**File:** `/home/ubuntu/app/backend/src/api/stripe_webhooks.py`

**Events Handled:**
- ✅ `payment_intent.succeeded` - Payment completed, grant access
- ✅ `payment_intent.payment_failed` - Payment failed, log error
- ✅ `charge.succeeded` - Charge confirmed

**Features:**
- ✅ Signature verification using `STRIPE_WEBHOOK_SECRET`
- ✅ Event type handling
- ✅ Metadata extraction (duration, user_id, etc.)
- ✅ Price verification from metadata
- ✅ Database integration for purchase records

## Stripe Dashboard Configuration Required

### Step 1: Create Webhook Endpoint

1. **Access Stripe Dashboard**
   - Go to https://dashboard.stripe.com
   - Log in to your account
   - Ensure you're in **Live mode** (for production)

2. **Navigate to Webhooks**
   - Click **Developers** → **Webhooks**
   - Click **Add endpoint**

3. **Configure Endpoint**
   - **Endpoint URL:** `https://burntbeats.com/api/webhooks/stripe`
   - Click **Add endpoint**

4. **Select Events**
   Select these events:
   - ✅ `payment_intent.succeeded`
   - ✅ `payment_intent.payment_failed`
   - ✅ `payment_intent.canceled`
   - ✅ `payment_intent.payment_method_attached` (optional)
   - ✅ `charge.succeeded` (optional)
   - ✅ `charge.failed` (optional)

5. **Get Signing Secret**
   - Click on your webhook endpoint
   - Click **Reveal** next to **Signing secret**
   - Copy the secret (starts with `whsec_`)
   - **Verify it matches:** `whsec_nCaUM9ArPRjwqAa1lieItdDevmBasGTI`
   - If different, update `.env` file with the correct secret

### Step 2: Verify Endpoint Status

- Check endpoint status: Should be **Active**
- Verify endpoint URL is correct
- Check events are selected
- Test webhook delivery from Stripe Dashboard

## Payment Flow Testing

### Test Script Available

**Location:** `/tmp/test_payment_flow.py`

**Run tests:**
```bash
cd /home/ubuntu/app/backend
python3 /tmp/test_payment_flow.py
```

### Manual Testing Steps

#### 1. Calculate Price ✅

```bash
curl http://127.0.0.1:8001/api/v1/payments/calculate-price?duration=120
```

**Result:** ✅ Working - Returns pricing options

#### 2. Create Payment Intent

**Option A: Using Stripe CLI (for testing)**
```bash
# First, login to Stripe CLI
stripe login

# Create test payment intent
stripe payment_intents create \
  --amount=200 \
  --currency=usd \
  --metadata duration_seconds=120 \
  --metadata user_id=test_user \
  --confirm
```

**Option B: Using API Endpoint (requires auth)**
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

#### 3. Verify Payment Intent

```bash
curl http://127.0.0.1:8001/api/v1/payments/verify-payment/pi_xxx \
  -H "Authorization: Bearer TOKEN"
```

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

## Webhook Testing

### Method 1: Stripe CLI Listener (Recommended)

**Step 1: Start Webhook Listener**
```bash
stripe listen --forward-to http://127.0.0.1:8001/api/webhooks/stripe
```

**Output will show:**
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
6. Check server logs for receipt

### Method 3: Real Payment Test

1. Create a real payment intent (small amount: $0.50)
2. Complete payment with test card: `4242 4242 4242 4242`
3. Webhook should fire automatically
4. Monitor logs for webhook processing

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
- **Events Handled:** `payment_intent.succeeded`, `payment_intent.payment_failed`

## Testing Results

### ✅ Working
- Price calculation endpoint
- Webhook endpoint accessible
- Signature verification implemented
- Event handling code ready

### ⚠️ Requires Configuration
- Stripe Dashboard webhook endpoint creation
- Stripe CLI login for local testing
- Authentication tokens for API testing

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

## Next Steps

1. **Configure Webhook in Stripe Dashboard**
   - Create endpoint: `https://burntbeats.com/api/webhooks/stripe`
   - Select required events
   - Verify webhook secret matches

2. **Test Webhook Delivery**
   - Use Stripe CLI: `stripe listen --forward-to http://127.0.0.1:8001/api/webhooks/stripe`
   - Trigger test events: `stripe trigger payment_intent.succeeded`
   - Verify events received in logs

3. **Test Complete Payment Flow**
   - Calculate price
   - Create payment intent (via API or CLI)
   - Verify payment
   - Generate song with payment_intent_id

---

**Status:** ✅ **WEBHOOK ENDPOINT READY**  
**Next:** Configure webhook endpoint in Stripe Dashboard
