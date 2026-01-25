# Server Testing Guide

**Date:** January 24, 2026  
**Purpose:** Test implementation on server after deployment

## Pre-Deployment Verification

### 1. Verify Code Implementation

```bash
# On local machine or server
cd /path/to/DiffRhythm-LINUX
bash verify_deployment_implementation.sh
```

This will verify:
- All files exist
- Python syntax is correct
- Required imports are present
- Configuration variables are set

### 2. Deploy to Server

```bash
# From local machine
scp -r backend/ user@server:/home/ubuntu/app/
scp test_server_implementation.py user@server:/home/ubuntu/app/
scp test_payment_flow.py user@server:/home/ubuntu/app/
```

## Server-Side Testing

### Step 1: Verify Service Status

```bash
# SSH to server
ssh user@server

# Check service status
sudo systemctl status burntbeats-api

# Check if service is running on port 8001
sudo netstat -tlnp | grep 8001
# or
sudo ss -tlnp | grep 8001
```

### Step 2: Test Health Endpoint

```bash
# Test health endpoint
curl http://127.0.0.1:8001/api/v1/health

# Expected response:
# {
#   "status": "healthy" or "degraded",
#   "models_loaded": true/false,
#   "device": "cpu",
#   "queue_length": 0,
#   "active_jobs": 0,
#   "version": "1.0.0"
# }
```

### Step 3: Run Implementation Tests

```bash
cd /home/ubuntu/app
python3 test_server_implementation.py
```

This will test:
- Health endpoint
- Generate endpoint (with/without payment)
- Webhook endpoint accessibility
- Route alias
- Quality defaults

### Step 4: Test Payment Flow

```bash
# Test payment calculation
curl "http://127.0.0.1:8001/api/v1/payments/calculate-price?duration=120"

# Test webhook endpoint (should return 400 for invalid signature)
curl -X POST http://127.0.0.1:8001/api/webhooks/stripe \
  -H "Content-Type: application/json" \
  -H "stripe-signature: invalid" \
  -d '{"type": "test"}'
```

### Step 5: Test Generate Endpoint

#### Without Payment (if REQUIRE_PAYMENT_FOR_GENERATION=false)

```bash
curl -X POST http://127.0.0.1:8001/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "lyrics": "[00:00.00]Test song\n[00:05.00]This is a test",
    "style_prompt": "pop, upbeat, energetic",
    "audio_length": 95,
    "batch_size": 1
  }'
```

**Expected:**
- If payment not required: Returns job_id and status "queued"
- If payment required: Returns 402 Payment Required

#### With Payment Intent ID

```bash
# First, create a test payment intent using Stripe CLI
stripe payment_intents create \
  --amount=200 \
  --currency=usd \
  --confirm

# Use the payment_intent_id from the response
curl -X POST http://127.0.0.1:8001/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "lyrics": "[00:00.00]Test song\n[00:05.00]This is a test",
    "style_prompt": "pop, upbeat",
    "audio_length": 95,
    "payment_intent_id": "pi_xxxxx"
  }'
```

### Step 6: Test Webhook Delivery

#### Using Stripe CLI

```bash
# Terminal 1: Start webhook listener
stripe listen --forward-to http://127.0.0.1:8001/api/webhooks/stripe

# Terminal 2: Trigger test events
stripe trigger payment_intent.succeeded
stripe trigger payment_intent.payment_failed
stripe trigger payment_intent.canceled
```

#### Using Stripe Dashboard

1. Go to https://dashboard.stripe.com
2. Navigate to **Developers** → **Webhooks**
3. Find your endpoint: `https://burntbeats.com/api/webhooks/stripe`
4. Click **Send test webhook**
5. Select event: `payment_intent.succeeded`
6. Click **Send test webhook**
7. Check server logs: `sudo journalctl -u burntbeats-api -n 50 | grep webhook`

### Step 7: Verify Quality Settings

```bash
# Check logs for quality preset usage
sudo journalctl -u burntbeats-api -n 100 | grep -i "preset\|quality\|master"

# Should see:
# - "Using quality preset 'high': 32 steps, CFG 4.0"
# - "Applying mastering with preset: balanced"
```

### Step 8: Generate Test Song

```bash
# Submit generation job
JOB_ID=$(curl -X POST http://127.0.0.1:8001/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "lyrics": "[00:00.00]Welcome to DiffRhythm\n[00:05.00]This is a test song\n[00:10.00]Testing quality settings",
    "style_prompt": "pop, upbeat, energetic, professional production",
    "audio_length": 95
  }' | jq -r '.job_id')

echo "Job ID: $JOB_ID"

# Check job status
curl http://127.0.0.1:8001/api/v1/status/$JOB_ID

# Wait for completion (check every 30 seconds)
while true; do
  STATUS=$(curl -s http://127.0.0.1:8001/api/v1/status/$JOB_ID | jq -r '.status')
  echo "Status: $STATUS"
  if [ "$STATUS" = "completed" ]; then
    echo "✅ Generation complete!"
    break
  elif [ "$STATUS" = "failed" ]; then
    echo "❌ Generation failed"
    break
  fi
  sleep 30
done

# Download generated audio
curl http://127.0.0.1:8001/api/v1/download/$JOB_ID -o test_song.wav
```

## Verification Checklist

- [ ] Service is running on port 8001
- [ ] Health endpoint returns `models_loaded: true`
- [ ] Generate endpoint accepts requests
- [ ] Payment verification works (if enabled)
- [ ] Webhook endpoint is accessible
- [ ] Webhook signature verification works
- [ ] Route alias `/api/generate` works
- [ ] Quality preset defaults to "high"
- [ ] Auto-mastering is enabled
- [ ] Generated song has clear vocals
- [ ] Generated song has professional production quality

## Troubleshooting

### Service Not Running

```bash
# Check service status
sudo systemctl status burntbeats-api

# Check logs
sudo journalctl -u burntbeats-api -n 100

# Restart service
sudo systemctl restart burntbeats-api
```

### Payment Verification Fails

```bash
# Check Stripe keys are set
grep STRIPE /home/ubuntu/app/backend/.env

# Check payment verification logs
sudo journalctl -u burntbeats-api -n 100 | grep -i payment
```

### Webhook Not Receiving Events

```bash
# Verify webhook secret matches
grep STRIPE_WEBHOOK_SECRET /home/ubuntu/app/backend/.env

# Check webhook endpoint logs
sudo journalctl -u burntbeats-api -n 100 | grep -i webhook

# Test endpoint accessibility
curl -X POST http://127.0.0.1:8001/api/webhooks/stripe \
  -H "stripe-signature: test" \
  -d '{"type": "test"}'
```

### Quality Issues

```bash
# Check preset is being used
sudo journalctl -u burntbeats-api -n 100 | grep "quality preset"

# Verify mastering is applied
sudo journalctl -u burntbeats-api -n 100 | grep -i master
```

## Next Steps

1. ✅ Code implementation complete
2. ⚠️ Deploy to server
3. ⚠️ Run tests
4. ⚠️ Configure Stripe webhook in Dashboard
5. ⚠️ Verify end-to-end flow
6. ⚠️ Monitor production usage

---

**Status:** Ready for server testing  
**See:** `IMPLEMENTATION_COMPLETE_SUMMARY.md` for implementation details
