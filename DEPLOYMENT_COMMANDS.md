# Deployment Commands

**Date:** January 24, 2026  
**Purpose:** Step-by-step commands to deploy and test implementation

## Prerequisites

- SSH access to server (keys configured)
- Server path: `/home/ubuntu/app`
- Service name: `burntbeats-api`

## Step 1: Deploy Code to Server

### From Local Machine (Windows PowerShell)

```powershell
# Navigate to project directory
cd d:\EMBERS-BANK\DiffRhythm-LINUX

# Deploy backend code
scp -r backend/ ubuntu@burntbeats.com:/home/ubuntu/app/

# Deploy test scripts
scp test_server_implementation.py ubuntu@burntbeats.com:/home/ubuntu/app/
scp test_payment_flow.py ubuntu@burntbeats.com:/home/ubuntu/app/
```

**Alternative:** If using a different server hostname/IP:
```powershell
scp -r backend/ ubuntu@YOUR_SERVER_IP:/home/ubuntu/app/
scp test_server_implementation.py ubuntu@YOUR_SERVER_IP:/home/ubuntu/app/
scp test_payment_flow.py ubuntu@YOUR_SERVER_IP:/home/ubuntu/app/
```

## Step 2: SSH to Server and Restart Service

```bash
# SSH to server
ssh ubuntu@burntbeats.com

# Navigate to app directory
cd /home/ubuntu/app

# Verify files are deployed
ls -la backend/api.py backend/payment_verification.py backend/config.py

# Restart service
sudo systemctl restart burntbeats-api

# Check service status
sudo systemctl status burntbeats-api
```

**Expected output:** Service should show as "active (running)"

## Step 3: Run Implementation Tests

```bash
# Still on server
cd /home/ubuntu/app

# Run tests
python3 test_server_implementation.py
```

**Expected results:**
- ✅ Health endpoint test
- ✅ Generate endpoint test (with/without payment)
- ✅ Webhook endpoint test
- ✅ Route alias test
- ✅ Quality defaults test

## Step 4: Verify Service Health

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

## Step 5: Test Generate Endpoint

### Without Payment (if REQUIRE_PAYMENT_FOR_GENERATION=false)

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
- If payment not required: Returns `{"job_id": "...", "status": "queued", ...}`
- If payment required: Returns `402 Payment Required`

### Test Route Alias

```bash
curl -X POST http://127.0.0.1:8001/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "lyrics": "[00:00.00]Test\n[00:05.00]Route test",
    "style_prompt": "test",
    "audio_length": 95
  }'
```

Should work the same as `/api/v1/generate`

## Step 6: Test Webhook Endpoint

```bash
# Test webhook endpoint (should return 400 for invalid signature)
curl -X POST http://127.0.0.1:8001/api/webhooks/stripe \
  -H "Content-Type: application/json" \
  -H "stripe-signature: invalid" \
  -d '{"type": "test"}'
```

**Expected:** Returns `400 Bad Request` (signature verification working)

## Step 7: Check Logs

```bash
# Check recent logs
sudo journalctl -u burntbeats-api -n 50

# Check for payment verification logs
sudo journalctl -u burntbeats-api -n 100 | grep -i payment

# Check for webhook logs
sudo journalctl -u burntbeats-api -n 100 | grep -i webhook

# Check for quality preset usage
sudo journalctl -u burntbeats-api -n 100 | grep -i "preset\|quality"
```

**Look for:**
- "Using quality preset 'high': 32 steps, CFG 4.0"
- "Payment verified: ..."
- "Received Stripe webhook: ..."

## Step 8: Configure Stripe Webhook (Manual)

**Action Required:** Configure in Stripe Dashboard

1. Go to https://dashboard.stripe.com
2. Navigate to **Developers** → **Webhooks**
3. Click **Add endpoint**
4. Enter URL: `https://burntbeats.com/api/webhooks/stripe`
5. Select events:
   - ✅ `payment_intent.succeeded`
   - ✅ `payment_intent.payment_failed`
   - ✅ `payment_intent.canceled` (optional)
6. Copy the webhook signing secret
7. Verify it matches `/home/ubuntu/app/backend/.env`:
   ```bash
   grep STRIPE_WEBHOOK_SECRET /home/ubuntu/app/backend/.env
   ```

**Detailed Instructions:** See `STRIPE_WEBHOOK_CONFIGURATION_FINAL.md`

## Step 9: Test Webhook Delivery

### Using Stripe CLI (if installed on server)

```bash
# Terminal 1: Start webhook listener
stripe listen --forward-to http://127.0.0.1:8001/api/webhooks/stripe

# Terminal 2: Trigger test events
stripe trigger payment_intent.succeeded
stripe trigger payment_intent.payment_failed
```

### Using Stripe Dashboard

1. Go to webhook endpoint in Stripe Dashboard
2. Click **Send test webhook**
3. Select event: `payment_intent.succeeded`
4. Click **Send test webhook**
5. Check server logs:
   ```bash
   sudo journalctl -u burntbeats-api -n 50 | grep webhook
   ```

## Step 10: Test Complete Payment Flow

```bash
# Run payment flow test
python3 test_payment_flow.py
```

This will test:
- Price calculation
- Payment intent creation
- Payment verification
- Webhook endpoint

## Troubleshooting

### Service Won't Start

```bash
# Check for errors
sudo journalctl -u burntbeats-api -n 100

# Check Python syntax
python3 -m py_compile /home/ubuntu/app/backend/api.py

# Check imports
cd /home/ubuntu/app
python3 -c "from backend.api import app; print('Imports OK')"
```

### Payment Verification Fails

```bash
# Check Stripe keys
grep STRIPE /home/ubuntu/app/backend/.env

# Check payment verification module
python3 -c "from backend.payment_verification import verify_payment_intent; print('Module OK')"
```

### Webhook Not Working

```bash
# Verify webhook secret
grep STRIPE_WEBHOOK_SECRET /home/ubuntu/app/backend/.env

# Test endpoint
curl -X POST http://127.0.0.1:8001/api/webhooks/stripe \
  -H "stripe-signature: test" \
  -d '{"type": "test"}'
```

## Quick Verification Checklist

- [ ] Files deployed to server
- [ ] Service restarted successfully
- [ ] Health endpoint returns `models_loaded: true`
- [ ] Generate endpoint accessible
- [ ] Payment verification working (if enabled)
- [ ] Webhook endpoint accessible
- [ ] Route alias working
- [ ] Quality defaults confirmed in logs
- [ ] Stripe webhook configured in Dashboard
- [ ] Webhook delivery tested

## Next Steps After Deployment

1. ✅ Code deployed
2. ✅ Service restarted
3. ✅ Tests passed
4. ⚠️ Configure Stripe webhook in Dashboard
5. ⚠️ Test end-to-end payment → generation flow
6. ⚠️ Monitor production usage

---

**See Also:**
- `SERVER_TESTING_GUIDE.md` - Detailed testing guide
- `STRIPE_WEBHOOK_CONFIGURATION_FINAL.md` - Webhook setup
- `FINAL_IMPLEMENTATION_STATUS.md` - Implementation status
