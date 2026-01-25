# Deployment Ready - SSH Fixed

**Date:** January 24, 2026  
**Status:** ✅ SSH connection working

## Next Steps - Deployment & Testing

### Step 1: Deploy Code to Server

```powershell
cd d:\EMBERS-BANK\DiffRhythm-LINUX

# Deploy backend code
scp -i "C:\Users\sammy\.ssh\server_saver_key" -r backend/ ubuntu@52.0.207.242:/home/ubuntu/app/

# Deploy test scripts
scp -i "C:\Users\sammy\.ssh\server_saver_key" test_server_implementation.py ubuntu@52.0.207.242:/home/ubuntu/app/
scp -i "C:\Users\sammy\.ssh\server_saver_key" test_payment_flow.py ubuntu@52.0.207.242:/home/ubuntu/app/
```

### Step 2: Restart Service

```powershell
ssh -i "C:\Users\sammy\.ssh\server_saver_key" ubuntu@52.0.207.242 "sudo systemctl restart burntbeats-api"
```

### Step 3: Verify Service Status

```powershell
ssh -i "C:\Users\sammy\.ssh\server_saver_key" ubuntu@52.0.207.242 "sudo systemctl status burntbeats-api"
```

### Step 4: Run Server Tests

```powershell
ssh -i "C:\Users\sammy\.ssh\server_saver_key" ubuntu@52.0.207.242 "cd /home/ubuntu/app && python3 test_server_implementation.py"
```

### Step 5: Test Payment Flow

```powershell
ssh -i "C:\Users\sammy\.ssh\server_saver_key" ubuntu@52.0.207.242 "cd /home/ubuntu/app && python3 test_payment_flow.py"
```

### Step 6: Configure Stripe Webhook

1. Go to Stripe Dashboard → Developers → Webhooks
2. Add endpoint: `https://burntbeats.com/api/webhooks/stripe`
3. Select events:
   - `payment_intent.succeeded`
   - `payment_intent.payment_failed`
   - `payment_intent.canceled`
4. Copy webhook signing secret to server `.env` file

### Step 7: Test Webhook Delivery

Use Stripe Dashboard "Send test webhook" or Stripe CLI:
```bash
stripe trigger payment_intent.succeeded
```

### Step 8: Verify Quality

Generate a test song and verify:
- Clear vocals
- Professional production quality
- Proper mastering applied

## Quick Deployment Script

```powershell
# Save as deploy_now.ps1
$KEY = "C:\Users\sammy\.ssh\server_saver_key"
$SERVER = "ubuntu@52.0.207.242"
$REMOTE_DIR = "/home/ubuntu/app"

Write-Host "Deploying backend code..."
scp -i $KEY -r backend/ ${SERVER}:${REMOTE_DIR}/

Write-Host "Deploying test scripts..."
scp -i $KEY test_server_implementation.py ${SERVER}:${REMOTE_DIR}/
scp -i $KEY test_payment_flow.py ${SERVER}:${REMOTE_DIR}/

Write-Host "Restarting service..."
ssh -i $KEY $SERVER "sudo systemctl restart burntbeats-api"

Write-Host "Checking service status..."
ssh -i $KEY $SERVER "sudo systemctl status burntbeats-api --no-pager"

Write-Host "Running server tests..."
ssh -i $KEY $SERVER "cd ${REMOTE_DIR} && python3 test_server_implementation.py"

Write-Host "Deployment complete!"
```

## Files Ready for Deployment

- ✅ `backend/api.py` - Payment verification integrated
- ✅ `backend/config.py` - Stripe configuration
- ✅ `backend/payment_verification.py` - Payment verification logic
- ✅ `test_server_implementation.py` - Server-side tests
- ✅ `test_payment_flow.py` - Payment flow tests

## Remaining Tasks

1. ✅ **deploy-code** - Ready to execute
2. ✅ **restart-service** - Ready to execute
3. ✅ **run-server-tests** - Ready to execute
4. ⚠️ **test-payment-flow** - After deployment
5. ⚠️ **test-webhook** - After Stripe Dashboard configuration
6. ⚠️ **verify-quality** - After deployment

---

**Status:** Ready for deployment  
**Next:** Execute deployment commands above
