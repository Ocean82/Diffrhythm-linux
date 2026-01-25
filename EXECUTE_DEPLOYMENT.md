# Execute Deployment - Ready to Run

**Date:** January 24, 2026  
**Status:** ✅ All Files Ready | Ready for Manual Execution

## Quick Deploy Commands

Since the automated connection timed out, please run these commands **from your local machine** where SSH access works:

### PowerShell (Windows)

```powershell
cd d:\EMBERS-BANK\DiffRhythm-LINUX

# Deploy backend
scp -i ~/.ssh/server_saver_key -r backend/ ubuntu@52.0.207.242:/home/ubuntu/app/

# Deploy test scripts
scp -i ~/.ssh/server_saver_key test_server_implementation.py ubuntu@52.0.207.242:/home/ubuntu/app/
scp -i ~/.ssh/server_saver_key test_payment_flow.py ubuntu@52.0.207.242:/home/ubuntu/app/
```

**OR use the PowerShell script:**
```powershell
.\deploy_with_key.ps1
```

### Bash/WSL

```bash
cd /mnt/d/EMBERS-BANK/DiffRhythm-LINUX

# Deploy backend
scp -i ~/.ssh/server_saver_key -r backend/ ubuntu@52.0.207.242:/home/ubuntu/app/

# Deploy test scripts
scp -i ~/.ssh/server_saver_key test_server_implementation.py ubuntu@52.0.207.242:/home/ubuntu/app/
scp -i ~/.ssh/server_saver_key test_payment_flow.py ubuntu@52.0.207.242:/home/ubuntu/app/
```

**OR use the bash script:**
```bash
chmod +x deploy_with_key.sh
./deploy_with_key.sh
```

## After Deployment - Server Commands

```bash
# SSH to server
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242

# Verify files
ls -la /home/ubuntu/app/backend/api.py
ls -la /home/ubuntu/app/backend/payment_verification.py

# Restart service
sudo systemctl restart burntbeats-api

# Check status
sudo systemctl status burntbeats-api

# Run tests
cd /home/ubuntu/app
python3 test_server_implementation.py

# Test health
curl http://127.0.0.1:8001/api/v1/health
```

## Files Ready for Deployment

✅ **backend/api.py** (29,873 bytes)
- Payment verification integrated
- Quality defaults: "high" preset, auto-mastering enabled
- Webhook handler at `/api/webhooks/stripe`
- Route alias `/api/generate` → `/api/v1/generate`

✅ **backend/payment_verification.py** (3,118 bytes)
- Payment intent verification
- Stripe API integration
- Error handling

✅ **backend/config.py** (3,940 bytes)
- Stripe configuration variables
- Payment requirement flag

✅ **test_server_implementation.py**
- Comprehensive server tests
- Health, generate, webhook, route alias tests

✅ **test_payment_flow.py**
- Payment flow testing
- Webhook testing

## Implementation Summary

### ✅ Completed
- Payment verification before generation
- Default quality preset: "high" (32 steps, CFG 4.0)
- Auto-mastering enabled by default
- Webhook handler implemented
- Route compatibility maintained
- All code passes linting

### ⚠️ Pending (After Deployment)
- Service restart verification
- Run server tests
- Test payment flow
- Configure Stripe webhook
- Verify quality settings

## Quick Verification After Deployment

```bash
# On server
cd /home/ubuntu/app

# Quick check
echo "=== Files ==="
ls -la backend/api.py backend/payment_verification.py

echo "=== Service ==="
sudo systemctl is-active burntbeats-api

echo "=== Health ==="
curl -s http://127.0.0.1:8001/api/v1/health | jq .

echo "=== Tests ==="
python3 test_server_implementation.py
```

---

**Ready to deploy!** Run the scp commands above from your machine where SSH works.
