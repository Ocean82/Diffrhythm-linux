# Deployment Status

**Date:** January 24, 2026  
**Status:** ⚠️ **READY FOR MANUAL DEPLOYMENT**

## Deployment Commands

Since automated deployment requires SSH authentication, please run these commands manually:

### Step 1: Deploy Code

**From Windows PowerShell or WSL:**

```powershell
# Navigate to project
cd d:\EMBERS-BANK\DiffRhythm-LINUX

# Deploy backend code (use your server IP/hostname)
scp -r backend/ ubuntu@52.0.207.242:/home/ubuntu/app/
# OR if using hostname:
# scp -r backend/ ubuntu@burntbeats.com:/home/ubuntu/app/

# Deploy test scripts
scp test_server_implementation.py ubuntu@52.0.207.242:/home/ubuntu/app/
scp test_payment_flow.py ubuntu@52.0.207.242:/home/ubuntu/app/
```

**If using SSH key:**
```powershell
scp -i ~/.ssh/your_key -r backend/ ubuntu@52.0.207.242:/home/ubuntu/app/
```

### Step 2: Restart Service

```bash
# SSH to server
ssh ubuntu@52.0.207.242
# OR
ssh ubuntu@burntbeats.com

# Restart service
sudo systemctl restart burntbeats-api

# Check status
sudo systemctl status burntbeats-api
```

### Step 3: Run Tests

```bash
# On server
cd /home/ubuntu/app
python3 test_server_implementation.py
```

### Step 4: Verify Health

```bash
# Test health endpoint
curl http://127.0.0.1:8001/api/v1/health
```

## Files Ready for Deployment

✅ `backend/api.py` - Updated with payment verification, quality defaults, webhook
✅ `backend/payment_verification.py` - Payment verification module
✅ `backend/config.py` - Stripe configuration variables
✅ `test_server_implementation.py` - Server testing script
✅ `test_payment_flow.py` - Payment flow testing script

## Remaining Tasks

### Code Implementation ✅
- [x] Payment verification integrated
- [x] Quality defaults set
- [x] Webhook handler implemented
- [x] Route alias added

### Server Deployment ⚠️
- [ ] Deploy code to server (manual step required)
- [ ] Restart service
- [ ] Verify service is running

### Testing ⚠️
- [ ] Run `test_server_implementation.py` on server
- [ ] Test payment flow
- [ ] Test webhook delivery
- [ ] Verify quality settings

### Stripe Configuration ⚠️
- [ ] Configure webhook in Stripe Dashboard
- [ ] Test webhook delivery
- [ ] Verify webhook secret matches

## Quick Verification Commands

After deployment, run these on the server:

```bash
# Check files deployed
ls -la /home/ubuntu/app/backend/api.py
ls -la /home/ubuntu/app/backend/payment_verification.py

# Check service
sudo systemctl status burntbeats-api

# Test health
curl http://127.0.0.1:8001/api/v1/health

# Check logs
sudo journalctl -u burntbeats-api -n 50
```

## Next Actions

1. **Deploy code manually** using scp commands above
2. **Restart service** on server
3. **Run tests** using provided test scripts
4. **Configure Stripe webhook** in Dashboard
5. **Verify end-to-end** payment → generation flow

---

**See:** `DEPLOYMENT_COMMANDS.md` for detailed step-by-step instructions
