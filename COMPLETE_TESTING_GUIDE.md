# Complete Testing Guide - Remaining Tasks

**Date:** January 24, 2026

## Overview

This guide covers testing for the three remaining tasks:
1. **test-payment-flow** - Complete payment flow testing
2. **test-webhook** - Webhook delivery testing
3. **verify-quality** - Quality verification testing

## Prerequisites

### Server Setup
- ✅ Code deployed to server
- ✅ Service running (`burntbeats-api`)
- ✅ Stripe keys configured in `.env`
- ✅ SSH access working

### Tools Required
- Python 3.8+
- `requests` library: `pip install requests`
- Stripe CLI (for webhook testing): https://stripe.com/docs/stripe-cli

## Task 1: Test Complete Payment Flow

### Script: `test_complete_payment_flow.py`

**Purpose:** Tests the complete payment flow from price calculation to song generation.

### Steps:

1. **Deploy script to server:**
   ```bash
   scp -i ~/.ssh/server_saver_key test_complete_payment_flow.py ubuntu@52.0.207.242:/home/ubuntu/app/
   ```

2. **SSH to server:**
   ```bash
   ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242
   ```

3. **Install Stripe CLI (if not installed):**
   ```bash
   # On Ubuntu
   sudo apt-get update
   sudo apt-get install stripe
   # Or download from: https://github.com/stripe/stripe-cli/releases
   ```

4. **Login to Stripe CLI:**
   ```bash
   stripe login
   ```

5. **Run test:**
   ```bash
   cd /home/ubuntu/app
   python3 test_complete_payment_flow.py
   ```

### Expected Results:

- ✅ Price calculation works
- ✅ Payment intent created successfully
- ✅ Payment verification works
- ✅ Song generation starts with payment

### Troubleshooting:

**Payment intent creation fails:**
- Ensure `stripe login` has been run
- Check Stripe CLI is installed: `stripe --version`
- Verify you're using the correct Stripe account

**Payment verification fails:**
- Payment intent may need time to process (wait 2-3 seconds)
- Check payment intent status in Stripe Dashboard
- Verify payment amount matches expected amount

**Generation fails with 402:**
- Payment intent may not be confirmed
- Check payment verification endpoint is working
- Verify `REQUIRE_PAYMENT_FOR_GENERATION` setting

## Task 2: Test Webhook Delivery

### Script: `test_webhook_delivery.py`

**Purpose:** Tests Stripe webhook delivery and processing.

### Steps:

1. **Deploy script to server:**
   ```bash
   scp -i ~/.ssh/server_saver_key test_webhook_delivery.py ubuntu@52.0.207.242:/home/ubuntu/app/
   ```

2. **Configure webhook in Stripe Dashboard:**
   - Go to https://dashboard.stripe.com
   - Navigate to Developers → Webhooks
   - Add endpoint: `https://burntbeats.com/api/webhooks/stripe`
   - Select events:
     - `payment_intent.succeeded`
     - `payment_intent.payment_failed`
     - `payment_intent.canceled`
   - Copy webhook signing secret

3. **Update server .env:**
   ```bash
   ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242
   cd /home/ubuntu/app/backend
   nano .env
   # Update STRIPE_WEBHOOK_SECRET with value from Stripe Dashboard
   sudo systemctl restart burntbeats-api
   ```

4. **Run test:**
   ```bash
   cd /home/ubuntu/app
   python3 test_webhook_delivery.py
   ```

5. **Test webhook delivery (Option A - Stripe CLI):**
   ```bash
   # Terminal 1: Start webhook listener
   stripe listen --forward-to http://127.0.0.1:8001/api/webhooks/stripe
   
   # Terminal 2: Trigger test events
   stripe trigger payment_intent.succeeded
   stripe trigger payment_intent.payment_failed
   ```

6. **Test webhook delivery (Option B - Stripe Dashboard):**
   - Go to webhook endpoint in Stripe Dashboard
   - Click "Send test webhook"
   - Select event: `payment_intent.succeeded`
   - Click "Send test webhook"
   - Check server logs: `sudo journalctl -u burntbeats-api -n 50 | grep -i webhook`

### Expected Results:

- ✅ Webhook endpoint accessible
- ✅ Signature verification working
- ✅ Events processed correctly
- ✅ Webhook configured in Stripe Dashboard

### Troubleshooting:

**Webhook endpoint not accessible:**
- Check Nginx configuration routes `/api/webhooks/stripe`
- Verify service is running: `sudo systemctl status burntbeats-api`
- Test endpoint: `curl -X POST http://127.0.0.1:8001/api/webhooks/stripe`

**Signature verification fails:**
- Verify `STRIPE_WEBHOOK_SECRET` matches Stripe Dashboard
- Check webhook handler validates signatures
- Restart service after updating `.env`

**Events not received:**
- Check endpoint URL in Stripe Dashboard
- Verify endpoint is accessible from internet
- Check webhook event logs in Stripe Dashboard
- Monitor server logs: `sudo journalctl -u burntbeats-api -f`

## Task 3: Verify Quality

### Script: `test_quality_verification.py`

**Purpose:** Generates a test song and verifies Suno-style quality.

### Steps:

1. **Deploy script to server:**
   ```bash
   scp -i ~/.ssh/server_saver_key test_quality_verification.py ubuntu@52.0.207.242:/home/ubuntu/app/
   ```

2. **Run test:**
   ```bash
   ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242
   cd /home/ubuntu/app
   python3 test_quality_verification.py
   ```

3. **If payment required:**
   - Run `test_complete_payment_flow.py` first to get payment_intent_id
   - Pass payment_intent_id to quality test

4. **Monitor job status:**
   - Test script will monitor job automatically
   - Or check manually: `GET /api/v1/jobs/{job_id}`

5. **Download and verify audio:**
   ```bash
   # Get output file from job status
   curl http://127.0.0.1:8001/api/v1/jobs/{job_id}
   
   # Download audio file
   curl http://127.0.0.1:8001/{output_file} -o test_song.wav
   
   # Listen and verify quality
   ```

### Quality Criteria:

- ✅ **Clear vocals** - Natural, not robotic or muffled
- ✅ **Good rhythm** - Proper timing and beat
- ✅ **Professional production** - Studio-quality sound
- ✅ **Proper mastering** - Balanced levels, no clipping
- ✅ **Suno-style quality** - Comparable to Suno.ai output

### Expected Settings:

- Preset: `high` (32 steps, CFG 4.0)
- Auto-master: `True`
- Master preset: `balanced`

### Troubleshooting:

**Generation fails:**
- Check service logs: `sudo journalctl -u burntbeats-api -n 100`
- Verify models are loaded: `GET /api/v1/health`
- Check payment if required

**Quality not meeting standards:**
- Verify preset is set to `high`
- Check auto_master is enabled
- Review style_prompt for quality keywords
- Try different master_preset (balanced/aggressive)

**Job takes too long:**
- Normal for CPU: 25-50 minutes
- Check job progress: `GET /api/v1/jobs/{job_id}`
- Monitor system resources: `htop`

## Quick Test Commands

### All Tests at Once

```bash
# Deploy all test scripts
scp -i ~/.ssh/server_saver_key test_*.py ubuntu@52.0.207.242:/home/ubuntu/app/

# SSH to server
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242

# Run tests in sequence
cd /home/ubuntu/app
python3 test_complete_payment_flow.py
python3 test_webhook_delivery.py
python3 test_quality_verification.py
```

### Check Service Status

```bash
# Service status
sudo systemctl status burntbeats-api

# Service logs
sudo journalctl -u burntbeats-api -n 100

# Health check
curl http://127.0.0.1:8001/api/v1/health
```

## Success Criteria

### Payment Flow ✅
- [ ] Price calculation works
- [ ] Payment intent created
- [ ] Payment verified
- [ ] Generation starts with payment

### Webhook ✅
- [ ] Endpoint accessible
- [ ] Signature verification works
- [ ] Events processed
- [ ] Dashboard configured

### Quality ✅
- [ ] Defaults correct (high preset, auto_master)
- [ ] Song generated successfully
- [ ] Audio quality meets standards
- [ ] Suno-style quality achieved

## Next Steps After Testing

1. **Document results** - Record test outcomes
2. **Fix issues** - Address any failures
3. **Production deployment** - Deploy to production
4. **Monitor** - Set up monitoring and alerts

---

**Status:** Ready for testing  
**Location:** All test scripts in project root  
**Server Path:** `/home/ubuntu/app/`
