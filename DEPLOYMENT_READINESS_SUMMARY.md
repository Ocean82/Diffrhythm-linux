# Deployment Readiness Summary

**Date:** January 24, 2026  
**Status:** ✅ **ALL CODE CHANGES COMPLETE - READY FOR DEPLOYMENT**

## Implementation Status

### ✅ Completed Code/Config Changes

1. **Stripe Environment Variables**
   - ✅ Added to `config/ec2-config.env` (template with placeholders)
   - ✅ `STRIPE_SECRET_KEY`, `STRIPE_PUBLISHABLE_KEY`, `STRIPE_WEBHOOK_SECRET`, `REQUIRE_PAYMENT_FOR_GENERATION=true`
   - ✅ Backend reads from env via `backend/config.py`

2. **Docker Compose Configuration**
   - ✅ `docker-compose.prod.yml` updated with:
     - `env_file: .env` to load all env vars
     - Explicit `STRIPE_SECRET_KEY`, `STRIPE_PUBLISHABLE_KEY`, `STRIPE_WEBHOOK_SECRET`, `REQUIRE_PAYMENT_FOR_GENERATION` in `environment:` section
   - ✅ Container will receive Stripe config at runtime

3. **Deploy Script Optimizations**
   - ✅ `scripts/deploy-to-server.sh`:
     - Excludes `pretrained/` from rsync (avoids re-uploading large model files)
     - Creates `output`, `temp`, `pretrained` directories on server
     - Fixed SSH heredoc issues (uses single `ssh "..."` commands)
   - ✅ Ready to deploy when server is reachable

4. **E2E Testing Script**
   - ✅ `scripts/e2e-payment-generation-check.sh` created
   - ✅ Tests health endpoint and payment-required behavior
   - ✅ Provides instructions for full payment → generation flow

5. **Backend Implementation**
   - ✅ Payment verification: `backend/payment_verification.py`
   - ✅ Webhook handler: `/api/webhooks/stripe` in `backend/api.py`
   - ✅ Payment check in `/api/v1/generate` endpoint
   - ✅ Returns 402 when payment required but not provided/verified

## Deployment Checklist

### Step 1: Deploy Code to Server

When server `52.0.207.242` is reachable:

```bash
bash scripts/deploy-to-server.sh
```

This will:
- Rsync code (excluding `pretrained/`)
- Build Docker image
- Start services with `docker-compose.prod.yml`

### Step 2: Configure Stripe Keys on Server

SSH to server and edit `/opt/diffrhythm/.env`:

```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242
cd /opt/diffrhythm
nano .env  # or vi .env
```

Add/update:
```bash
STRIPE_SECRET_KEY=sk_live_YOUR_ACTUAL_KEY
STRIPE_PUBLISHABLE_KEY=pk_live_YOUR_ACTUAL_KEY
STRIPE_WEBHOOK_SECRET=whsec_YOUR_WEBHOOK_SECRET  # Get from Stripe Dashboard
REQUIRE_PAYMENT_FOR_GENERATION=true
```

Then restart:
```bash
cd /opt/diffrhythm
sudo docker-compose -f docker-compose.prod.yml restart diffrhythm-api
```

### Step 3: Configure Stripe Webhook

Follow **STRIPE_WEBHOOK_CONFIGURATION_FINAL.md**:

1. Go to https://dashboard.stripe.com → **Developers** → **Webhooks** (Live mode)
2. Click **Add endpoint**
3. URL: `https://burntbeats.com/api/webhooks/stripe`
4. Select events:
   - ✅ `payment_intent.succeeded`
   - ✅ `payment_intent.payment_failed`
   - ✅ `payment_intent.canceled` (optional)
5. Copy **Signing secret** (`whsec_...`)
6. Update `/opt/diffrhythm/.env` with the signing secret
7. Restart API: `sudo docker-compose -f docker-compose.prod.yml restart diffrhythm-api`

### Step 4: Verify Deployment

```bash
# Health check
curl http://52.0.207.242:8000/api/v1/health

# Or run E2E script
bash scripts/e2e-payment-generation-check.sh
```

### Step 5: Test Payment → Generation Flow

1. **Create Payment Intent:**
   ```bash
   stripe payment_intents create --amount=200 --currency=usd --confirm
   ```
   Note the `pi_...` ID

2. **Verify Webhook Delivery:**
   - Check Stripe Dashboard → Webhooks → [your endpoint] → Events
   - Should see `payment_intent.succeeded` delivered

3. **Generate Song:**
   ```bash
   curl -X POST "http://52.0.207.242:8000/api/v1/generate" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer <API_KEY>" \
     -d '{
       "lyrics": "[00:00.00]Line one\n[00:05.00]Line two",
       "style_prompt": "upbeat pop, clear vocals",
       "audio_length": 95,
       "payment_intent_id": "pi_YOUR_PAYMENT_INTENT_ID"
     }'
   ```

4. **Poll Status:**
   ```bash
   curl "http://52.0.207.242:8000/api/v1/status/{job_id}"
   ```

## Files Modified

- ✅ `config/ec2-config.env` - Added Stripe vars template
- ✅ `docker-compose.prod.yml` - Added `env_file` and explicit Stripe env vars
- ✅ `scripts/deploy-to-server.sh` - Optimized (exclude pretrained, fixed SSH)
- ✅ `scripts/e2e-payment-generation-check.sh` - New E2E test script

## Files Already Implemented (from previous work)

- ✅ `backend/config.py` - Stripe config vars
- ✅ `backend/payment_verification.py` - Payment verification logic
- ✅ `backend/api.py` - Webhook handler and payment check in generate endpoint

## Next Steps

1. **Wait for server to be reachable** (currently `52.0.207.242` is not accessible)
2. **Run deployment:** `bash scripts/deploy-to-server.sh`
3. **Configure Stripe keys** on server `.env`
4. **Configure webhook** in Stripe Dashboard
5. **Test end-to-end** payment → generation flow

## Troubleshooting

- **SSH connection fails:** Check server status, firewall, SSH key permissions
- **Docker build fails:** Check disk space, Docker daemon running
- **Webhook 4xx/5xx:** Verify `STRIPE_WEBHOOK_SECRET` matches Dashboard, check Nginx/proxy config
- **402 on generate:** Ensure `payment_intent_id` is valid and payment is succeeded in Stripe

---

**All code changes are complete. System is ready for deployment when server is accessible.**
