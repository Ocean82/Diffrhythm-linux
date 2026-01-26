# Remaining TODO Items Summary

**Date:** January 24, 2026  
**Last updated:** After SSH fix and deploy reinvestigation

## ✅ Completed Tasks

1. ✅ **verify-502-fix** – Verified 502 error is fixed
2. ✅ **add-payment-verification** – Added payment verification to generate endpoint
3. ✅ **configure-webhook** – Webhook configuration instructions in `STRIPE_WEBHOOK_CONFIGURATION_FINAL.md`
4. ✅ **set-quality-defaults** – Default preset `high`, auto_master enabled
5. ✅ **update-nginx-routing** – Nginx configuration verified/updated
6. ✅ **Git Security** – Stripe keys redacted, `.gitignore` updated
7. ✅ **fix-ssh-access** – **SSH now working.** Deploy script uses `server_saver_key` (primary) or Burnt-Beats-KEY.pem fallbacks.

## ⚠️ Pending Tasks (Server-Side)

### 1. **deploy-code** (Status: in progress when last run)
- **Command:** `bash scripts/deploy-to-server.sh`
- **Flow:** SSH → ec2-setup → rsync (excl. pretrained) → .env setup → Docker build → `docker-compose.prod.yml` up
- **Target:** `/opt/diffrhythm`, Docker `diffrhythm-api` on port 8000
- **Note:** Deploy was run successfully through rsync; Docker build may still be running or needs to complete. Re-run deploy if build was interrupted.

### 2. **restart-service** (after deploy)
- **Command:**  
  `ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "cd /opt/diffrhythm && sudo docker-compose -f docker-compose.prod.yml restart diffrhythm-api"`
- Use when only restarting (e.g. after .env/Stripe changes) without re-deploying code.

### 3. **configure-stripe-on-server**
- Edit `/opt/diffrhythm/.env` on server: set `STRIPE_SECRET_KEY`, `STRIPE_PUBLISHABLE_KEY`, `STRIPE_WEBHOOK_SECRET`, `REQUIRE_PAYMENT_FOR_GENERATION=true`.
- Restart API after changes (see **restart-service**).

### 4. **configure-stripe-webhook** (manual)
- **Stripe Dashboard** → Developers → Webhooks → Add endpoint
- **URL:** `https://burntbeats.com/api/webhooks/stripe`
- **Events:** `payment_intent.succeeded`, `payment_intent.payment_failed`, `payment_intent.canceled`
- Copy **Signing secret** → update `STRIPE_WEBHOOK_SECRET` in server `.env` → restart API.
- **Ref:** `STRIPE_WEBHOOK_CONFIGURATION_FINAL.md`

### 5. **test-payment-flow**
- **Script:** `bash scripts/e2e-payment-generation-check.sh` (health + payment-required check).
- **Full flow:** Calculate price → create intent → verify payment → generate with `payment_intent_id`. See E2E instructions printed by the script.
- **Prerequisite:** Deploy complete, API healthy, Stripe keys in `.env`.

### 6. **test-webhook**
- Stripe Dashboard → **Send test webhook** (`payment_intent.succeeded`) or Stripe CLI: `stripe trigger payment_intent.succeeded`.
- Check server logs:  
  `ssh ... "cd /opt/diffrhythm && sudo docker-compose -f docker-compose.prod.yml logs -f diffrhythm-api"` and grep for `webhook` / `payment_intent`.
- **Prerequisite:** Webhook configured in Dashboard, `STRIPE_WEBHOOK_SECRET` in `.env`.

### 7. **verify-quality**
- Generate a test song (with valid `payment_intent_id` if `REQUIRE_PAYMENT=true`).
- Confirm: clear vocals, good rhythm, professional production, mastering applied.
- **Prerequisite:** Deploy complete, service running.

## Changes Applied (Reinvestigation)

### SSH
- **Fix:** Deploy script now prefers `C:/Users/sammy/.ssh/server_saver_key` (known working). Fallbacks: Burnt-Beats-KEY.pem in AWS ITEMS, KEYS, BURNING-EMBERS, SERVER-SAVER, `.ssh`.
- **Verification:** `bash scripts/deploy-to-server.sh` → SSH connection successful, ec2-setup and rsync completed.

### Deploy Flow
- **Script:** `scripts/deploy-to-server.sh` (not `scp` to `/home/ubuntu/app/`).
- **Target:** `/opt/diffrhythm`, Docker Compose prod. Restart via `docker-compose ... restart`, not `systemctl restart burntbeats-api`.

### Config
- **Stripe:** `config/ec2-config.env` template; `docker-compose.prod.yml` uses `env_file: .env` and explicit Stripe env vars.
- **Server .env:** Must contain real Stripe keys; deploy creates `.env` from `ec2-config.env` only if missing.

## Next Steps

1. **Ensure deploy completes:** Re-run `bash scripts/deploy-to-server.sh` if the previous run timed out during Docker build. Allow 30–60 min for build.
2. **Configure Stripe on server:** Update `/opt/diffrhythm/.env`, then restart API.
3. **Configure webhook** in Stripe Dashboard per `STRIPE_WEBHOOK_CONFIGURATION_FINAL.md`.
4. **Test E2E:** `bash scripts/e2e-payment-generation-check.sh`, then full payment → generation flow.
5. **Verify quality:** Generate test song and review output.

## Useful Commands

```bash
# Deploy
bash scripts/deploy-to-server.sh

# E2E check (after API is up)
BASE_URL=http://52.0.207.242:8000 bash scripts/e2e-payment-generation-check.sh

# Restart API only
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "cd /opt/diffrhythm && sudo docker-compose -f docker-compose.prod.yml restart diffrhythm-api"

# API logs
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "cd /opt/diffrhythm && sudo docker-compose -f docker-compose.prod.yml logs -f diffrhythm-api"
```

---

**Status:** SSH fixed; deploy flow verified through rsync. Complete Docker build → configure Stripe → webhook → E2E tests.
