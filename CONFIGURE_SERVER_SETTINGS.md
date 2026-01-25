# Server Configuration Guide

**Date:** January 24, 2026  
**Server:** ubuntu@52.0.207.242

## Current Server Status

### Service Configuration
- ✅ Service running: `burntbeats-api`
- ✅ Port: 8001
- ✅ Health endpoint: Working

### Environment Variables
**Location:** `/home/ubuntu/app/backend/.env`

**Configured:**
- ✅ `STRIPE_SECRET_KEY` - Set
- ✅ `STRIPE_PUBLISHABLE_KEY` - Set
- ✅ `STRIPE_WEBHOOK_SECRET` - Set
- ✅ `REQUIRE_PAYMENT_FOR_GENERATION=true`

**Missing/Issues:**
- ❌ `CLERK_SECRET_KEY` - Not set (required for authentication)
- ❌ `API_KEY` - Not set (may be optional)

## Server Codebase Structure

**Server uses different structure:**
- Main file: `/home/ubuntu/app/backend/main.py`
- Routes: `src/api/routes.py`
- Config: `src/config/settings.py`
- Auth: `src/middleware/auth.py`

**Generate Endpoint:**
- Path: `/api/v1/generate`
- Method: POST
- Authentication: Required (Bearer token via Clerk)
- Request format: `{"text_prompt": "...", "genre": "...", "style": "...", "duration": 95.0}`

## Configuration Steps

### Step 1: Configure Clerk Authentication (If Required)

**Option A: Set CLERK_SECRET_KEY**
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242
cd /home/ubuntu/app/backend
# Add to .env
echo "CLERK_SECRET_KEY=your_clerk_secret_key" >> .env
sudo systemctl restart burntbeats-api
```

**Option B: Disable Authentication (For Testing)**
```bash
# Check if there's a setting to disable auth
grep -r "ENABLE_AUTH\|REQUIRE_AUTH\|AUTH_REQUIRED" src/
# If found, set to false in .env
```

### Step 2: Verify Stripe Configuration

```bash
# Check Stripe keys are loaded correctly
cd /home/ubuntu/app/backend
python3 -c "from src.config import settings; print('Stripe Secret:', bool(settings.STRIPE_SECRET_KEY)); print('Webhook Secret:', bool(settings.STRIPE_WEBHOOK_SECRET))"
```

### Step 3: Check Webhook Endpoint

```bash
# Verify webhook handler exists
cat src/api/stripe_webhooks.py | head -50
# Check webhook secret matches
grep STRIPE_WEBHOOK_SECRET .env
```

### Step 4: Update Test Scripts

After configuring authentication, update test scripts to:
1. Use correct request format (`text_prompt` instead of `lyrics`)
2. Include Bearer token in Authorization header
3. Use correct endpoint paths

## Recommended Configuration

### For Testing (Development)

**If Clerk authentication can be disabled:**
```bash
# Add to .env
ENABLE_AUTH=false  # If this setting exists
# OR
CLERK_SECRET_KEY=  # Empty to disable
```

**If Clerk authentication is required:**
```bash
# Get Clerk secret key from Clerk Dashboard
# Add to .env
CLERK_SECRET_KEY=sk_test_...  # Or sk_live_... for production
```

### For Production

```bash
# Set production Clerk secret
CLERK_SECRET_KEY=sk_live_...
# Ensure all Stripe keys are live keys
STRIPE_SECRET_KEY=sk_live_...
STRIPE_PUBLISHABLE_KEY=pk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
```

## Next Steps

1. **Check authentication requirements:**
   ```bash
   ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242
   cd /home/ubuntu/app/backend
   cat src/middleware/auth.py | grep -A 20 "def verify"
   ```

2. **Configure Clerk or disable authentication**

3. **Update test scripts** to match server's API format

4. **Rerun tests**

---

**Status:** Server requires Clerk authentication  
**Action:** Configure CLERK_SECRET_KEY or disable authentication for testing
