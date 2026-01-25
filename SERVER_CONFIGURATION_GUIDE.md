# Server Configuration Guide

**Date:** January 24, 2026  
**Server:** ubuntu@52.0.207.242

## Current Server Configuration

### Service Configuration
- **Service:** `burntbeats-api`
- **Working Directory:** `/home/ubuntu/app/backend`
- **Command:** `python3 -m uvicorn main:app --host 127.0.0.1 --port 8001`
- **Status:** Active and running

### Environment Variables (.env)
**Location:** `/home/ubuntu/app/backend/.env`

**Current Settings:**
```
API_PORT=8001
API_HOST=127.0.0.1
STRIPE_SECRET_KEY=sk_live_... (configured)
STRIPE_PUBLISHABLE_KEY=pk_live_... (configured)
STRIPE_WEBHOOK_SECRET=whsec_... (configured)
REQUIRE_PAYMENT_FOR_GENERATION=true
```

**Missing:**
- `API_KEY` - Not set (authentication may be required by middleware)

## Issues Identified

### 1. API Key Authentication
**Problem:** Tests getting 401 "Authentication required"  
**Cause:** Server uses `src/middleware/auth.py` which requires authentication  
**Solution:** Either:
- Set `API_KEY` in `.env` and use it in tests, OR
- Disable API key requirement in server configuration

### 2. Server Code Structure
**Problem:** Server uses different codebase (`src/` structure) than local (`backend/` structure)  
**Impact:** 
- Server runs `main.py` from `backend/` directory
- Imports from `src.config`, `src.middleware`, `src.api`
- Different authentication middleware than `backend/api.py`

### 3. Webhook Endpoint
**Problem:** Webhook returning 500 errors  
**Cause:** Server uses `/home/ubuntu/app/backend/src/api/stripe_webhooks.py` (different file)  
**Status:** Webhook secret is configured, but endpoint may have different implementation

## Configuration Steps

### Step 1: Check Authentication Requirements

```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242
cd /home/ubuntu/app/backend
cat src/middleware/auth.py | grep -A 10 "ENABLE\|REQUIRE\|API_KEY"
cat src/config/settings.py | grep -i "api_key\|auth"
```

### Step 2: Configure API Key (If Required)

**Option A: Disable API Key Requirement**
```bash
# Check if there's a setting to disable
grep -r "ENABLE_API_KEY\|REQUIRE_API_KEY" src/
# If found, set to false in .env
```

**Option B: Set API Key for Testing**
```bash
# Add to .env
echo "API_KEY=test_api_key_12345" >> /home/ubuntu/app/backend/.env
# Restart service
sudo systemctl restart burntbeats-api
```

### Step 3: Verify Stripe Configuration

```bash
# Check Stripe keys are loaded
cd /home/ubuntu/app/backend
python3 -c "from src.config import settings; print('Stripe configured:', bool(settings.STRIPE_SECRET_KEY))"
```

### Step 4: Check Webhook Implementation

```bash
# Check webhook handler
cat src/api/stripe_webhooks.py | head -100
# Verify webhook secret matches .env
grep STRIPE_WEBHOOK_SECRET .env
```

### Step 5: Update Test Scripts

After configuring server, update test scripts to:
1. Include API key in headers (if required)
2. Use correct endpoint paths
3. Handle authentication properly

## Recommended Configuration

### For Testing (Development)
```bash
# Add to .env
API_KEY=test_dev_key_12345
ENABLE_API_KEY=false  # If this setting exists
```

### For Production
```bash
# Generate secure API key
API_KEY=$(openssl rand -hex 32)
# Add to .env
echo "API_KEY=$API_KEY" >> /home/ubuntu/app/backend/.env
```

## Next Steps

1. **Check server authentication middleware:**
   ```bash
   ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242
   cd /home/ubuntu/app/backend
   cat src/middleware/auth.py
   ```

2. **Configure API key or disable requirement**

3. **Update test scripts with authentication**

4. **Rerun tests**

---

**Status:** Need to check server authentication configuration  
**Action:** Review `src/middleware/auth.py` and configure accordingly
