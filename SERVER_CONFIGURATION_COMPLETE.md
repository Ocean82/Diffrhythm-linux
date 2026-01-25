# Server Configuration - Complete Guide

**Date:** January 24, 2026  
**Server:** ubuntu@52.0.207.242

## Current Server Analysis

### Server Codebase Structure
- **Main:** `/home/ubuntu/app/backend/main.py`
- **Routes:** `src/api/routes.py`
- **Auth:** `src/middleware/auth.py` (Clerk authentication)
- **Config:** `src/config/settings.py`

### Generate Endpoint Details
- **Path:** `/api/v1/generate`
- **Method:** POST
- **Authentication:** Required (Clerk Bearer token)
- **Request Format:**
  ```json
  {
    "text_prompt": "A happy upbeat pop song",
    "genre": "pop",
    "style": "upbeat",
    "duration": 95.0
  }
  ```

### Authentication Flow
1. Endpoint receives `Authorization: Bearer <token>` header
2. Calls `verify_clerk_token(authorization)`
3. If token is valid, returns `user_id`
4. If `user_id` is None, raises 401 "Authentication required"

## Configuration Options

### Option 1: Configure Clerk for Testing (Recommended)

**If you have Clerk account:**
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242
cd /home/ubuntu/app/backend

# Get Clerk secret key from Clerk Dashboard
# Add to .env
echo "CLERK_SECRET_KEY=sk_test_..." >> .env

# Restart service
sudo systemctl restart burntbeats-api
```

**Then update test scripts to include Clerk token:**
```python
headers = {
    "Authorization": "Bearer <clerk_token>",
    "Content-Type": "application/json"
}
```

### Option 2: Modify Server to Allow Testing Without Auth

**Temporarily modify the generate endpoint to allow testing:**

```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242
cd /home/ubuntu/app/backend

# Backup original
cp src/api/routes.py src/api/routes.py.backup

# Modify to allow testing (add environment check)
# Edit src/api/routes.py around line 580
# Change:
#   user_id = await verify_clerk_token(authorization)
#   if not user_id:
#       raise HTTPException(status_code=401, detail="Authentication required")
# To:
#   from ..config import settings
#   if settings.ENVIRONMENT == "development" or not settings.CLERK_SECRET_KEY:
#       user_id = "test_user"  # Allow testing without auth
#   else:
#       user_id = await verify_clerk_token(authorization)
#       if not user_id:
#           raise HTTPException(status_code=401, detail="Authentication required")

# Restart service
sudo systemctl restart burntbeats-api
```

### Option 3: Add Test Mode Setting

**Add to .env:**
```bash
ENVIRONMENT=development
ENABLE_AUTH=false  # If this setting exists in code
```

## Recommended Configuration Steps

### Step 1: Check Current Settings

```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242
cd /home/ubuntu/app/backend
cat .env | grep -E "ENVIRONMENT|CLERK|DEBUG"
```

### Step 2: Configure Based on Option Chosen

**For Development/Testing:**
- Set `ENVIRONMENT=development` in `.env`
- Optionally modify routes to bypass auth in development mode
- Or configure Clerk with test keys

**For Production:**
- Set `CLERK_SECRET_KEY` with production key
- Ensure `ENVIRONMENT=production`
- Keep authentication enabled

### Step 3: Update Test Scripts

After configuration, update test scripts to:
1. Use correct request format (`text_prompt`, `genre`, `style`, `duration`)
2. Include Authorization header if auth is enabled
3. Use correct endpoint paths

## Current Status

✅ **Stripe Configuration:**
- Secret key: Configured
- Publishable key: Configured
- Webhook secret: Configured
- Payment required: true

⚠️ **Authentication:**
- Clerk secret key: Not set
- Authentication: Required for generate endpoint
- Test scripts: Need to be updated

✅ **Service:**
- Running on port 8001
- Health endpoint: Working

## Next Steps

1. **Choose configuration option** (Option 1, 2, or 3)
2. **Apply configuration** to server
3. **Update test scripts** to match server API format
4. **Rerun tests**

---

**Status:** Server requires Clerk authentication  
**Action Required:** Configure Clerk or modify server for testing
