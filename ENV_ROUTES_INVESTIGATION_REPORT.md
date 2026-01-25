# Environment Variables & Routes Investigation Report
**Date:** January 24, 2026

## Executive Summary
Comprehensive investigation of environment variable management, route configurations, and `.cursorignore` files to identify conflicts and improper paths.

## Findings

### 1. Environment Files

#### Local Repository
- **Found:** `config/ec2-config.env` (template file, not a secret)
- **Status:** Not in `.gitignore` (intentional - it's a template)
- **Recommendation:** Keep as-is, or add exception `!config/*.env.example` pattern

#### Server Configuration
- **Location:** `/home/ubuntu/app/backend/.env` (actual secrets)
- **Status:** Properly ignored by `.gitignore` (pattern: `backend/.env`)
- **Loading:** Server uses `load_dotenv()` in `main.py`

### 2. Environment Variable Loading

#### Local Code (`backend/config.py`)
- **Current:** Uses `os.getenv()` directly
- **Issue:** Does NOT load from `.env` files automatically
- **Impact:** Environment variables must be set in shell/system, not from `.env` files
- **Fix Required:** Add `load_dotenv()` support for local development

#### Server Code (`/home/ubuntu/app/backend/src/config.py`)
- **Current:** Uses `load_dotenv()` in `main.py` before importing config
- **Status:** ✅ Correctly configured

### 3. Route Configurations

#### Local Routes (`backend/api.py`)
- **API Prefix:** `Config.API_PREFIX` (defaults to `/api/v1`)
- **Routes:**
  - `POST /api/v1/generate` (or `/api/generate` alias)
  - `GET /api/v1/health`
  - `GET /api/v1/status/{job_id}`
  - `GET /api/v1/download/{job_id}`
  - `GET /api/v1/queue`
  - `POST /api/webhooks/stripe`

#### Server Routes (from context)
- **Main File:** `/home/ubuntu/app/backend/main.py`
- **Routers Registered:**
  1. `src.api.routes` → prefix `/api`
  2. `src.api.stripe_webhooks` → prefix `/api`
  3. `src.api.payments` → prefix `/api`
  4. `src.api.v1` → prefix `/api`
  5. `src.api.enhanced_routes` → prefix `/api` (optional)

#### Potential Route Conflicts
- **Issue:** Multiple routers with same `/api` prefix
- **Risk:** Route handler conflicts for `/api/v1/generate`
- **Status:** Server uses `src.api.v1` router which should handle `/api/v1/*` routes
- **Recommendation:** Verify route registration order and ensure no duplicate handlers

### 4. `.cursorignore` Files

#### Investigation Results
- **Found:** No `.cursorignore` files in repository
- **Status:** ✅ All files accessible to Cursor AI
- **User Requirement:** User mentioned checking `.cursorignore` to ensure no files are left unexplored
- **Action:** No action needed - all files are accessible

### 5. Environment Variable Path Issues

#### Potential Conflicts
1. **Local Development:**
   - `backend/config.py` doesn't load `.env` files
   - Developers must set env vars manually or in shell
   - **Fix:** Add `load_dotenv()` support

2. **Docker Deployment:**
   - `docker-compose.prod.yml` uses `env_file: - .env`
   - Also has inline `environment:` variables
   - **Status:** ✅ Correct - Docker Compose handles this

3. **Server Deployment:**
   - Server uses `load_dotenv()` in `main.py`
   - `.env` file at `/home/ubuntu/app/backend/.env`
   - **Status:** ✅ Correctly configured

## Issues Identified

### Critical Issues
1. **❌ Local `backend/config.py` doesn't load `.env` files**
   - **Impact:** Local development requires manual env var setup
   - **Fix:** Add `load_dotenv()` with proper path resolution

### Medium Priority Issues
1. **⚠️ `config/ec2-config.env` not explicitly handled in `.gitignore`**
   - **Impact:** Low (it's a template, not secrets)
   - **Fix:** Add exception pattern or document as template

2. **⚠️ Route registration order on server could cause conflicts**
   - **Impact:** Medium (potential for wrong route handler)
   - **Fix:** Verify server route registration order

### Low Priority Issues
1. **ℹ️ Multiple API entry points (`api.py` root vs `backend/api.py`)**
   - **Impact:** Low (different use cases)
   - **Status:** Documented, no action needed

## Recommended Fixes

1. **Add `load_dotenv()` to `backend/config.py`** for local development
2. **Update `.gitignore`** to explicitly handle template env files
3. **Verify server route registration** order to prevent conflicts
4. **Document environment variable loading order** for clarity

## Fixes Applied

### 1. Added `load_dotenv()` Support to `backend/config.py`
- **Change:** Added optional `python-dotenv` import and automatic `.env` file loading
- **Behavior:** 
  - Tries to load `backend/.env` first
  - Falls back to project root `.env`
  - Gracefully handles missing `python-dotenv` package
- **Impact:** Local development can now use `.env` files

### 2. Updated `.gitignore`
- **Change:** Added explicit exception for `config/ec2-config.env` template file
- **Pattern:** `!config/ec2-config.env`
- **Impact:** Template file is tracked, actual secrets remain ignored

### 3. Added `python-dotenv` to Requirements
- **Change:** Added `python-dotenv>=1.0.0` to `backend/requirements.txt`
- **Impact:** Ensures dependency is available for `.env` file loading

## Next Steps

1. ✅ Investigation complete
2. ✅ Fixes applied for environment variable loading
3. ⏳ Test local development with `.env` file loading
4. ⏳ Verify server route registration order (server-side verification needed)
