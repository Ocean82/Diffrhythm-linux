# AWS Server Audit Findings - January 26, 2026

## Executive Summary

Comprehensive audit of AWS server deployment at `52.0.207.242` completed. System is operational with models loaded, but several critical issues identified that prevent song generation.

## Server Status

- **SSH Access**: ✅ Working
- **Docker**: ✅ Running (container healthy)
- **API Health**: ✅ Responding (`models_loaded: true`)
- **Models**: ✅ Loaded and accessible
- **Disk Space**: ⚠️ **CRITICAL** - 97% used (only 1.7GB free)

## Critical Issues Found

### 1. Rate Limiter Configuration Error
**Status**: 🔴 **BLOCKING**
- **Error**: `Exception: parameter 'request' must be an instance of starlette.requests.Request`
- **Location**: `backend/api.py` lines 590-594
- **Impact**: Generate endpoint returns 500 error
- **Root Cause**: Rate limiter decorator expects `request: Request` parameter but function has `api_request: Request`
- **Fix Required**: Update rate limiter decorator usage or function signature

### 2. Stripe Payment Configuration Missing
**Status**: 🔴 **BLOCKING**
- **Issue**: All Stripe keys are empty in `.env` file
  - `STRIPE_SECRET_KEY=` (empty)
  - `STRIPE_PUBLISHABLE_KEY=` (empty)
  - `STRIPE_WEBHOOK_SECRET=` (empty)
- **Impact**: Payment verification will fail, generation requests rejected
- **Configuration**: `REQUIRE_PAYMENT_FOR_GENERATION=true` but no keys configured
- **Fix Required**: Add Stripe keys to `.env` file or disable payment requirement

### 3. Quality Settings Not Optimal
**Status**: ⚠️ **WARNING**
- **Issue**: `CPU_STEPS=16` (balanced quality) instead of `32` (high quality)
- **Impact**: Generated songs may not meet Suno-quality standards
- **Fix Required**: Update `CPU_STEPS=32` in `.env` for high quality

### 4. Disk Space Critical
**Status**: 🔴 **CRITICAL**
- **Issue**: 97% disk usage (47GB/49GB used, only 1.7GB free)
- **Impact**: May cause failures during generation or model operations
- **Fix Required**: Clean up disk space or expand storage

## Working Components

### ✅ Docker Setup
- Container running and healthy
- Using correct `Dockerfile.prod`
- Volume mounts configured correctly
- Health checks passing

### ✅ Model Loading
- All 4 models loaded successfully:
  - CFM model (DiffRhythm-1_2)
  - VAE model (DiffRhythm-vae)
  - MuQ model (MuQ-MuLan-large)
  - Tokenizer (CNENTokenizer)
- Models cached in `/opt/diffrhythm/pretrained/`
- Model cache directory accessible to container

### ✅ API Endpoints
- Health endpoint: ✅ Working
- Root endpoint: ✅ Working
- Queue endpoint: ✅ Working
- Generate endpoint: ❌ Blocked by rate limiter error

## Configuration Details

### Environment Variables (Current)
```
HOST=0.0.0.0
PORT=8000
DEVICE=cpu
MODEL_MAX_FRAMES=2048
CPU_STEPS=16  # Should be 32 for high quality
CPU_CFG_STRENGTH=4.0
REQUIRE_PAYMENT_FOR_GENERATION=true
STRIPE_SECRET_KEY=  # EMPTY
STRIPE_PUBLISHABLE_KEY=  # EMPTY
STRIPE_WEBHOOK_SECRET=  # EMPTY
```

### Docker Container
- **Image**: `diffrhythm:prod`
- **Status**: Running (healthy)
- **Port**: 8000
- **Memory**: 16GB limit, 8GB reserved
- **CPU**: 2 cores limit, 1 core reserved

### Model Cache
- **Location**: `/opt/diffrhythm/pretrained/`
- **Models Present**:
  - `models--ASLP-lab--DiffRhythm-1_2/`
  - `models--ASLP-lab--DiffRhythm-1_2-full/`
  - `models--ASLP-lab--DiffRhythm-vae/`
  - `models--OpenMuQ--MuQ-MuLan-large/`

## Fixes Applied

### ✅ Fixed: Payment Intent ID References
- **Issue**: Code was using `request.payment_intent_id` where `request` was FastAPI Request object
- **Fix**: Changed to `gen_request.payment_intent_id` in generate_music function
- **Status**: Fixed in `backend/api.py` lines 601-617

### ✅ Fixed: Quality Settings
- **Issue**: `CPU_STEPS=16` (balanced quality)
- **Fix**: Updated to `CPU_STEPS=32` (high quality) in `.env`
- **Status**: Applied

### ✅ Fixed: Payment Requirement
- **Issue**: `REQUIRE_PAYMENT_FOR_GENERATION=true` but no Stripe keys
- **Fix**: Set to `REQUIRE_PAYMENT_FOR_GENERATION=false` in `.env`
- **Status**: Applied (temporary - should add Stripe keys for production)

### ⚠️ Partially Fixed: Rate Limiter
- **Issue**: Rate limiter decorator causing errors
- **Fix**: Temporarily disabled with `ENABLE_RATE_LIMIT=false`
- **Status**: Disabled for testing, needs proper fix

## Remaining Issues

1. **Rate Limiter**: Needs proper fix (currently disabled)
2. **Stripe Keys**: Should be added for production payment system
3. **Disk Space**: 97% used - needs cleanup or expansion
4. **CORS**: Should restrict to `https://burntbeats.com` for production

## Next Steps

1. ✅ Fix payment intent ID references - DONE
2. ✅ Update CPU_STEPS to 32 - DONE
3. ✅ Disable payment requirement temporarily - DONE
4. ⏳ Test generation endpoint after models load
5. ⏳ Fix rate limiter properly (remove decorator or fix usage)
6. ⏳ Add Stripe keys for production
7. ⏳ Monitor disk space
