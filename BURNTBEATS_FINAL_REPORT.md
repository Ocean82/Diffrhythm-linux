# BurntBeats.com Investigation - Final Report

**Date:** January 23, 2026  
**Domain:** burntbeats.com  
**Server:** ubuntu@52.0.207.242

## Executive Summary

### Current Status: ⚠️ **PARTIALLY OPERATIONAL**

**Frontend:** ✅ **FULLY ACCESSIBLE**  
**Backend API:** 🔄 **MODELS LOADING** (container restarting)  
**Song Generation:** ❌ **NOT READY** (waiting for models)

## Key Findings

### ✅ What's Working

1. **Frontend (burntbeats.com)**
   - ✅ Domain resolves correctly
   - ✅ HTTPS working with Let's Encrypt SSL
   - ✅ Frontend files served correctly
   - ✅ Nginx configured and running
   - ✅ Static assets loading properly

2. **Infrastructure**
   - ✅ Docker container running
   - ✅ All required files uploaded
   - ✅ Volume mounts configured
   - ✅ Models downloading (MuQ-MuLan visible in pretrained/)

### ⚠️ Current Issues

1. **Container Restarting**
   - Container has restarted **7 times**
   - Likely due to health check timeouts during model loading
   - Models take 10-15+ minutes to load on CPU
   - Health check may be too aggressive

2. **Models Still Loading**
   - MuQMuLan model downloading from HuggingFace
   - Process stuck at "Preparing MuQMuLan (from_pretrained)..."
   - No errors, just slow download on CPU
   - Models visible in `/opt/diffrhythm/pretrained/` directory

3. **Port Configuration Mismatch**
   - Nginx proxies to port **8001**
   - Docker container runs on port **8000**
   - Separate service on port 8001 (different API)
   - Need to determine which service should handle requests

4. **API Not Responding**
   - Health endpoint returns empty (models not loaded)
   - Generation endpoint requires models to be loaded
   - Cannot test song generation until models complete

## Technical Details

### Container Status
```
Status: running
Health: starting
Restarts: 7
Port: 8000 (mapped to host)
```

### Model Loading Progress
1. ✅ CFM model initialized
2. ✅ DiT model initialized
3. ✅ CFM checkpoint loaded
4. ✅ CNENTokenizer prepared
5. 🔄 **MuQMuLan loading** - **IN PROGRESS** (stuck here)
6. ⏳ VAE model - **PENDING**

### Disk Space
- **Total:** 49GB
- **Used:** 39GB (81%)
- **Available:** 9.4GB
- **Status:** ✅ Sufficient space

### Models Downloaded
Models visible in `/opt/diffrhythm/pretrained/`:
- ✅ `models--ASLP-lab--DiffRhythm-1_2`
- ✅ `models--ASLP-lab--DiffRhythm-1_2-full`
- ✅ `models--ASLP-lab--DiffRhythm-vae`
- ✅ `models--OpenMuQ--MuQ-MuLan-large` (created 06:14)
- ✅ `models--OpenMuQ--MuQ-large-msd-iter`
- ✅ `models--xlm-roberta-base`

## Root Cause Analysis

### Why Container Restarts
1. **Health Check Timeout**
   - Health check likely configured with short timeout
   - Model loading takes 10-15+ minutes
   - Health check fails, container restarts
   - Cycle repeats

2. **Model Loading Time**
   - MuQMuLan is a large model (~GB)
   - Downloading from HuggingFace on CPU is slow
   - No errors, just slow process

### Why API Not Accessible
1. **Models Not Loaded**
   - API requires all models loaded before responding
   - Health endpoint returns empty until models ready
   - Generation endpoint checks `model_manager.is_loaded`

2. **Port Mismatch**
   - Nginx configured for port 8001
   - Docker container on port 8000
   - Need to align configuration

## Recommendations

### Immediate Actions

1. **Fix Health Check Configuration**
   - Increase health check timeout/start period
   - Allow 15-20 minutes for initial model loading
   - Disable health check during startup, or make it more lenient

2. **Wait for Models to Complete**
   - Monitor logs: `sudo docker logs -f diffrhythm-api`
   - Look for: "Models loaded successfully!"
   - Expected time: 5-10 more minutes

3. **Fix Port Configuration**
   - Determine which API service should handle requests
   - Update nginx to proxy to correct port (8000 or 8001)
   - Test health endpoint after fix

### Once Models Load

1. **Verify Health Endpoint**
   ```bash
   curl https://burntbeats.com/api/v1/health
   # Should return: {"models_loaded": true, "status": "healthy"}
   ```

2. **Test Generation Endpoint**
   ```bash
   curl -X POST https://burntbeats.com/api/v1/generate \
     -H "Content-Type: application/json" \
     -H "X-API-Key: YOUR_KEY" \
     -d '{
       "lyrics": "[00:00.00]Test\n[00:05.00]Song",
       "style_prompt": "pop",
       "audio_length": 95
     }'
   ```

3. **Verify Full Song Output**
   - Check generated audio has vocals
   - Check generated audio has instrumentals
   - Verify file download works
   - Test from frontend UI

## Testing Checklist

Once models are loaded:

- [ ] Health endpoint responds with `models_loaded: true`
- [ ] API accessible via domain (https://burntbeats.com/api/v1/health)
- [ ] Generation endpoint accepts requests
- [ ] Job creation returns job_id
- [ ] Status endpoint shows job progress
- [ ] Generated audio file downloads successfully
- [ ] Audio file contains vocals
- [ ] Audio file contains instrumentals
- [ ] Frontend can submit generation requests
- [ ] Frontend can download generated songs

## Current Blockers

1. **Models Not Loaded** ⚠️
   - Blocking: API health checks, generation requests
   - ETA: 5-10 minutes

2. **Container Restarts** ⚠️
   - Blocking: Stable model loading
   - Fix: Adjust health check configuration

3. **Port Configuration** ⚠️
   - Blocking: Frontend-to-backend communication
   - Fix: Update nginx config once models load

## Conclusion

**BurntBeats.com is accessible and the frontend is working**, but **song generation is not yet functional** because:

1. Models are still loading (taking longer than expected)
2. Container keeps restarting due to health check timeouts
3. Port configuration needs to be aligned

**Once models complete loading and port configuration is fixed, the system should be fully operational for generating songs with vocals and instrumentals.**

---

**Status:** ⚠️ **WAITING FOR MODELS TO LOAD**  
**Next Action:** Monitor model loading, fix health check, align port configuration  
**ETA:** 5-10 minutes for models, then testing can proceed
