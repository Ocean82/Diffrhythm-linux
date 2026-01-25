# BurntBeats.com Investigation Report

**Date:** January 23, 2026  
**Server:** ubuntu@52.0.207.242  
**Domain:** burntbeats.com

## Current Status

### ✅ Frontend Accessibility
- **Domain:** burntbeats.com is **ACCESSIBLE**
- **HTTPS:** ✅ Working (redirects HTTP to HTTPS)
- **Frontend Files:** ✅ Present at `/home/ubuntu/app/frontend/dist/`
- **Nginx:** ✅ Running and configured

### ⚠️ API Status
- **Docker Container:** ✅ Running on port 8000
- **Models:** 🔄 **STILL LOADING** (MuQMuLan downloading from HuggingFace)
- **Health Endpoint:** ❌ Not responding (models not loaded yet)
- **Port 8001:** ⚠️ Different service running (not the Docker container)

## Configuration Analysis

### Nginx Configuration
- **Location:** `/etc/nginx/sites-available/burntbeats`
- **Proxy Target:** `http://127.0.0.1:8001/api/` (port 8001)
- **Frontend Root:** `/home/ubuntu/app/frontend/dist`
- **SSL:** ✅ Configured with Let's Encrypt

### Services Running
1. **Docker Container (diffrhythm-api)**
   - Port: 8000
   - Status: Running, models loading
   - Process: `python3 -m uvicorn backend.api:app --host 0.0.0.0 --port 8000`

2. **Native Python Service (port 8001)**
   - Port: 8001
   - Process: `/usr/bin/python3 -m uvicorn main:app --host 127.0.0.1 --port 8001`
   - Status: Running (different service, not Docker)

### Model Loading Status
From logs, models are stuck at:
1. ✅ CFM model initialized
2. ✅ DiT model initialized
3. ✅ CFM checkpoint loaded
4. ✅ CNENTokenizer prepared
5. 🔄 **MuQMuLan loading** (from_pretrained) - **IN PROGRESS**
6. ⏳ VAE model loading - **PENDING**

**Issue:** Models are taking a very long time to load. MuQMuLan is downloading from HuggingFace which can take 5-10 minutes on CPU.

## Issues Identified

### 1. Port Mismatch
- Nginx proxies to port **8001**
- Docker container runs on port **8000**
- There's a separate service on port 8001 (different API)

**Action Needed:** Either:
- Update nginx to proxy to port 8000 (Docker container), OR
- Ensure the service on 8001 is the correct one and has models loaded

### 2. Models Still Loading
- Models have been loading for 20+ minutes
- MuQMuLan download from HuggingFace is slow on CPU
- Health endpoint won't respond until models are loaded

**Action Needed:** Wait for models to complete loading, or investigate why it's taking so long.

### 3. API Health Check
- Health endpoint not responding because models aren't loaded
- Frontend may not be able to connect to API

## Testing Results

### Frontend
```bash
curl -I https://burntbeats.com
# Result: HTTP/2 200 ✅
```

### API via Nginx
```bash
curl https://burntbeats.com/api/v1/health
# Result: Empty response (models not loaded) ❌
```

### API Direct (Docker)
```bash
curl http://localhost:8000/api/v1/health
# Result: Empty response (models not loaded) ❌
```

### API Direct (Port 8001)
```bash
curl http://127.0.0.1:8001/health
# Result: Need to check what this service is
```

## Next Steps

1. **Wait for Models to Load** (5-10 more minutes)
   - Monitor: `sudo docker logs -f diffrhythm-api`
   - Look for: "Models loaded successfully!"

2. **Verify Port Configuration**
   - Check if port 8001 service is the correct one
   - Update nginx if needed to point to port 8000

3. **Test Generation Endpoint**
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

4. **Test Full Song Generation**
   - Submit a generation job
   - Check status endpoint
   - Download generated audio
   - Verify it has both instrumentals and vocals

## Recommendations

1. **Fix Port Configuration**
   - Determine which service should handle API requests
   - Update nginx to proxy to the correct port
   - Ensure only one API service is running

2. **Monitor Model Loading**
   - Models are taking longer than expected
   - Consider pre-downloading models to speed up startup
   - Add better logging for model loading progress

3. **Health Check**
   - Once models load, verify health endpoint works
   - Test from both localhost and via nginx
   - Verify frontend can connect

4. **Full Integration Test**
   - Test complete song generation flow
   - Verify output has vocals and instrumentals
   - Check file download works

---

**Status:** ⚠️ **PARTIALLY OPERATIONAL**
- Frontend: ✅ Accessible
- API: 🔄 Models loading (not ready yet)
- Generation: ⏳ Waiting for models to load

**ETA:** 5-10 minutes for models to complete loading
