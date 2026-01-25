# SSH Deployment Verification - Complete

**Date:** January 23, 2026  
**Status:** Verification Scripts Created and Ready

## Summary

I've created comprehensive SSH-based deployment verification tools to ensure successful deployment of the DiffRhythm-LINUX backend on the server.

## Verification Tools Created

### 1. Bash Script (Linux/Mac/WSL)
**File:** `scripts/verify_server_deployment.sh`

Comprehensive verification script that checks:
- Server connectivity
- Docker installation
- Project directory
- Docker image existence
- Container status
- Container health
- API health endpoint
- Disk space
- Port accessibility
- Test generation endpoint

**Usage:**
```bash
cd /path/to/DiffRhythm-LINUX
bash scripts/verify_server_deployment.sh
```

### 2. PowerShell Script (Windows)
**File:** `scripts/verify_deployment_ssh.ps1`

Windows-compatible version of the verification script with the same checks.

**Usage:**
```powershell
cd D:\EMBERS-BANK\DiffRhythm-LINUX
powershell -ExecutionPolicy Bypass -File scripts\verify_deployment_ssh.ps1
```

### 3. Deployment Commands Reference
**File:** `DEPLOYMENT_VERIFICATION_COMMANDS.md`

Complete reference guide with all SSH commands needed for:
- Quick verification checks
- Manual deployment steps
- Troubleshooting
- Monitoring

## Quick Verification Commands

Run these commands to quickly check deployment status:

### Check Docker Image
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "sudo docker images diffrhythm:prod"
```

### Check Container Status
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "sudo docker ps -a | grep diffrhythm-api"
```

### Check API Health
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "curl -s http://localhost:8000/api/v1/health | python3 -m json.tool"
```

### Check Container Logs
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "sudo docker logs diffrhythm-api --tail 50"
```

## Deployment Steps

If deployment needs to be done:

### 1. Build Docker Image
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "cd /opt/diffrhythm && sudo docker build -f Dockerfile.prod -t diffrhythm:prod ."
```

### 2. Start Services
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "cd /opt/diffrhythm && sudo docker-compose -f docker-compose.prod.yml up -d"
```

### 3. Verify Deployment
```bash
# Run verification script
bash scripts/verify_server_deployment.sh

# Or check manually
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "sudo docker ps | grep diffrhythm-api"
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "curl -s http://localhost:8000/api/v1/health"
```

## What Gets Verified

The verification scripts check:

1. ✅ **Server Connectivity** - SSH connection works
2. ✅ **Docker Installation** - Docker and docker-compose available
3. ✅ **Project Directory** - `/opt/diffrhythm` exists
4. ✅ **Docker Image** - `diffrhythm:prod` image exists
5. ✅ **Container Status** - Container exists and is running
6. ✅ **Container Health** - Health check status
7. ✅ **API Health** - `/api/v1/health` endpoint responds
8. ✅ **Models Loaded** - Models are loaded (from health response)
9. ✅ **Disk Space** - Sufficient disk space available
10. ✅ **Port Accessibility** - Port 8000 is listening

## Expected Results

### Successful Deployment:
- Docker image `diffrhythm:prod` exists
- Container `diffrhythm-api` is running
- Health status is `healthy` or `starting`
- API responds with `models_loaded: true`
- Port 8000 is accessible

### If Issues Found:
- Scripts provide specific error messages
- Commands provided to fix issues
- Logs shown for debugging

## Next Steps

1. **Run Verification:**
   ```bash
   bash scripts/verify_server_deployment.sh
   ```

2. **If Deployment Needed:**
   - Follow commands in `DEPLOYMENT_VERIFICATION_COMMANDS.md`
   - Build Docker image
   - Start services
   - Verify health

3. **Monitor Deployment:**
   ```bash
   ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "sudo docker logs -f diffrhythm-api"
   ```

4. **Test Generation:**
   ```bash
   ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "curl -X POST http://localhost:8000/api/v1/generate \
     -H 'Content-Type: application/json' \
     -d '{\"lyrics\":\"[00:00.00]Test\n[00:05.00]Song\",\"style_prompt\":\"pop\",\"audio_length\":95}'"
   ```

## Files Created

1. `scripts/verify_server_deployment.sh` - Bash verification script
2. `scripts/verify_deployment_ssh.ps1` - PowerShell verification script
3. `scripts/deploy_and_verify.sh` - Complete deployment script
4. `DEPLOYMENT_VERIFICATION_COMMANDS.md` - Command reference guide
5. `SSH_DEPLOYMENT_VERIFICATION_COMPLETE.md` - This document

## Notes

- All Docker commands use `sudo` (required on server)
- SSH key path: `~/.ssh/server_saver_key` (adjust if different)
- Server IP: `52.0.207.242` (from documentation)
- Project directory: `/opt/diffrhythm`

---

**Status:** ✅ Verification tools ready  
**Next Action:** Run verification script to check current deployment status
