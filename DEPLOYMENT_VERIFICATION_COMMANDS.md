# Deployment Verification Commands

**Date:** January 23, 2026  
**Server:** ubuntu@52.0.207.242

## Quick Verification Commands

Run these commands via SSH to verify deployment status:

### 1. Check Server Connectivity
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "echo 'Connected'"
```

### 2. Check Docker Installation
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "docker --version"
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "docker-compose --version"
```

### 3. Check Project Directory
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "ls -la /opt/diffrhythm"
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "du -sh /opt/diffrhythm"
```

### 4. Check Docker Image
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "sudo docker images diffrhythm:prod"
```

### 5. Check Docker Container
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "sudo docker ps -a | grep diffrhythm-api"
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "sudo docker ps | grep diffrhythm-api"
```

### 6. Check Container Health
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "sudo docker inspect diffrhythm-api --format '{{.State.Health.Status}}'"
```

### 7. Check Container Logs
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "sudo docker logs diffrhythm-api --tail 50"
```

### 8. Check API Health Endpoint
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "curl -s http://localhost:8000/api/v1/health | python3 -m json.tool"
```

### 9. Check Disk Space
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "df -h /"
```

### 10. Check Port 8000
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "netstat -tuln | grep 8000 || ss -tuln | grep 8000"
```

## Complete Verification Script

### Using Bash Script (Linux/Mac/WSL)
```bash
cd /path/to/DiffRhythm-LINUX
bash scripts/verify_server_deployment.sh
```

### Using PowerShell Script (Windows)
```powershell
cd D:\EMBERS-BANK\DiffRhythm-LINUX
powershell -ExecutionPolicy Bypass -File scripts\verify_deployment_ssh.ps1
```

## Manual Deployment Steps

If deployment needs to be done manually:

### 1. Build Docker Image
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "cd /opt/diffrhythm && sudo docker build -f Dockerfile.prod -t diffrhythm:prod ."
```

### 2. Start Services
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "cd /opt/diffrhythm && sudo docker-compose -f docker-compose.prod.yml up -d"
```

### 3. Check Service Status
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "cd /opt/diffrhythm && sudo docker-compose -f docker-compose.prod.yml ps"
```

### 4. View Logs
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "sudo docker logs -f diffrhythm-api"
```

### 5. Restart Services
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "cd /opt/diffrhythm && sudo docker-compose -f docker-compose.prod.yml restart"
```

### 6. Stop Services
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "cd /opt/diffrhythm && sudo docker-compose -f docker-compose.prod.yml down"
```

## Test Generation Endpoint

Once the API is running and models are loaded:

```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "curl -X POST http://localhost:8000/api/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{\"lyrics\":\"[00:00.00]Test song\n[00:05.00]This is a test\",\"style_prompt\":\"pop, upbeat, energetic\",\"audio_length\":95}'"
```

## Expected Results

### Successful Deployment Should Show:

1. **Docker Image:** `diffrhythm:prod` exists
2. **Container:** `diffrhythm-api` is running
3. **Health Status:** `healthy` or `starting`
4. **API Response:** JSON with `models_loaded: true`
5. **Port 8000:** Listening
6. **Disk Space:** > 10GB free

### Common Issues:

1. **Container not running:** Start with `docker-compose up -d`
2. **Models not loaded:** Wait 2-5 minutes, check logs
3. **Health check failing:** Check container logs for errors
4. **Port not accessible:** Check firewall/security groups
5. **Disk space low:** Clean up or increase storage

## Monitoring Commands

### Watch Logs in Real-Time
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "docker logs -f diffrhythm-api"
```

### Monitor Resource Usage
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "docker stats diffrhythm-api --no-stream"
```

### Check API Metrics
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "curl -s http://localhost:8000/api/v1/metrics"
```

## Troubleshooting

### If Container Won't Start:
```bash
# Check logs
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "docker logs diffrhythm-api"

# Check Docker system
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "docker system df"

# Clean up if needed
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "docker system prune -f"
```

### If Models Won't Load:
```bash
# Check model cache directory
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "ls -lh /opt/diffrhythm/pretrained/"

# Check network connectivity
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "curl -I https://huggingface.co"

# Check container logs for model loading errors
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "docker logs diffrhythm-api | grep -i model"
```

### If API Not Responding:
```bash
# Check if container is running
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "docker ps | grep diffrhythm-api"

# Check port binding
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "docker port diffrhythm-api"

# Test from inside container
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "docker exec diffrhythm-api curl -s http://localhost:8000/api/v1/health"
```
