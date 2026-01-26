# Payment and Storage System - Deployment Instructions
**Date**: January 26, 2026

## Deployment Steps

### 1. Copy Files to Server

```bash
# From local machine
scp -i "C:\Users\sammy\.ssh\server_saver_key" \
  d:\EMBERS-BANK\DiffRhythm-LINUX\backend\api.py \
  ubuntu@52.0.207.242:/tmp/api.py

scp -i "C:\Users\sammy\.ssh\server_saver_key" \
  d:\EMBERS-BANK\DiffRhythm-LINUX\backend\config.py \
  ubuntu@52.0.207.242:/tmp/config.py

scp -i "C:\Users\sammy\.ssh\server_saver_key" \
  d:\EMBERS-BANK\DiffRhythm-LINUX\backend\s3_storage.py \
  ubuntu@52.0.207.242:/tmp/s3_storage.py

scp -i "C:\Users\sammy\.ssh\server_saver_key" \
  d:\EMBERS-BANK\DiffRhythm-LINUX\backend\cleanup.py \
  ubuntu@52.0.207.242:/tmp/cleanup.py

scp -i "C:\Users\sammy\.ssh\server_saver_key" \
  d:\EMBERS-BANK\DiffRhythm-LINUX\backend\requirements.txt \
  ubuntu@52.0.207.242:/tmp/requirements.txt
```

### 2. Update Server Files

```bash
ssh -i "C:\Users\sammy\.ssh\server_saver_key" ubuntu@52.0.207.242 << 'EOF'
cd /opt/diffrhythm

# Backup existing files
sudo cp backend/api.py backend/api.py.backup.$(date +%Y%m%d_%H%M%S)
sudo cp backend/config.py backend/config.py.backup.$(date +%Y%m%d_%H%M%S)

# Copy new files
sudo cp /tmp/api.py backend/api.py
sudo cp /tmp/config.py backend/config.py
sudo cp /tmp/s3_storage.py backend/s3_storage.py
sudo cp /tmp/cleanup.py backend/cleanup.py

# Update requirements
sudo cp /tmp/requirements.txt backend/requirements.txt

echo "Files updated successfully"
EOF
```

### 3. Update .env File

```bash
ssh -i "C:\Users\sammy\.ssh\server_saver_key" ubuntu@52.0.207.242 << 'EOF'
cd /opt/diffrhythm

# Add new configuration to .env
cat >> .env << 'ENVEOF'

# S3 Configuration (optional)
S3_ENABLED=false
S3_BUCKET=
S3_REGION=us-east-1
S3_ACCESS_KEY=
S3_SECRET_KEY=
S3_PREFIX=songs/

# Cleanup Configuration
FILE_RETENTION_DAYS=30
CLEANUP_ENABLED=true
CLEANUP_INTERVAL_HOURS=24
ENVEOF

echo ".env updated"
EOF
```

### 4. Install boto3 (if not in image)

```bash
ssh -i "C:\Users\sammy\.ssh\server_saver_key" ubuntu@52.0.207.242 << 'EOF'
cd /opt/diffrhythm

# Option 1: Install in running container
sudo docker exec diffrhythm-api pip install boto3>=1.28.0

# Option 2: Rebuild image (if boto3 needs to be in image)
# sudo docker-compose -f docker-compose.prod.yml build --no-cache
EOF
```

### 5. Restart Container

```bash
ssh -i "C:\Users\sammy\.ssh\server_saver_key" ubuntu@52.0.207.242 << 'EOF'
cd /opt/diffrhythm

# Restart to load new code
sudo docker-compose -f docker-compose.prod.yml restart diffrhythm-api

# Wait for models to load
echo "Waiting for API to start..."
sleep 120

# Verify API is running
curl -s http://localhost:8000/api/v1/health | python3 -m json.tool
EOF
```

### 6. Verify Implementation

```bash
# Test payment flow
cd d:\EMBERS-BANK\DiffRhythm-LINUX
python test_payment_download_flow.py

# Check cleanup is scheduled
ssh -i "C:\Users\sammy\.ssh\server_saver_key" ubuntu@52.0.207.242 \
  "sudo docker logs diffrhythm-api | grep -i cleanup | tail -5"
```

## Configuration Options

### Enable S3 Storage

If you want to use S3 storage:

1. Create S3 bucket
2. Set IAM permissions
3. Update `.env`:
```env
S3_ENABLED=true
S3_BUCKET=diffrhythm-songs
S3_REGION=us-east-1
S3_ACCESS_KEY=your_access_key
S3_SECRET_KEY=your_secret_key
S3_PREFIX=songs/
```

### Adjust File Retention

To change retention period:

```env
FILE_RETENTION_DAYS=7  # Keep files for 7 days instead of 30
CLEANUP_INTERVAL_HOURS=12  # Run cleanup every 12 hours
```

### Enable Payment Requirement

To require payment for downloads:

```env
REQUIRE_PAYMENT_FOR_GENERATION=true
```

## Verification

After deployment, verify:

1. ✅ API starts successfully
2. ✅ Cleanup task starts (check logs)
3. ✅ Generation works without payment
4. ✅ Download requires payment (if enabled)
5. ✅ S3 upload works (if enabled)

---

**Status**: Ready for deployment
