# Deployment Issue: Missing Model Files

**Date:** January 23, 2026  
**Status:** ⚠️ ISSUE IDENTIFIED

## Problem

The Docker container is failing to start because the `model` directory is empty on the server. The container logs show:

```
ERROR: Failed to import DiffRhythm modules: cannot import name 'DiT' from 'model' (unknown location)
```

## Root Cause

The `model/` directory on the server (`/opt/diffrhythm/model/`) is empty. The model Python files are not present:
- `model/__init__.py`
- `model/dit.py`
- `model/cfm.py`
- `model/modules.py`
- `model/utils.py`
- `model/trainer.py`

## Solution

The model files need to be present on the server before building the Docker image, OR the Docker build needs to include them.

### Option 1: Upload Model Files to Server (Recommended)

```bash
# From local machine
cd D:\EMBERS-BANK\DiffRhythm-LINUX
scp -i ~/.ssh/server_saver_key -r model/ ubuntu@52.0.207.242:/opt/diffrhythm/model/
```

Then rebuild the Docker image:
```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "cd /opt/diffrhythm && sudo docker build -f Dockerfile.prod -t diffrhythm:prod ."
```

### Option 2: Rebuild Docker Image with Model Files

If model files exist locally, ensure they're included in the Docker build context and rebuild:

```bash
# Verify model files exist locally
ls model/*.py

# Rebuild Docker image (this will copy model files)
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "cd /opt/diffrhythm && sudo docker build -f Dockerfile.prod -t diffrhythm:prod ."
```

### Option 3: Check Git Repository

If model files are in git but not on server:

```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "cd /opt/diffrhythm && git pull && git checkout model/"
```

## Verification

After uploading/rebuilding, verify model files exist:

```bash
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "ls -la /opt/diffrhythm/model/"
```

Should show:
- `__init__.py`
- `dit.py`
- `cfm.py`
- `modules.py`
- `utils.py`
- `trainer.py`

## Next Steps

1. Upload model files to server
2. Rebuild Docker image
3. Start container
4. Verify container starts successfully
5. Check health endpoint

---

**Status:** ⚠️ Action Required  
**Priority:** High - Container cannot start without model files
