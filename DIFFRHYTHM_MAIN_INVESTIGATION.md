# DiffRhythm-Main Directory Investigation

**Date:** 2026-01-24  
**Server:** ubuntu@52.0.207.242

## Summary

Investigated whether `diffrhythm-main` directory exists and if it's safe to remove.

## Findings

### Current Project Structure

- **Active Project Directory:** `/opt/diffrhythm` (16MB)
- **No `diffrhythm-main` directory found** in common locations:
  - `/opt/diffrhythm-main` - NOT FOUND
  - `/opt/diffrhythm/diffrhythm-main` - NOT FOUND
  - `/home/ubuntu/diffrhythm-main` - NOT FOUND
  - `/root/diffrhythm-main` - NOT FOUND

### Docker Build Context

**Dockerfile.prod Analysis:**
- Uses `COPY . .` which copies the current directory
- Build context is where `docker build` is executed
- Current build location: `/opt/diffrhythm`
- **No references to `diffrhythm-main` in Dockerfile**

### Codebase References

- **No references found** to `diffrhythm-main` in:
  - Dockerfile.prod
  - docker-compose.prod.yml
  - Deployment scripts
  - Configuration files

### Deployment Scripts

**scripts/deploy-to-server.sh:**
- Uses `PROJECT_DIR="/opt/diffrhythm"`
- Copies files to `/opt/diffrhythm/`
- **No mention of `diffrhythm-main`**

## Investigation Results

**Status:** ✅ Found `DiffRhythm-main` directory at `~/app/models/DiffRhythm-main`

**Directory Details:**
- **Location:** `/home/ubuntu/app/models/DiffRhythm-main`
- **Size:** **7.6GB** (7.5GB in `pretrained/` directory)
- **Structure:** Old version (no `backend` directory)
- **Last Modified:** December 22, 2025 (over a month ago)
- **Docker Status:** NOT mounted in any containers

**Current Setup:**
- Active project: `/opt/diffrhythm` (16MB, has `backend` directory)
- Docker builds from: `/opt/diffrhythm`
- Different from `DiffRhythm-main` at `~/app/models/`

## Conclusion

### If `diffrhythm-main` Exists Elsewhere

If you find a `diffrhythm-main` directory (possibly from a GitHub clone that extracted to a `-main` suffix):

1. **It's NOT part of the current Docker build**
   - Docker builds from `/opt/diffrhythm`
   - No references in Dockerfile or scripts

2. **It's likely an old/unused directory**
   - Could be from an initial clone/download
   - Not referenced in any active scripts

3. **Safe to remove if:**
   - Not currently being used
   - Not mounted in Docker containers
   - Not referenced in any running services

### Verification Steps

Before removing, verify:
```bash
# Check if it's mounted in Docker
docker ps --format "{{.Mounts}}" | grep diffrhythm-main

# Check if it's referenced in any configs
grep -r "diffrhythm-main" /opt/diffrhythm

# Check disk usage
du -sh /path/to/diffrhythm-main
```

### Safe Removal Command

If confirmed unused:
```bash
# Backup first (optional)
sudo tar -czf /tmp/diffrhythm-main-backup.tar.gz /path/to/diffrhythm-main

# Remove
sudo rm -rf /path/to/diffrhythm-main
```

## Current Status

- ✅ **No `diffrhythm-main` directory found on server**
- ✅ Current project is at `/opt/diffrhythm` (16MB)
- ✅ Docker builds from `/opt/diffrhythm` (uses `COPY . .`)
- ✅ Dockerfile.prod does NOT reference `diffrhythm-main`
- ✅ Deployment scripts use `/opt/diffrhythm`, not `diffrhythm-main`

## Final Answer

**`DiffRhythm-main` at `~/app/models/DiffRhythm-main` is NOT part of the current Docker build.**

### Key Findings

1. **Size:** 7.6GB (7.5GB in pretrained models)
2. **Structure:** Old version (no `backend` directory)
3. **Location:** `~/app/models/` (not `/opt/diffrhythm`)
4. **Docker:** NOT mounted, NOT used
5. **Last Modified:** December 22, 2025 (old)

### Comparison

| Feature | DiffRhythm-main | Current Project |
|---------|----------------|-----------------|
| Location | `~/app/models/DiffRhythm-main` | `/opt/diffrhythm` |
| Size | 7.6GB | 16MB |
| Structure | Old (no `backend`) | New (has `backend`) |
| Docker | Not used | Active |
| Models | 7.5GB downloaded | Not downloaded yet |

### Safe to Remove

✅ **Safe to remove** - it's not used by Docker  
✅ **Safe to remove** - not referenced in any scripts  
✅ **Safe to remove** - current project is at `/opt/diffrhythm`  
✅ **Safe to remove** - old structure (no `backend` directory)  
✅ **Safe to remove** - will free **7.6GB** of disk space

### Removal

**Space to be freed:** 7.6GB

**Command:**
```bash
# Verify first
ls -lah ~/app/models/DiffRhythm-main

# Remove
rm -rf ~/app/models/DiffRhythm-main
```

**Recommendation:** ✅ **SAFE TO REMOVE** - This is an old version that's not part of the current Docker build or deployment. Removing it will free 7.6GB of disk space.
