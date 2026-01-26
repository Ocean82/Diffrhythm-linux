# Disk Space Cleanup Analysis
**Date**: January 26, 2026  
**Server**: ubuntu@52.0.207.242  
**Current Disk Usage**: 97% (47GB/49GB used, 1.7GB free)  
**Status**: ⚠️ **CRITICAL**

## Backup Directory Analysis: `/home/ubuntu/backups/`

**Total Size**: 2.4GB

### Backup Files

| File | Size | Date | Status | Recommendation |
|------|------|------|--------|----------------|
| `app-models-backup-.tar.gz` | 0 bytes | Dec 9 | Empty | ✅ **DELETE** |
| `backup-.tar.gz` | 312KB | Dec 22 | Small backup | ⚠️ Review |
| `backup-20251222-103614.tar.gz` | 312KB | Dec 22 | Duplicate? | ⚠️ Review |
| `burntbeats-full-backup-.tar.gz` | **1.5GB** | Dec 9 | Large backup | ⚠️ **Archive or Delete** |
| `phoenix-full-backup-.tar.gz` | 346MB | Dec 9 | Medium backup | ⚠️ Review |

### Backup Directories

| Directory | Size | Purpose | Recommendation |
|-----------|------|---------|----------------|
| `cleanup-20251229-190056/` | **615MB** | Cleanup artifacts | ✅ **DELETE** |
| `cleanup_/` | 8KB | Old cleanup | ✅ **DELETE** |
| `old-scripts/` | 120KB | Old scripts | ⚠️ Review |

### Analysis

1. **Empty Backup**: `app-models-backup-.tar.gz` (0 bytes) - Safe to delete
2. **Duplicate Backups**: `backup-.tar.gz` and `backup-20251222-103614.tar.gz` appear identical - Keep one, delete other
3. **Large Backups**: 
   - `burntbeats-full-backup-.tar.gz` (1.5GB) - From Dec 9 (old)
   - `phoenix-full-backup-.tar.gz` (346MB) - From Dec 9 (old)
4. **Cleanup Directory**: `cleanup-20251229-190056/` (615MB) - Temporary cleanup artifacts, safe to delete

## Other Cleanup Opportunities

### Docker Build Cache: **3.3GB** ✅ **HIGH PRIORITY**
```bash
sudo docker system prune -a  # Removes unused images, containers, build cache
```
**Potential Space Freed**: ~3.3GB (build cache)

### Large Directories
- `/home/ubuntu/.local`: 8.7GB (Python packages, pip cache)
- `/home/ubuntu/app`: 1.1GB
- `/home/ubuntu/backups`: 2.4GB

## Recommended Cleanup Strategy

### Phase 1: Safe Deletions (Immediate - ~1.2GB freed)

1. **Delete Empty Backup** (0 bytes)
   ```bash
   rm ~/backups/app-models-backup-.tar.gz
   ```

2. **Delete Cleanup Directories** (~615MB)
   ```bash
   rm -rf ~/backups/cleanup-20251229-190056/
   rm -rf ~/backups/cleanup_/
   ```

3. **Delete Duplicate Backup** (312KB)
   ```bash
   # Keep the one with timestamp, delete the generic one
   rm ~/backups/backup-.tar.gz
   ```

**Total Freed**: ~615MB

### Phase 2: Docker Cleanup (High Priority - ~3.3GB freed)

```bash
# Remove unused Docker build cache
sudo docker system prune -a --volumes

# Or more aggressive:
sudo docker builder prune -a -f
```

**Total Freed**: ~3.3GB

### Phase 3: Archive Old Backups (If Needed - ~1.8GB)

**Option A: Delete Old Backups** (if not needed)
```bash
# If backups are stored elsewhere (S3, etc.), delete:
rm ~/backups/burntbeats-full-backup-.tar.gz  # 1.5GB
rm ~/backups/phoenix-full-backup-.tar.gz      # 346MB
```

**Option B: Move to S3/External Storage**
```bash
# Upload to S3 before deleting
aws s3 cp ~/backups/burntbeats-full-backup-.tar.gz s3://your-bucket/backups/
aws s3 cp ~/backups/phoenix-full-backup-.tar.gz s3://your-bucket/backups/
# Then delete local copies
```

**Total Freed**: ~1.8GB (if deleted)

### Phase 4: Python Cache Cleanup (Optional - ~500MB-1GB)

```bash
# Clean pip cache
pip cache purge

# Clean Python bytecode
find /home/ubuntu -type d -name __pycache__ -exec rm -r {} + 2>/dev/null
find /home/ubuntu -name "*.pyc" -delete 2>/dev/null
```

## Total Potential Space Recovery

| Phase | Action | Space Freed |
|-------|--------|-------------|
| Phase 1 | Safe deletions | ~615MB |
| Phase 2 | Docker cleanup | **~3.3GB** |
| Phase 3 | Archive/delete old backups | ~1.8GB |
| Phase 4 | Python cache cleanup | ~500MB-1GB |
| **TOTAL** | | **~6-7GB** |

## Recommended Immediate Actions

### Priority 1: Docker Cleanup (Safest, Most Impact)
```bash
sudo docker system prune -a
```
**Frees**: ~3.3GB  
**Risk**: Low (only removes unused resources)

### Priority 2: Delete Cleanup Directories
```bash
rm -rf ~/backups/cleanup-20251229-190056/
rm -rf ~/backups/cleanup_/
rm ~/backups/app-models-backup-.tar.gz
rm ~/backups/backup-.tar.gz
```
**Frees**: ~615MB  
**Risk**: Low (cleanup artifacts and empty files)

### Priority 3: Archive/Delete Old Backups
**Decision Required**: Are these backups stored elsewhere?
- If YES (S3, etc.): Safe to delete local copies
- If NO: Should archive to S3 first, then delete

## Verification Commands

After cleanup, verify space freed:
```bash
df -h /
du -sh ~/backups
sudo docker system df
```

## Backup Recommendations

1. **Move backups to S3** instead of keeping on server
2. **Set up automated backup rotation** (keep only last N backups)
3. **Use incremental backups** instead of full backups
4. **Clean up after backup operations** (remove temporary files)

## Questions to Answer

1. Are the Dec 9 backups (`burntbeats-full-backup`, `phoenix-full-backup`) stored elsewhere?
2. Are these backups still needed, or can they be archived/deleted?
3. Should we set up S3 backup storage for future backups?
4. What's the backup retention policy?

---

**Next Steps**: Execute Phase 1 and Phase 2 cleanup to free ~4GB immediately.
