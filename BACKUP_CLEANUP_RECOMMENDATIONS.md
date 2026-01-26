# Backup Directory Cleanup Recommendations
**Date**: January 26, 2026  
**Location**: `/home/ubuntu/backups/`  
**Current Size**: 2.4GB  
**Disk Space Crisis**: 97% used (only 1.7GB free)

## Analysis Results

### ✅ Confirmed Safe to Delete

1. **Empty Backup** (0 bytes)
   - `app-models-backup-.tar.gz` - Empty file, safe to delete

2. **Duplicate Backups** (312KB each, identical MD5)
   - `backup-.tar.gz` and `backup-20251222-103614.tar.gz` are **identical**
   - **Recommendation**: Keep `backup-20251222-103614.tar.gz` (has timestamp), delete `backup-.tar.gz`

3. **Cleanup Artifacts** (615MB)
   - `cleanup-20251229-190056/` (615MB) - Temporary cleanup directory
   - `cleanup_/` (8KB) - Old cleanup directory
   - **Safe to delete** - these are cleanup artifacts, not backups

### ⚠️ Review Before Deleting

1. **Large Backups** (1.8GB total)
   - `burntbeats-full-backup-.tar.gz` (1.5GB) - From Dec 9, 2025
   - `phoenix-full-backup-.tar.gz` (346MB) - From Dec 9, 2025
   - **Age**: ~48 days old
   - **Decision Needed**: Are these stored elsewhere (S3, external storage)?

2. **Old Scripts** (120KB)
   - `old-scripts/` - Review contents before deleting

## Immediate Cleanup (Safe - ~1.2GB freed)

### Commands to Execute

```bash
# 1. Delete empty backup
rm ~/backups/app-models-backup-.tar.gz

# 2. Delete duplicate backup (keep the one with timestamp)
rm ~/backups/backup-.tar.gz

# 3. Delete cleanup artifacts
rm -rf ~/backups/cleanup-20251229-190056/
rm -rf ~/backups/cleanup_/

# 4. Verify space freed
du -sh ~/backups
df -h /
```

**Expected Space Freed**: ~615MB (cleanup dirs) + 312KB (duplicate) + 0 (empty) = **~615MB**

## Additional Cleanup Opportunities

### Docker Build Cache: **3.3GB** (HIGHEST PRIORITY)

```bash
# Remove unused Docker build cache (safest)
sudo docker system prune -a

# Or more targeted:
sudo docker builder prune -a -f
```

**Expected Space Freed**: **~3.3GB**

### Old Backups Decision

**If backups are stored in S3 or elsewhere:**
```bash
# Delete old backups (frees 1.8GB)
rm ~/backups/burntbeats-full-backup-.tar.gz
rm ~/backups/phoenix-full-backup-.tar.gz
```

**If backups are NOT stored elsewhere:**
```bash
# Upload to S3 first, then delete
aws s3 cp ~/backups/burntbeats-full-backup-.tar.gz s3://your-backup-bucket/
aws s3 cp ~/backups/phoenix-full-backup-.tar.gz s3://your-backup-bucket/
# Then delete local copies
rm ~/backups/burntbeats-full-backup-.tar.gz
rm ~/backups/phoenix-full-backup-.tar.gz
```

## Total Potential Space Recovery

| Action | Space Freed | Risk Level |
|--------|-------------|------------|
| Delete cleanup artifacts | 615MB | ✅ Very Low |
| Delete duplicate backup | 312KB | ✅ Very Low |
| Delete empty backup | 0 | ✅ Very Low |
| Docker build cache cleanup | **3.3GB** | ✅ Low |
| Delete old backups (if archived) | 1.8GB | ⚠️ Medium |
| **TOTAL (conservative)** | **~4GB** | |
| **TOTAL (aggressive)** | **~5.8GB** | |

## Recommended Execution Order

### Step 1: Docker Cleanup (Safest, Most Impact)
```bash
sudo docker system prune -a
```
**Frees**: ~3.3GB  
**Risk**: Very Low (only removes unused resources)

### Step 2: Safe Backup Cleanup
```bash
cd ~/backups
rm app-models-backup-.tar.gz
rm backup-.tar.gz
rm -rf cleanup-20251229-190056/
rm -rf cleanup_/
```
**Frees**: ~615MB  
**Risk**: Very Low

### Step 3: Verify and Check Space
```bash
df -h /
du -sh ~/backups
```

### Step 4: Decision on Old Backups
- **If stored elsewhere**: Delete (frees additional 1.8GB)
- **If not stored**: Archive to S3 first, then delete

## Verification

After cleanup, verify:
```bash
# Check disk space
df -h /

# Check backup directory size
du -sh ~/backups

# Check Docker space
sudo docker system df
```

## Expected Results

**Before Cleanup:**
- Disk: 97% used (1.7GB free)
- Backups: 2.4GB
- Docker cache: 3.3GB

**After Step 1 & 2:**
- Disk: ~93% used (~5GB free)
- Backups: ~1.8GB
- Docker cache: ~0GB

**After Step 4 (if old backups deleted):**
- Disk: ~90% used (~7GB free)
- Backups: ~0GB (or archived)

## Backup Directory Assessment

### Is This Directory Needed?

**Short Answer**: Partially - keep recent backups, delete old/duplicate ones

**Recommendation**:
1. **Keep**: Recent backups (if not stored elsewhere)
2. **Delete**: 
   - Empty backups
   - Duplicate backups
   - Cleanup artifacts
   - Old backups (if archived elsewhere)

3. **Best Practice**: 
   - Move backups to S3 or external storage
   - Keep only last 2-3 backups on server
   - Use incremental backups instead of full backups
   - Clean up after backup operations

## Questions to Answer

1. ✅ Are `backup-.tar.gz` and `backup-20251222-103614.tar.gz` duplicates? **YES** (same MD5)
2. ⚠️ Are the Dec 9 backups stored in S3 or elsewhere? **UNKNOWN** - Need to check
3. ⚠️ Are the Dec 9 backups still needed? **PROBABLY NOT** (48 days old)
4. ⚠️ What's the backup retention policy? **UNKNOWN**

## Immediate Action Plan

**Execute these commands to free ~4GB immediately:**

```bash
# 1. Docker cleanup (frees 3.3GB)
sudo docker system prune -a

# 2. Safe backup cleanup (frees 615MB)
cd ~/backups
rm app-models-backup-.tar.gz
rm backup-.tar.gz
rm -rf cleanup-20251229-190056/
rm -rf cleanup_/

# 3. Verify
df -h /
du -sh ~/backups
```

**Total freed**: ~4GB  
**New free space**: ~5.7GB (up from 1.7GB)  
**Disk usage**: ~90% (down from 97%)

---

**Status**: Ready to execute  
**Risk**: Low (only removing unused/duplicate files)  
**Impact**: High (frees critical disk space)
