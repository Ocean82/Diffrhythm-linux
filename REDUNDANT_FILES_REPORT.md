# Redundant Files Investigation Report

## Summary
Investigated the server for redundant files, old virtual environments, and unnecessary artifacts.

## Findings

### ✅ Good News
1. **No Virtual Environments Found** ✓
   - No `.venv`, `venv`, `env`, or `ENV` directories
   - Since we're using Docker for deployment, virtual environments are not needed on the server
   - This is correct - all dependencies are installed in Docker containers

2. **No Python Cache Files** ✓
   - No `__pycache__/` directories
   - No `.pyc` or `.pyo` files
   - Already cleaned in previous cleanup

3. **No Temporary/Backup Files** ✓
   - No `.bak`, `.tmp`, `.swp`, `.swo`, `~`, or `.DS_Store` files

4. **No Log Files** ✓
   - No `.log` files found in project directory

5. **No Build Artifacts** ✓
   - No `dist/`, `build/`, or `*.egg-info` directories

6. **No Duplicate Files** ✓
   - No obvious duplicate Python files

### ⚠️ Opportunities for Cleanup

1. **Docker Build Cache: 132.7MB** ⚠️
   - **Status**: Can be reclaimed
   - **Action**: Run `sudo docker builder prune -af` to free this space
   - **Impact**: Will free ~133MB of disk space

2. **Large File: `g2p/sources/chinese_lexicon.txt` (15MB)** ✓
   - **Status**: **KEEP** - Required for phonemizer functionality
   - **Action**: Do not remove

3. **Output/Temp Directories** ✓
   - **Status**: Empty (4KB each)
   - **Action**: No action needed

## Disk Usage
- **Project Directory**: 16MB (clean)
- **System Disk**: 83% usage (8.4GB free)
- **Docker Cache**: 132.7MB (reclaimable)

## Recommended Actions

### Immediate Cleanup
```bash
# Clean Docker build cache
sudo docker builder prune -af

# This will free ~133MB of space
```

### Verification
After cleanup, verify:
```bash
sudo docker system df
df -h /
```

## Conclusion
The server is already quite clean! The main opportunity is the Docker build cache which can free ~133MB. No virtual environments or other redundant files were found, which is correct for a Docker-based deployment.

## Files to Keep
- `g2p/` directory (15MB) - Required for phonemizer
- All production code and configuration files
- Model files in `pretrained/`

## Cleanup Script
Use `scripts/find_redundant_files.sh` to periodically check for redundant files.
