# Server Cleanup Summary

## Files/Directories to Remove

### 1. Python Cache Files
- `__pycache__/` directories (12 found)
- `*.pyc` files (39 found)
- `*.pyo` files
- **Estimated space**: ~224KB

### 2. Development/Investigation Documentation
The following markdown files will be removed (keeping only essential docs):
- `AUDIO_SAVING_TROUBLESHOOTING.md`
- `BACKEND_CONNECTION_SUMMARY.md`
- `CODEC_AND_FORMAT_TROUBLESHOOTING.md`
- `CODEC_COMPATIBILITY_FIX_SUMMARY.md`
- `CODEC_COMPATIBILITY_INVESTIGATION.md`
- `CODEC_INVESTIGATION_INDEX.md`
- `CODEC_VALIDATION_SUMMARY.md`
- `CODE_CHANGES_REFERENCE.md`
- `CPU_DEPLOYMENT_ANALYSIS.md`
- `CPU_DEPLOYMENT_GUIDE.md`
- `DEPLOYMENT_READY.md`
- `FINAL_INVESTIGATION_REPORT.md`
- `FIXES_APPLIED_SUMMARY.md`
- `FRONTEND_BACKEND_CONNECTION.md`
- `FRONTEND_INTEGRATION.md`
- `GETTING_STARTED_WITH_AUDIO_FIX.md`
- `HANG_UP_FIXES_APPLIED.md`
- `HANG_UP_FIXES_INDEX.md`
- `IMPLEMENTATION_COMPLETE.md`
- `IMPLEMENTATION_STATUS.md`
- `INVESTIGATION_COMPLETE.md`
- `INVESTIGATION_SUMMARY.md`
- `MODEL_LOADING_REPORT.md`
- `ODE_STALL_FIX_SUMMARY.md`
- `OUTPUT_BREAKDOWN_ANALYSIS.md`
- `OUTPUT_QUALITY_CONFIGURATION.md`
- `PRODUCTION_READINESS_SUMMARY.md`
- `QUALITY_IMPROVEMENTS_GUIDE.md`
- `FIXES_VISUAL_SUMMARY.txt`
- **Estimated space**: ~200KB

**Keeping**: `DEPLOYMENT.md`, `README.md`, `Readme.md`, `LICENSE`, `LICENSE.md`

### 3. Training/Development Directories
- `ckpts/` - Model checkpoints (8.1MB)
- `dataset/` - Training dataset (5.2MB)
- `train/` - Training code (4KB)
- **Total**: ~13.3MB

### 4. IDE/Development Directories
- `.continue/` - Continue IDE (12KB)
- `.cursor/` - Cursor IDE (8KB)
- `.refact/` - Refact IDE (20KB)
- `.github/` - GitHub workflows (12KB)
- **Total**: ~52KB

### 5. Old Docker Files
- `Dockerfile` - Old Dockerfile (4KB)
- `docker/` - Old docker directory (16KB)
- `docker-compose.yml` - Old compose file (4KB)
- **Total**: ~24KB

### 6. Third-party Code
- `thirdparty/` - Will be installed via pip (12KB)

### 7. Log Files
- `*.log` files throughout the project

## Total Estimated Space to Free
- **Minimum**: ~13.6MB (directories + files)
- **With logs and cache**: Potentially more

## Large Files Found (Keep)
- `g2p/sources/chinese_lexicon.txt` (>10MB) - **KEEP** (needed for phonemizer)

## Script Usage

### Dry Run (Preview)
```bash
bash scripts/cleanup_server.sh --dry-run
```

### Actual Cleanup
```bash
bash scripts/cleanup_server.sh
```

## Safety
- Script preserves essential documentation
- Keeps production Docker files (`Dockerfile.prod`, `docker-compose.prod.yml`)
- Keeps essential scripts
- Does not remove model files or pretrained models
- Does not remove `g2p` directory (needed for phonemizer)

## After Cleanup
- Server will have cleaner structure
- More disk space available for Docker builds
- Only production-necessary files remain
