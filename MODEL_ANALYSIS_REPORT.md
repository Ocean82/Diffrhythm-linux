# Model Files Analysis Report

**Date:** 2026-01-24  
**Server:** ubuntu@52.0.207.242  
**Location:** `/opt/diffrhythm/pretrained`

## Required Models for DiffRhythm

Based on code analysis (`infer/infer_utils.py`), the following models are **REQUIRED**:

### 1. CFM Model (Conditional Flow Matching)
- **Repository:** `ASLP-lab/DiffRhythm-1_2` (for 95s songs, max_frames=2048)
- **Repository (Full):** `ASLP-lab/DiffRhythm-1_2-full` (for 96-285s songs, max_frames=6144)
- **File:** `cfm_model.pt`
- **Size:** ~2GB each
- **Usage:** Main generation model - **REQUIRED**
- **Note:** Only one is needed based on use case (95s vs longer songs)

### 2. VAE Model (Variational Autoencoder)
- **Repository:** `ASLP-lab/DiffRhythm-vae`
- **File:** `vae_model.pt`
- **Size:** ~500MB-1GB
- **Usage:** Audio encoding/decoding - **REQUIRED**

### 3. MuQ-MuLan Model
- **Repository:** `OpenMuQ/MuQ-MuLan-large`
- **Files:** Multiple (loaded via `from_pretrained`)
- **Size:** ~500MB-1GB
- **Usage:** Style embedding extraction - **REQUIRED**

## Potentially Unnecessary Models

### 1. OpenMuQ/MuQ-large-msd-iter
- **Status:** **NOT USED** in current codebase
- **Can Remove:** ✅ YES (if not needed for other purposes)
- **Size:** Check actual size

### 2. xlm-roberta-base
- **Status:** **NOT USED** in current codebase
- **Can Remove:** ✅ YES (if not needed for other purposes)
- **Size:** Check actual size

### 3. DiffRhythm-1_2-full (if only using 95s songs)
- **Status:** Only needed for songs >95 seconds
- **Can Remove:** ⚠️ CONDITIONAL
  - Remove if only generating 95-second songs
  - Keep if generating longer songs (96-285s)

## Current Status on Server

**Location:** `/opt/diffrhythm/pretrained`

**Findings:**
- Model directories exist but are mostly empty (4KB each)
- Actual model files (`.pt` files) have **NOT been downloaded yet**
- Total pretrained directory size: Very small (<200KB)
- Models will be downloaded automatically on first API run

**This means:**
- ✅ No significant disk space is currently used by models
- ⚠️ Models will need to be downloaded when API starts (~4-5GB total)
- ⚠️ This will happen during Docker build or first API startup

## Recommendations

### Immediate Action (Before Docker Build)
Since models aren't downloaded yet, we can safely remove unused model directories:

1. **Remove unused models:**
   - `MuQ-large-msd-iter` - NOT USED in codebase
   - `xlm-roberta-base` - NOT USED in codebase
   - **Space saved:** Minimal (directories only, ~40KB)

2. **Optional - Remove DiffRhythm-1_2-full if only using 95s songs:**
   - Only needed for songs >95 seconds
   - **Space saved:** Will prevent ~2GB download later

### After Models Are Downloaded
Once models are downloaded, you can:
1. **If only generating 95-second songs:**
   - Keep: `DiffRhythm-1_2`, `DiffRhythm-vae`, `MuQ-MuLan-large`
   - Can remove: `DiffRhythm-1_2-full` (~2GB)
   - **Potential space saved:** ~2GB

2. **If generating both 95s and longer songs:**
   - Keep: All DiffRhythm models, `MuQ-MuLan-large`
   - Can remove: `MuQ-large-msd-iter`, `xlm-roberta-base`
   - **Potential space saved:** ~500MB-1GB

## Safe Removal Script

See `scripts/safe_remove_unused_models.sh` for safe removal of unused models.
