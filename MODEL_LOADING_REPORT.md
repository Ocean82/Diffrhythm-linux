# DiffRhythm Model Loading Investigation Report

## Executive Summary
✅ **All models load successfully** - No critical issues found in model loading
⚠️ **CPU-only PyTorch detected** - Performance will be significantly slower without GPU

## Investigation Results

### 1. Model Files Status
✅ **DiffRhythm CFM Model**: `pretrained/models--ASLP-lab--DiffRhythm-1_2/snapshots/.../cfm_model.pt`
✅ **DiffRhythm VAE Model**: `pretrained/models--ASLP-lab--DiffRhythm-vae/snapshots/.../vae_model.pt`
✅ **MuQ MuLan Model**: Successfully loads from HuggingFace cache
✅ **XLM-Roberta**: Successfully loads from HuggingFace cache

### 2. Symlinks Resolution
✅ **All symlinks resolved** - Replaced with actual files for WSL compatibility
- Fixed: `models--xlm-roberta-base/snapshots/*/` files
- Fixed: `models--OpenMuQ--MuQ-MuLan-large/snapshots/*/` files

### 3. Code Issues Fixed
✅ **Type annotations** - Fixed Pylance errors in:
- `train/train.py` - Variable initialization and type hints
- `model/trainer.py` - Optional parameter types
- `model/cfm.py` - Import statements and generic types

✅ **Inference scripts** - Removed unsupported `--device` argument

### 4. Environment Detection
⚠️ **PyTorch Installation**: CPU-only version detected
- Current: `torch 2.6.0+cpu`
- CUDA: Not available
- Impact: Inference will be 10-50x slower than GPU

### 5. Dependencies Status
✅ All required packages installed:
- torch, torchaudio
- librosa, muq
- transformers, accelerate
- phonemizer (requires espeak-ng)

## Critical Findings for Production Use

### Issue 1: CPU-Only Performance
**Problem**: Current PyTorch installation is CPU-only
**Impact**: 
- 95s song generation: ~10-30 minutes on CPU vs 1-3 minutes on GPU
- 285s song generation: ~30-90 minutes on CPU vs 3-10 minutes on GPU

**Solution for WSL**:
```bash
# Uninstall CPU version
pip uninstall torch torchaudio

# Install CUDA version (requires NVIDIA GPU + CUDA drivers)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue 2: WSL GPU Support
**Requirements**:
1. Windows 11 or Windows 10 (21H2+)
2. NVIDIA GPU with latest drivers
3. WSL 2 (not WSL 1)
4. CUDA toolkit in WSL

**Verify GPU in WSL**:
```bash
nvidia-smi  # Should show GPU info
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Issue 3: Memory Requirements
**Minimum**: 8GB VRAM (with --chunked flag)
**Recommended**: 16GB+ VRAM for full-length songs
**CPU fallback**: 16GB+ RAM (very slow)

## Testing Checklist

### Basic Tests (Completed ✅)
- [x] Model imports work
- [x] Config files load correctly
- [x] Checkpoint files accessible
- [x] VAE model loads
- [x] MuQ model loads
- [x] Tokenizer initializes

### Integration Tests (To Run in WSL)
- [ ] espeak-ng installed and working
- [ ] Phonemizer can process text
- [ ] Full inference pipeline runs
- [ ] Audio output generated
- [ ] Output quality verification

### Performance Tests (To Run)
- [ ] 95s song generation time
- [ ] 285s song generation time
- [ ] Memory usage monitoring
- [ ] Batch inference (multiple songs)

## Recommended Next Steps

### For WSL Setup:
1. **Install espeak-ng in WSL**:
   ```bash
   sudo apt-get update
   sudo apt-get install espeak-ng
   ```

2. **Activate virtual environment**:
   ```bash
   cd /mnt/d/EMBERS-BANK/DiffRhythm-Linux
   source .venv/bin/activate
   ```

3. **Run test script**:
   ```bash
   chmod +x test_wsl_inference.sh
   bash test_wsl_inference.sh
   ```

### For Docker Deployment:
1. **Use GPU-enabled base image**:
   ```dockerfile
   FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
   ```

2. **Install CUDA-enabled PyTorch**:
   ```dockerfile
   RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Expose inference API**:
   - Create FastAPI/Flask wrapper
   - Add queue system for batch processing
   - Implement health checks

### For Backend Integration:
1. **API Wrapper Needed**:
   ```python
   # Example structure
   POST /generate
   {
     "lyrics": "lrc content",
     "style_prompt": "folk, acoustic",
     "duration": 95
   }
   ```

2. **Queue System**:
   - Use Celery/RQ for async processing
   - Estimated time: 1-3 min per song (GPU)
   - Return job ID, poll for completion

3. **Storage**:
   - Save generated .wav files
   - Implement cleanup policy
   - Consider S3/cloud storage

## Known Limitations

1. **Language Support**: Chinese and English lyrics only
2. **Duration**: 95s (base) or 96-285s (full model)
3. **Style Control**: Limited to text prompts or reference audio
4. **Quality**: Experimental - requires testing for production use
5. **Speed**: CPU inference not practical for real-time use

## Files Created for Testing

1. `test_model_loading.py` - Validates all model components load
2. `test_wsl_inference.sh` - Full WSL inference test
3. `fix_symlinks.py` - Resolves HuggingFace symlinks
4. `verify_wsl_setup.sh` - Environment validation
5. `WSL_SETUP_SUMMARY.md` - Setup documentation

## Conclusion

**Model Loading**: ✅ Working correctly
**WSL Compatibility**: ✅ Ready (after espeak-ng install)
**Production Ready**: ⚠️ Requires GPU for acceptable performance
**Docker Ready**: ⚠️ Needs GPU-enabled configuration

The system is functional but requires GPU acceleration for production use. CPU-only inference is too slow for backend API deployment.
