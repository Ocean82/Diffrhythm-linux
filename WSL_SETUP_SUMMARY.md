# DiffRhythm WSL Setup Summary

## ✅ Completed Tasks

### 1. Symlinks Resolved
- ✅ Fixed all HuggingFace model symlinks in `pretrained/` directory
- ✅ Replaced symlinks with actual files for AWS/WSL compatibility
- ✅ Verified model files are now proper files (not symlinks)

### 2. File Structure Verified
- ✅ All required directories present: config, infer, model, pretrained, scripts, g2p
- ✅ Training data available in dataset/ (latent, lrc, style folders)
- ✅ Example files present for testing
- ✅ Scripts are executable and properly configured

### 3. WSL-Specific Configurations
- ✅ Updated inference script to use GPU (cuda) instead of CPU
- ✅ Scripts include proper PYTHONPATH export
- ✅ MacOS-specific phonemizer paths handled in scripts

## 📋 WSL Setup Checklist

### Prerequisites (Run in WSL)
```bash
# Install espeak-ng
sudo apt-get update
sudo apt-get install espeak-ng

# Verify installation
espeak-ng --version
```

### Python Environment Setup
```bash
# Navigate to project directory
cd /mnt/d/EMBERS-BANK/DiffRhythm-Linux

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Verify Setup
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Test inference (prompt-based)
bash scripts/infer_prompt_ref.sh

# Test inference (audio reference)
bash scripts/infer_wav_ref.sh
```

## 🎯 Key Files Ready for WSL

### Model Files (No longer symlinks)
- `pretrained/models--ASLP-lab--DiffRhythm-1_2/snapshots/*/cfm_model.pt`
- `pretrained/models--ASLP-lab--DiffRhythm-vae/snapshots/*/vae_model.pt`
- `pretrained/models--xlm-roberta-base/snapshots/*/model.safetensors`
- `pretrained/models--OpenMuQ--MuQ-MuLan-large/snapshots/*/pytorch_model.bin`

### Inference Scripts
- `scripts/infer_prompt_ref.sh` - Text prompt-based generation
- `scripts/infer_wav_ref.sh` - Audio reference-based generation

### Example Files
- `infer/example/eg_cn_full.lrc` - Chinese lyrics example
- `infer/example/eg_en_full.lrc` - English lyrics example
- `infer/example/eg_en.mp3` - Audio reference example

## ⚠️ Important Notes

1. **VRAM Requirements**: DiffRhythm-base needs minimum 8GB VRAM
2. **Chunked Processing**: Use `--chunked` flag for lower VRAM usage
3. **Output Directory**: Results saved to `infer/example/output/`
4. **GPU Usage**: Scripts now configured for CUDA (change to `cpu` if needed)

## 🚀 Ready to Run!

Your DiffRhythm setup is now properly configured for WSL deployment with all symlinks resolved and files ready for use.