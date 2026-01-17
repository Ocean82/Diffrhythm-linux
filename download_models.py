#!/usr/bin/env python3
"""
Download DiffRhythm models at Docker runtime
"""
import os
from huggingface_hub import hf_hub_download

from huggingface_hub import hf_hub_download

# Determine the default cache directory
# Priority: HUGGINGFACE_HUB_CACHE -> HF_HOME -> /mnt/d/_hugging-face (WSL) -> D:\_hugging-face (Windows) -> ./pretrained
DEFAULT_CACHE_DIR = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HOME")
if not DEFAULT_CACHE_DIR:
    if os.name == 'nt': # Windows
        DEFAULT_CACHE_DIR = "D:\\_hugging-face"
    else: # Linux/WSL
        if os.path.exists("/mnt/d"):
            DEFAULT_CACHE_DIR = "/mnt/d/_hugging-face"
        else:
            DEFAULT_CACHE_DIR = "./pretrained"

print(f"DEBUG: Using cache directory: {DEFAULT_CACHE_DIR}", flush=True)

def download_models():
    """Download all required models"""
    print("Downloading DiffRhythm models...")
    
    models = [
        ("ASLP-lab/DiffRhythm-1_2", "cfm_model.pt"),
        ("ASLP-lab/DiffRhythm-1_2-full", "cfm_model.pt"), 
        ("ASLP-lab/DiffRhythm-vae", "vae_model.pt"),
        ("OpenMuQ/MuQ-MuLan-large", "pytorch_model.bin"),
        ("OpenMuQ/MuQ-MuLan-large", "config.json"),
        ("xlm-roberta-base", "model.safetensors"),
        ("xlm-roberta-base", "config.json"),
        ("xlm-roberta-base", "tokenizer.json"),
    ]
    
    for repo_id, filename in models:
        try:
            print(f"Downloading {repo_id}/{filename}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=DEFAULT_CACHE_DIR,
                force_download=False
            )
            print(f"✓ Downloaded {filename}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
    
    print("Model download complete!")

if __name__ == "__main__":
    download_models()