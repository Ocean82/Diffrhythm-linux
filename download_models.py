#!/usr/bin/env python3
"""
Download DiffRhythm models at Docker runtime
"""
import os
from huggingface_hub import hf_hub_download

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
                cache_dir="./pretrained",
                force_download=False
            )
            print(f"✓ Downloaded {filename}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
    
    print("Model download complete!")

if __name__ == "__main__":
    download_models()