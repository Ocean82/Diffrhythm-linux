#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.getcwd())

print("Testing model loading...")
print(f"Python path: {sys.path}")
print(f"Current dir: {os.getcwd()}")

try:
    print("\n1. Testing torch import...")
    import torch
    print(f"   [OK] Torch version: {torch.__version__}")
    print(f"   [OK] CUDA available: {torch.cuda.is_available()}")
    
    print("\n2. Testing model imports...")
    from model import DiT, CFM
    print("   [OK] Model imports successful")
    
    print("\n3. Testing config loading...")
    import json
    with open("./config/diffrhythm-1b.json") as f:
        model_config = json.load(f)
    print(f"   [OK] Config loaded: {model_config['model_type']}")
    
    print("\n4. Testing model initialization...")
    max_frames = 2048
    cfm = CFM(
        transformer=DiT(**model_config["model"], max_frames=max_frames),
        num_channels=model_config["model"]["mel_dim"],
        max_frames=max_frames
    )
    print(f"   [OK] Model initialized")
    
    print("\n5. Testing checkpoint loading...")
    from huggingface_hub import hf_hub_download
    dit_ckpt_path = hf_hub_download(
        repo_id="ASLP-lab/DiffRhythm-1_2", 
        filename="cfm_model.pt", 
        cache_dir="./pretrained"
    )
    print(f"   [OK] Checkpoint path: {dit_ckpt_path}")
    
    print("\n6. Loading checkpoint weights...")
    checkpoint = torch.load(dit_ckpt_path, weights_only=True, map_location="cpu")
    print(f"   [OK] Checkpoint keys: {list(checkpoint.keys())}")
    
    print("\n7. Testing VAE loading...")
    vae_ckpt_path = hf_hub_download(
        repo_id="ASLP-lab/DiffRhythm-vae",
        filename="vae_model.pt",
        cache_dir="./pretrained",
    )
    print(f"   [OK] VAE path: {vae_ckpt_path}")
    vae = torch.jit.load(vae_ckpt_path, map_location="cpu")
    print(f"   [OK] VAE loaded successfully")
    
    print("\n8. Testing MuQ loading...")
    from muq import MuQMuLan
    muq = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir="./pretrained")
    print(f"   [OK] MuQ loaded successfully")
    
    print("\n[OK] ALL TESTS PASSED!")
    
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
