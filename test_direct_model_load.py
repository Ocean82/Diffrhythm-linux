#!/usr/bin/env python3
"""
Test loading models directly from cached files without HuggingFace Hub
"""
import sys
import os
import time
import torch
import json
from pathlib import Path

sys.path.append(os.getcwd())

print("="*80)
print("Direct Model Load Test (No HuggingFace Hub)")
print("="*80)

try:
    # Load models directly from pretrained cache
    print("\n[1/4] Loading CFM model from cache...")
    start = time.time()
    
    from model import DiT, CFM
    
    # Direct path to cached model
    cfm_path = "./pretrained/models--ASLP-lab--DiffRhythm-1_2/snapshots/185bdeb80541b9260d266c5f041859017441f307/cfm_model.pt"
    
    if not Path(cfm_path).exists():
        print(f"  ✗ Model not found at: {cfm_path}")
        sys.exit(1)
    
    print(f"  Found model at: {cfm_path}")
    print(f"  Size: {Path(cfm_path).stat().st_size / (1024**3):.2f} GB")
    
    # Load config
    dit_config_path = "./config/diffrhythm-1b.json"
    with open(dit_config_path) as f:
        model_config = json.load(f)
    
    device = "cpu"
    max_frames = 2048
    
    print("  Creating CFM model structure...")
    cfm = CFM(
        transformer=DiT(**model_config["model"], max_frames=max_frames),
        num_channels=model_config["model"]["mel_dim"],
        max_frames=max_frames
    )
    cfm = cfm.to(device)
    
    print("  Loading checkpoint...")
    from infer.infer_utils import load_checkpoint
    cfm = load_checkpoint(cfm, cfm_path, device=device, use_ema=False)
    
    print(f"  ✓ CFM loaded in {time.time()-start:.2f}s")
    
    # Load VAE directly
    print("\n[2/4] Loading VAE model from cache...")
    start = time.time()
    
    vae_path = "./pretrained/models--ASLP-lab--DiffRhythm-vae/snapshots/74e2afacfd91dd1b96662c96dcef763c1258768b/vae_model.pt"
    
    if not Path(vae_path).exists():
        print(f"  ✗ VAE not found at: {vae_path}")
        sys.exit(1)
    
    print(f"  Found VAE at: {vae_path}")
    print(f"  Size: {Path(vae_path).stat().st_size / (1024**3):.2f} GB")
    print("  Loading...")
    
    vae = torch.jit.load(vae_path, map_location="cpu").to(device)
    
    print(f"  ✓ VAE loaded in {time.time()-start:.2f}s")
    
    # Load MuQ
    print("\n[3/4] Loading MuQ model...")
    start = time.time()
    
    from muq import MuQMuLan
    muq = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir="./pretrained")
    muq = muq.to(device).eval()
    
    print(f"  ✓ MuQ loaded in {time.time()-start:.2f}s")
    
    # Load tokenizer
    print("\n[4/4] Loading tokenizer...")
    print("  Note: This will take ~90s due to Chinese g2p model")
    start = time.time()
    
    from infer.infer_utils import CNENTokenizer
    tokenizer = CNENTokenizer()
    
    print(f"  ✓ Tokenizer loaded in {time.time()-start:.2f}s")
    
    print("\n" + "="*80)
    print("✓ ALL MODELS LOADED SUCCESSFULLY!")
    print("="*80)
    print("\nReady for generation!")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
