#!/usr/bin/env python3
"""Test full model loading with timing"""
import sys
import os
import time

sys.path.append(os.getcwd())

print("="*80)
print("Full DiffRhythm Model Loading Test")
print("="*80)

total_start = time.time()

print("\n[1/4] Importing prepare_model...")
start = time.time()
from infer.infer_utils import prepare_model
print(f"  ✓ Imported in {time.time()-start:.2f}s")

print("\n[2/4] Loading CFM model...")
print("  This will download from HuggingFace if not cached...")
start = time.time()

print("\n[3/4] Loading tokenizer (includes g2p - will take ~90s)...")
print("  Note: Chinese g2p model is very slow to load...")

print("\n[4/4] Calling prepare_model...")
print("  Device: cpu")
print("  Max frames: 2048")

try:
    cfm, tokenizer, muq, vae = prepare_model(max_frames=2048, device='cpu')
    
    total_time = time.time() - total_start
    print(f"\n✓ All models loaded successfully!")
    print(f"  Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    
    print("\nModel types:")
    print(f"  CFM: {type(cfm)}")
    print(f"  Tokenizer: {type(tokenizer)}")
    print(f"  MuQ: {type(muq)}")
    print(f"  VAE: {type(vae)}")
    
except Exception as e:
    total_time = time.time() - total_start
    print(f"\n✗ Failed after {total_time:.2f}s")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
