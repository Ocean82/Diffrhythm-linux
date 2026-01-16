#!/usr/bin/env python3
"""
Test loading model with weights_only=True to skip metadata
"""
import torch
import time
from pathlib import Path

print("="*80)
print("Testing weights_only Load")
print("="*80)

cfm_path = "./pretrained/models--ASLP-lab--DiffRhythm-1_2/snapshots/185bdeb80541b9260d266c5f041859017441f307/cfm_model.pt"

if not Path(cfm_path).exists():
    print(f"Model not found: {cfm_path}")
    exit(1)

print(f"\nModel: {cfm_path}")
print(f"Size: {Path(cfm_path).stat().st_size / (1024**3):.2f} GB")

# Test 1: weights_only=True
print("\n[Test 1] Loading with weights_only=True...")
start = time.time()
try:
    checkpoint = torch.load(cfm_path, map_location='cpu', weights_only=True)
    load_time = time.time() - start
    print(f"  ✓ Loaded in {load_time:.2f}s")
    print(f"  Type: {type(checkpoint)}")
    if isinstance(checkpoint, dict):
        print(f"  Keys: {list(checkpoint.keys())[:10]}")
except Exception as e:
    load_time = time.time() - start
    print(f"  ✗ Failed after {load_time:.2f}s: {e}")

# Test 2: Load with mmap
print("\n[Test 2] Loading with mmap_mode...")
start = time.time()
try:
    checkpoint = torch.load(cfm_path, map_location='cpu', weights_only=False, mmap=True)
    load_time = time.time() - start
    print(f"  ✓ Loaded in {load_time:.2f}s")
    print(f"  Type: {type(checkpoint)}")
except Exception as e:
    load_time = time.time() - start
    print(f"  ✗ Failed after {load_time:.2f}s: {e}")

# Test 3: Load state dict only
print("\n[Test 3] Attempting to extract state_dict...")
start = time.time()
try:
    # Try loading with weights_only first
    checkpoint = torch.load(cfm_path, map_location='cpu', weights_only=True)
    
    if isinstance(checkpoint, dict):
        # Look for common state dict keys
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"  ✓ Found state_dict with {len(state_dict)} keys")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print(f"  ✓ Found model with {len(state_dict)} keys")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"  ✓ Found model_state_dict with {len(state_dict)} keys")
        else:
            # Checkpoint itself might be the state dict
            state_dict = checkpoint
            print(f"  ✓ Checkpoint is state_dict with {len(state_dict)} keys")
        
        # Show some keys
        keys = list(state_dict.keys())
        print(f"  Sample keys: {keys[:5]}")
        
    load_time = time.time() - start
    print(f"  Total time: {load_time:.2f}s")
    
except Exception as e:
    load_time = time.time() - start
    print(f"  ✗ Failed after {load_time:.2f}s: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
