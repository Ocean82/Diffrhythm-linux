#!/usr/bin/env python3
"""Test basic torch.load"""
import torch
import time
from pathlib import Path

print("Testing torch.load on CFM model...")

cfm_path = "./pretrained/models--ASLP-lab--DiffRhythm-1_2/snapshots/185bdeb80541b9260d266c5f041859017441f307/cfm_model.pt"

if not Path(cfm_path).exists():
    print(f"Model not found: {cfm_path}")
    exit(1)

print(f"Model path: {cfm_path}")
print(f"Model size: {Path(cfm_path).stat().st_size / (1024**3):.2f} GB")

print("\nLoading with torch.load (map_location='cpu')...")
start = time.time()

try:
    checkpoint = torch.load(cfm_path, map_location='cpu', weights_only=False)
    load_time = time.time() - start
    
    print(f"✓ Loaded in {load_time:.2f}s")
    print(f"Checkpoint type: {type(checkpoint)}")
    if isinstance(checkpoint, dict):
        print(f"Keys: {list(checkpoint.keys())[:10]}")
    
except Exception as e:
    load_time = time.time() - start
    print(f"✗ Failed after {load_time:.2f}s")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
