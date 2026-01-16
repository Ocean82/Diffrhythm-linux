#!/usr/bin/env python3
"""
Test loading from WSL native filesystem vs cross-filesystem
"""
import torch
import time
from pathlib import Path
import os

print("="*80)
print("Native Filesystem vs Cross-Filesystem Load Test")
print("="*80)

# Test 1: Load from WSL native filesystem
native_path = os.path.expanduser("~/test_models/cfm_model.pt")
print(f"\n[Test 1] Loading from WSL native filesystem...")
print(f"  Path: {native_path}")

if Path(native_path).exists():
    print(f"  Size: {Path(native_path).stat().st_size / (1024**3):.2f} GB")
    print("  Loading...")
    start = time.time()
    
    try:
        checkpoint = torch.load(native_path, map_location='cpu', weights_only=False)
        load_time = time.time() - start
        
        print(f"  ✓ SUCCESS! Loaded in {load_time:.2f}s")
        print(f"  Type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"  Keys: {list(checkpoint.keys())[:10]}")
            
            # Check for state dict
            if 'state_dict' in checkpoint:
                print(f"  state_dict keys: {len(checkpoint['state_dict'])}")
            elif 'model' in checkpoint:
                print(f"  model keys: {len(checkpoint['model'])}")
            else:
                print(f"  Total keys: {len(checkpoint)}")
        
    except Exception as e:
        load_time = time.time() - start
        print(f"  ✗ Failed after {load_time:.2f}s")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"  ✗ File not found")

# Test 2: Load from cross-filesystem (for comparison)
cross_path = "./pretrained/models--ASLP-lab--DiffRhythm-1_2/snapshots/185bdeb80541b9260d266c5f041859017441f307/cfm_model.pt"
print(f"\n[Test 2] Loading from cross-filesystem (/mnt/d/)...")
print(f"  Path: {cross_path}")
print("  (This will likely hang - timeout after 30s)")

if Path(cross_path).exists():
    print(f"  Size: {Path(cross_path).stat().st_size / (1024**3):.2f} GB")
    print("  Loading...")
    start = time.time()
    
    try:
        import signal
        
        # Set timeout
        def timeout_handler(signum, frame):
            raise TimeoutError("Load timed out after 30s")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        checkpoint = torch.load(cross_path, map_location='cpu', weights_only=False)
        signal.alarm(0)  # Cancel alarm
        
        load_time = time.time() - start
        print(f"  ✓ Loaded in {load_time:.2f}s")
        
    except TimeoutError as e:
        load_time = time.time() - start
        print(f"  ✗ Timed out after {load_time:.2f}s (as expected)")
    except Exception as e:
        load_time = time.time() - start
        print(f"  ✗ Failed after {load_time:.2f}s: {e}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\nIf Test 1 succeeded and Test 2 failed/timed out:")
print("  → Cross-filesystem I/O is the bottleneck")
print("  → Solution: Copy DiffRhythm to WSL native filesystem")
print("\nIf both failed:")
print("  → Deeper PyTorch/memory issue")
print("  → Try Docker or EC2 environment")
