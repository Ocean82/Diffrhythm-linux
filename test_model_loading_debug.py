#!/usr/bin/env python3
"""Debug model loading"""
import sys
import os
import time

sys.path.append(os.getcwd())

print("="*80)
print("DiffRhythm Model Loading Debug")
print("="*80)

print("\n[1/3] Importing prepare_model...")
try:
    from infer.infer_utils import prepare_model
    print("  ✓ Import successful")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[2/3] Calling prepare_model...")
print("  This may take several minutes...")
print("  Device: cpu")
print("  Max frames: 2048")

start_time = time.time()

try:
    print("\n  Starting model loading...")
    cfm, tokenizer, muq, vae = prepare_model(max_frames=2048, device='cpu')
    
    load_time = time.time() - start_time
    print(f"\n  ✓ Models loaded successfully in {load_time:.2f} seconds")
    
    print("\n[3/3] Verifying models...")
    print(f"  CFM model: {type(cfm)}")
    print(f"  Tokenizer: {type(tokenizer)}")
    print(f"  MuQ model: {type(muq)}")
    print(f"  VAE model: {type(vae)}")
    
    print("\n✓ All models loaded and verified!")
    
except KeyboardInterrupt:
    print("\n\n✗ Interrupted by user")
    sys.exit(1)
except Exception as e:
    load_time = time.time() - start_time
    print(f"\n✗ Model loading failed after {load_time:.2f} seconds")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
