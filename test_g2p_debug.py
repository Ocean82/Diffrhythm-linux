#!/usr/bin/env python3
"""Debug g2p/phonemizer issues"""
import sys
import os
import time

sys.path.append(os.getcwd())

print("="*80)
print("G2P/Phonemizer Debug Test")
print("="*80)

print("\n[1/4] Testing g2p module import...")
start = time.time()
try:
    import g2p
    print(f"  ✓ g2p module imported in {time.time()-start:.2f}s")
    print(f"  Location: {g2p.__file__ if hasattr(g2p, '__file__') else 'built-in'}")
except Exception as e:
    print(f"  ✗ g2p import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[2/4] Testing g2p.g2p_generation import...")
start = time.time()
try:
    from g2p import g2p_generation
    print(f"  ✓ g2p_generation imported in {time.time()-start:.2f}s")
except Exception as e:
    print(f"  ✗ g2p_generation import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[3/4] Testing chn_eng_g2p import...")
start = time.time()
try:
    from g2p.g2p_generation import chn_eng_g2p
    print(f"  ✓ chn_eng_g2p imported in {time.time()-start:.2f}s")
except Exception as e:
    print(f"  ✗ chn_eng_g2p import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[4/4] Testing chn_eng_g2p function call...")
start = time.time()
try:
    test_text = "Hello world"
    result = chn_eng_g2p(test_text)
    print(f"  ✓ chn_eng_g2p executed in {time.time()-start:.2f}s")
    print(f"  Result type: {type(result)}")
    print(f"  Result: {result}")
except Exception as e:
    print(f"  ✗ chn_eng_g2p execution failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All g2p tests passed!")
