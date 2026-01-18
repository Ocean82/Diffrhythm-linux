#!/usr/bin/env python3
"""Debug import issues"""
import sys
import os

print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("Python path:", sys.path[:3])

print("\n[1/5] Testing basic imports...")
try:
    import torch
    print(f"  ✓ torch {torch.__version__}")
except Exception as e:
    print(f"  ✗ torch failed: {e}")
    sys.exit(1)

print("\n[2/5] Adding current directory to path...")
sys.path.append(os.getcwd())
print(f"  ✓ Added {os.getcwd()}")

print("\n[3/5] Testing infer module import...")
try:
    import infer
    print(f"  ✓ infer module found at: {infer.__file__}")
except Exception as e:
    print(f"  ✗ infer import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n[4/5] Testing infer.infer_utils import...")
try:
    from infer import infer_utils
    print(f"  ✓ infer_utils found")
except Exception as e:
    print(f"  ✗ infer_utils import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n[5/5] Testing prepare_model import...")
try:
    from infer.infer_utils import prepare_model
    print(f"  ✓ prepare_model imported successfully")
    print("\n✓ All imports successful!")
except Exception as e:
    print(f"  ✗ prepare_model import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
