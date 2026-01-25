#!/usr/bin/env python3
"""
Test model loading to verify all models can be loaded successfully
"""
import os
import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_model_loading():
    """Test loading all models"""
    print("=" * 80)
    print("Model Loading Test")
    print("=" * 80)
    print()
    
    device = os.getenv("DEVICE", "cpu")
    print(f"Device: {device}")
    print()
    
    # Test imports
    print("Step 1: Testing imports...")
    try:
        from infer.infer_utils import prepare_model
        print("  [OK] prepare_model imported successfully")
    except Exception as e:
        print(f"  [FAIL] Failed to import prepare_model: {e}")
        traceback.print_exc()
        return False
    
    # Test model loading
    print("\nStep 2: Loading models (this may take several minutes)...")
    print("  This will download models from HuggingFace if not cached")
    print()
    
    try:
        max_frames = 2048  # For 95s songs
        print(f"  Loading models with max_frames={max_frames}...")
        
        cfm_model, tokenizer, muq_model, vae_model = prepare_model(
            max_frames=max_frames,
            device=device
        )
        
        print("  [OK] All models loaded successfully!")
        print()
        
        # Verify models
        print("Step 3: Verifying models...")
        
        if cfm_model is not None:
            print(f"  [OK] CFM model loaded: {type(cfm_model).__name__}")
        else:
            print("  [FAIL] CFM model is None")
            return False
        
        if tokenizer is not None:
            print(f"  [OK] Tokenizer loaded: {type(tokenizer).__name__}")
        else:
            print("  [FAIL] Tokenizer is None")
            return False
        
        if muq_model is not None:
            print(f"  [OK] MuQ model loaded: {type(muq_model).__name__}")
        else:
            print("  [FAIL] MuQ model is None")
            return False
        
        if vae_model is not None:
            print(f"  [OK] VAE model loaded: {type(vae_model).__name__}")
        else:
            print("  [FAIL] VAE model is None")
            return False
        
        print()
        print("=" * 80)
        print("[SUCCESS] All models loaded and verified successfully!")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"  [FAIL] Failed to load models: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
