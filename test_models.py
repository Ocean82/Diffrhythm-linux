#!/usr/bin/env python3
"""
Simple test script to verify DiffRhythm models load correctly
"""
import os
import sys
import torch

# Add current directory to path
sys.path.append(os.getcwd())

def test_model_loading():
    """Test if all models can be loaded successfully"""
    print("Testing DiffRhythm Model Loading...")
    print("=" * 50)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from infer.infer_utils import prepare_model
        print("   [OK] Imports successful")
        
        # Test model loading
        print("2. Testing model loading...")
        device = "cpu"  # Force CPU for compatibility
        max_frames = 2048  # Test with base model first
        
        print(f"   Loading models (max_frames={max_frames}, device={device})...")
        cfm, tokenizer, muq, vae = prepare_model(max_frames, device)
        
        print("   [OK] CFM model loaded")
        print("   [OK] Tokenizer loaded") 
        print("   [OK] MuQ model loaded")
        print("   [OK] VAE model loaded")
        
        # Test model properties
        print("3. Testing model properties...")
        print(f"   CFM device: {next(cfm.parameters()).device}")
        print(f"   CFM dtype: {next(cfm.parameters()).dtype}")
        print(f"   MuQ device: {next(muq.parameters()).device}")
        print(f"   VAE device: {vae.device if hasattr(vae, 'device') else 'JIT model'}")
        
        # Test tokenizer
        print("4. Testing tokenizer...")
        test_text = "Hello world"
        tokens = tokenizer.encode(test_text)
        print(f"   Text: '{test_text}' -> Tokens: {tokens[:10]}...")
        
        # Test MuQ with text prompt
        print("5. Testing MuQ with text prompt...")
        with torch.no_grad():
            style_emb = muq(texts="folk, acoustic guitar")
            print(f"   Style embedding shape: {style_emb.shape}")
        
        print("\n[SUCCESS] ALL TESTS PASSED!")
        print("Your DiffRhythm models are ready for use!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)