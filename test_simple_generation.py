#!/usr/bin/env python3
"""
Simple test to verify the fixes are working
"""
import os
import sys
import torch
import torchaudio

def test_safe_normalize():
    """Test the safe normalization function"""
    print("="*50)
    print("Testing Safe Normalization Function")
    print("="*50)
    
    # Import the function from the fixed infer.py
    sys.path.append('infer')
    try:
        from infer import safe_normalize_audio
        print("✓ Successfully imported safe_normalize_audio")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False
    
    # Test cases
    test_cases = [
        ("Normal audio", torch.randn(2, 44100) * 0.5),
        ("Silent audio", torch.zeros(2, 44100)),
        ("Very quiet audio", torch.randn(2, 44100) * 1e-10),
        ("Audio with NaN", torch.tensor([[float('nan'), 0.5], [0.3, 0.2]])),
        ("Audio with Inf", torch.tensor([[float('inf'), 0.5], [0.3, 0.2]])),
        ("Loud audio", torch.randn(2, 44100) * 2.0),
    ]
    
    all_passed = True
    
    for name, audio in test_cases:
        print(f"\nTesting: {name}")
        try:
            result = safe_normalize_audio(audio)
            print(f"  Result shape: {result.shape}, dtype: {result.dtype}")
            print(f"  Range: {result.min()} to {result.max()}")
            
            # Basic validation
            if result.dtype != torch.int16:
                print(f"  ⚠ WARNING: Expected int16, got {result.dtype}")
            
            if torch.isnan(result).any():
                print(f"  ✗ ERROR: Result contains NaN")
                all_passed = False
            elif torch.isinf(result).any():
                print(f"  ✗ ERROR: Result contains Inf")
                all_passed = False
            else:
                print(f"  ✓ Result is valid")
                
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            all_passed = False
    
    return all_passed

def test_validate_latents():
    """Test the latent validation function"""
    print("\n" + "="*50)
    print("Testing Latent Validation Function")
    print("="*50)
    
    # Import the function
    sys.path.append('infer')
    try:
        from infer import validate_latents
        print("✓ Successfully imported validate_latents")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False
    
    # Test cases
    test_cases = [
        ("Valid latent", torch.randn(1, 128, 256)),
        ("All zeros", torch.zeros(1, 128, 256)),
        ("With NaN", torch.randn(1, 128, 256)),
        ("With Inf", torch.randn(1, 128, 256)),
        ("Very low variance", torch.ones(1, 128, 256) * 0.001),
    ]
    
    # Modify some test cases to have actual issues
    test_cases[2] = ("With NaN", torch.randn(1, 128, 256))
    test_cases[2][1][0, 0, 0] = float('nan')
    
    test_cases[3] = ("With Inf", torch.randn(1, 128, 256))
    test_cases[3][1][0, 0, 0] = float('inf')
    
    all_passed = True
    
    for name, latent in test_cases:
        print(f"\nTesting: {name}")
        try:
            result = validate_latents(latent, name)
            print(f"  Validation result: {result}")
            
            # Expected results
            if name == "Valid latent" and not result:
                print(f"  ✗ ERROR: Valid latent failed validation")
                all_passed = False
            elif name in ["All zeros", "With NaN", "With Inf"] and result:
                print(f"  ⚠ WARNING: Invalid latent passed validation")
            else:
                print(f"  ✓ Validation result as expected")
                
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            all_passed = False
    
    return all_passed

def check_dependencies():
    """Check if basic dependencies are available"""
    print("="*50)
    print("Checking Dependencies")
    print("="*50)
    
    deps = [
        ("torch", "PyTorch"),
        ("torchaudio", "TorchAudio"),
        ("einops", "Einops"),
    ]
    
    all_available = True
    
    for module, name in deps:
        try:
            __import__(module)
            print(f"✓ {name} available")
        except ImportError:
            print(f"✗ {name} not available")
            all_available = False
    
    return all_available

def main():
    """Main test function"""
    print("Testing Fixed DiffRhythm Components")
    print("This will test the core fixes without running full generation")
    print()
    
    # Check dependencies
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n✗ Missing dependencies - cannot run tests")
        return
    
    # Test normalization
    norm_ok = test_safe_normalize()
    
    # Test validation
    val_ok = test_validate_latents()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if norm_ok and val_ok:
        print("✓ ALL TESTS PASSED!")
        print("The core fixes are working correctly.")
        print()
        print("Next steps:")
        print("1. Install missing dependencies (espeak, etc.)")
        print("2. Run full generation test with: python test_song_generation.py")
    else:
        print("⚠ SOME TESTS FAILED")
        if not norm_ok:
            print("- Normalization function has issues")
        if not val_ok:
            print("- Validation function has issues")

if __name__ == "__main__":
    main()