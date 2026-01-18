#!/usr/bin/env python3
"""
Quick debug to identify the normalization issue
"""
import os
import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

def test_normalization_issue():
    """Test the specific normalization issue in infer.py"""
    print("="*60)
    print("Testing Normalization Issue")
    print("="*60)
    
    # Test case 1: Normal audio (should work)
    print("\n[Test 1] Normal audio normalization:")
    normal_audio = torch.randn(2, 44100) * 0.5  # Normal audio with reasonable amplitude
    max_val = torch.max(torch.abs(normal_audio))
    print(f"   Input max amplitude: {max_val:.6f}")
    
    try:
        normalized = (
            normal_audio.to(torch.float32)
            .div(torch.max(torch.abs(normal_audio)))
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
        )
        print(f"   ✓ Normalization successful")
        print(f"   Output range: {normalized.min()} to {normalized.max()}")
    except Exception as e:
        print(f"   ✗ Normalization failed: {e}")
    
    # Test case 2: All zeros (will fail)
    print("\n[Test 2] All-zeros audio normalization:")
    zero_audio = torch.zeros(2, 44100)
    max_val = torch.max(torch.abs(zero_audio))
    print(f"   Input max amplitude: {max_val:.6f}")
    
    try:
        normalized = (
            zero_audio.to(torch.float32)
            .div(torch.max(torch.abs(zero_audio)))  # Division by zero!
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
        )
        print(f"   Output range: {normalized.min()} to {normalized.max()}")
        has_nan = torch.isnan(normalized).any()
        has_inf = torch.isinf(normalized).any()
        print(f"   Contains NaN: {has_nan}, Contains Inf: {has_inf}")
        if has_nan or has_inf:
            print(f"   ✗ Normalization produced NaN/Inf values!")
        else:
            print(f"   ✓ Normalization completed (but likely wrong)")
    except Exception as e:
        print(f"   ✗ Normalization failed: {e}")
    
    # Test case 3: Very small values (will fail)
    print("\n[Test 3] Very small amplitude audio normalization:")
    tiny_audio = torch.randn(2, 44100) * 1e-10  # Extremely small values
    max_val = torch.max(torch.abs(tiny_audio))
    print(f"   Input max amplitude: {max_val:.2e}")
    
    try:
        normalized = (
            tiny_audio.to(torch.float32)
            .div(torch.max(torch.abs(tiny_audio)))
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
        )
        print(f"   Output range: {normalized.min()} to {normalized.max()}")
        has_nan = torch.isnan(normalized).any()
        has_inf = torch.isinf(normalized).any()
        print(f"   Contains NaN: {has_nan}, Contains Inf: {has_inf}")
        if has_nan or has_inf:
            print(f"   ✗ Normalization produced NaN/Inf values!")
        else:
            print(f"   ✓ Normalization completed")
    except Exception as e:
        print(f"   ✗ Normalization failed: {e}")
    
    # Test case 4: Fixed normalization approach
    print("\n[Test 4] Fixed normalization approach:")
    
    def safe_normalize(audio, target_amplitude=0.95):
        """Safe normalization that handles edge cases"""
        max_val = torch.max(torch.abs(audio))
        
        if max_val < 1e-8:  # Essentially silent
            print(f"   ⚠ Audio is silent (max={max_val:.2e}), returning zeros")
            return torch.zeros_like(audio, dtype=torch.int16)
        
        # Normalize to target amplitude
        normalized = audio * (target_amplitude / max_val)
        
        # Convert to int16
        normalized = normalized.clamp(-1, 1).mul(32767).to(torch.int16)
        
        return normalized
    
    # Test on all cases
    test_cases = [
        ("Normal", normal_audio),
        ("Zeros", zero_audio), 
        ("Tiny", tiny_audio)
    ]
    
    for name, audio in test_cases:
        print(f"   Testing {name} audio with safe normalization:")
        try:
            result = safe_normalize(audio)
            print(f"     ✓ Success - range: {result.min()} to {result.max()}")
        except Exception as e:
            print(f"     ✗ Failed: {e}")
    
    print("\n" + "="*60)
    print("DIAGNOSIS:")
    print("="*60)
    print("The issue is in infer/infer.py lines 92-95:")
    print("  .div(torch.max(torch.abs(output)))")
    print("")
    print("This fails when:")
    print("1. Output is all zeros (division by zero)")
    print("2. Output has very small values (division by near-zero)")
    print("3. CFM/VAE produces silent or corrupted audio")
    print("")
    print("SOLUTION: Add validation before normalization")
    
    return True

if __name__ == "__main__":
    test_normalization_issue()