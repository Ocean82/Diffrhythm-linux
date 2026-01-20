#!/usr/bin/env python3
"""
Test LoRA Integration and Quality Improvements
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_lora_adapter_exists():
    """Check if LoRA adapter is available"""
    lora_path = PROJECT_ROOT / "ckpts" / "lora" / "lora_adapter_epoch_1"
    print("Checking LoRA adapter...")
    
    if lora_path.exists():
        print(f"OK LoRA adapter found at: {lora_path}")
        
        # Check adapter files
        adapter_config = lora_path / "adapter_config.json"
        adapter_model = lora_path / "adapter_model.safetensors"
        
        if adapter_config.exists() and adapter_model.exists():
            print("OK Adapter files are valid")
            
            # Read config
            import json
            with open(adapter_config, 'r') as f:
                config = json.load(f)
            
            print(f"  LoRA rank (r): {config.get('r', 'N/A')}")
            print(f"  LoRA alpha: {config.get('lora_alpha', 'N/A')}")
            print(f"  Target modules: {', '.join(config.get('target_modules', []))}")
            
            return True
        else:
            print("ERROR Adapter files missing")
            return False
    else:
        print("ERROR LoRA adapter not found")
        return False


def test_peft_import():
    """Test if PEFT library can be imported"""
    print("\nChecking PEFT library...")
    try:
        import peft
        print(f"OK PEFT library installed (version {peft.__version__})")
        return True
    except ImportError:
        print("ERROR PEFT library not installed")
        print("  Install with: pip install peft")
        return False


def test_infer_with_lora():
    """Test infer.py with LoRA option"""
    print("\nTesting infer.py with LoRA option...")
    
    # Check if infer.py has --lora-path option
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "infer.infer", "--help"],
            capture_output=True,
            text=True
        )
        
        if "--lora-path" in result.stdout:
            print("OK infer.py supports --lora-path option")
        else:
            print("ERROR infer.py does not support --lora-path option")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error checking infer.py: {e}")
        return False


def test_generate_high_quality_lora():
    """Test generate_high_quality.py with LoRA option"""
    print("\nTesting generate_high_quality.py with LoRA option...")
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "generate_high_quality.py", "--help"],
            capture_output=True,
            text=True
        )
        
        if "--lora-path" in result.stdout:
            print("OK generate_high_quality.py supports --lora-path option")
        else:
            print("ERROR generate_high_quality.py does not support --lora-path option")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error checking generate_high_quality.py: {e}")
        return False


def test_quality_improvements():
    """Test quality improvements are available"""
    print("\nTesting quality improvements...")
    
    # Test quality presets
    try:
        from infer.quality_presets import get_preset, print_presets
        presets = ["preview", "draft", "standard", "high", "maximum", "ultra"]
        
        print("OK Quality presets available:")
        for preset in presets:
            cfg = get_preset(preset)
            print(f"  - {preset:10} (steps: {cfg.steps}, CFG: {cfg.cfg_strength})")
            
    except Exception as e:
        print(f"✗ Error with quality presets: {e}")
        return False
        
    # Test prompt builder
    try:
        from infer.prompt_builder import StylePromptBuilder
        builder = StylePromptBuilder()
        
        # Test random prompt
        prompt = builder.random_prompt()
        print(f"\nOK Prompt builder working: {prompt[:60]}...")
        
    except Exception as e:
        print(f"✗ Error with prompt builder: {e}")
        return False
        
    # Test mastering
    try:
        from post_processing.mastering import master_audio_file
        print("OK Mastering module available")
        
    except Exception as e:
        print(f"✗ Error with mastering: {e}")
        return False
        
    return True


def test_train_script():
    """Test training script can verify setup"""
    print("\nTesting training script...")
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "train/train_lora.py", "--help"],
            capture_output=True,
            text=True
        )
        
        if "--verify-only" in result.stdout:
            print("OK Training script supports --verify-only mode")
        else:
            print("ERROR Training script does not support --verify-only mode")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error checking training script: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("  DiffRhythm Quality & LoRA Integration Test")
    print("=" * 70)
    
    all_passed = True
    
    # Run tests
    tests = [
        ("LoRA Adapter", test_lora_adapter_exists),
        ("PEFT Library", test_peft_import),
        ("infer.py LoRA", test_infer_with_lora),
        ("generate_high_quality.py LoRA", test_generate_high_quality_lora),
        ("Quality Improvements", test_quality_improvements),
        ("Training Script", test_train_script),
    ]
    
    for name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"Testing: {name}")
            print('='*60)
            
            passed = test_func()
            all_passed = all_passed and passed
            
            if passed:
                print(f"OK {name} test PASSED")
            else:
                print(f"ERROR {name} test FAILED")
                
        except Exception as e:
            print(f"ERROR {name} test failed with exception: {e}")
            import traceback
            print(traceback.format_exc())
            all_passed = False
            
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("\nTo use LoRA for generation:")
        print("  python -m infer.infer --lrc-path lyrics.lrc --ref-prompt \"style\" --preset high --lora-path ckpts/lora/lora_adapter_epoch_1")
        print("\nOr with high-quality pipeline:")
        print("  python generate_high_quality.py --lyrics lyrics.lrc --genre Pop --mood Upbeat --lora-path ckpts/lora/lora_adapter_epoch_1")
    else:
        print("SOME TESTS FAILED - see details above")
        
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
