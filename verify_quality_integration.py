#!/usr/bin/env python3
"""
Verify Quality Improvements Integration

This script verifies that all quality improvements have been
properly integrated into the DiffRhythm project.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_file_exists(path, description):
    """Check if a file exists"""
    full_path = PROJECT_ROOT / path
    exists = full_path.exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {description}: {path}")
    return exists

def check_import(module_path, description):
    """Check if a module can be imported"""
    try:
        parts = module_path.split(".")
        if len(parts) > 1:
            exec(f"from {'.'.join(parts[:-1])} import {parts[-1]}")
        else:
            exec(f"import {module_path}")
        print(f"  ✓ {description}: {module_path}")
        return True
    except ImportError as e:
        print(f"  ✗ {description}: {module_path} - {e}")
        return False
    except Exception as e:
        print(f"  ⚠ {description}: {module_path} - {e}")
        return False

def check_integration_in_file(file_path, search_terms, description):
    """Check if specific code is integrated in a file"""
    full_path = PROJECT_ROOT / file_path
    if not full_path.exists():
        print(f"  ✗ {description}: File not found - {file_path}")
        return False

    content = full_path.read_text()
    all_found = True
    for term in search_terms:
        if term in content:
            print(f"  ✓ {description}: Found '{term[:40]}...'")
        else:
            print(f"  ✗ {description}: Missing '{term[:40]}...'")
            all_found = False
    return all_found

def main():
    print("\n" + "="*70)
    print(" DiffRhythm Quality Integration Verification")
    print("="*70)

    all_passed = True

    # 1. Check created files exist
    print("\n[1/5] Checking created files...")
    files_to_check = [
        ("infer/prompt_builder.py", "Prompt Builder"),
        ("infer/quality_presets.py", "Quality Presets"),
        ("post_processing/__init__.py", "Post-processing module"),
        ("post_processing/enhance.py", "Audio Enhancement"),
        ("post_processing/mastering.py", "Audio Mastering"),
        ("train/train_lora.py", "LoRA Training Script"),
        ("generate_high_quality.py", "High-Quality Generation Script"),
        ("diagnose_ode_stall.py", "ODE Diagnostic Tool"),
    ]

    for path, desc in files_to_check:
        if not check_file_exists(path, desc):
            all_passed = False

    # 2. Check imports work
    print("\n[2/5] Checking module imports...")
    imports_to_check = [
        ("infer.prompt_builder", "Prompt Builder"),
        ("infer.quality_presets", "Quality Presets"),
        ("post_processing.enhance", "Enhancement Module"),
        ("post_processing.mastering", "Mastering Module"),
    ]

    for module, desc in imports_to_check:
        if not check_import(module, desc):
            all_passed = False

    # 3. Check integration in infer.py
    print("\n[3/5] Checking infer.py integration...")
    infer_integrations = [
        ("--preset", "Preset argument"),
        ("--auto-master", "Auto-master argument"),
        ("from infer.quality_presets import get_preset", "Quality presets import"),
        ("from post_processing.mastering import master_audio_file", "Mastering import"),
    ]

    for term, desc in infer_integrations:
        if not check_integration_in_file("infer/infer.py", [term], desc):
            all_passed = False

    # 4. Check training data
    print("\n[4/5] Checking training data...")
    training_files = [
        ("dataset/train.scp", "Training manifest"),
        ("dataset/latent/2626046476.pt", "Sample latent"),
        ("dataset/lrc/2626046476.pt", "Sample lyrics"),
        ("dataset/style/2626046476.pt", "Sample style"),
    ]

    for path, desc in training_files:
        if not check_file_exists(path, desc):
            all_passed = False

    # Count training samples
    train_scp = PROJECT_ROOT / "dataset/train.scp"
    if train_scp.exists():
        num_samples = len(train_scp.read_text().strip().split("\n"))
        print(f"  ✓ Training samples available: {num_samples}")

    # 5. Check ODE improvements in cfm.py
    print("\n[5/5] Checking ODE improvements in model/cfm.py...")
    ode_improvements = [
        ("ODEProgressTracker", "Progress tracker class"),
        ("ODETimeoutError", "Timeout error class"),
        ("_manual_euler_integration", "Manual Euler integration"),
        ("set_ode_timeout", "Timeout setter method"),
    ]

    for term, desc in ode_improvements:
        if not check_integration_in_file("model/cfm.py", [term], desc):
            all_passed = False

    # Summary
    print("\n" + "="*70)
    if all_passed:
        print(" ✓ ALL QUALITY IMPROVEMENTS VERIFIED!")
        print("="*70)
        print("\nYou can now use:")
        print("  1. Quality presets:    python -m infer.infer --preset high ...")
        print("  2. Auto-mastering:     python -m infer.infer --auto-master ...")
        print("  3. Prompt builder:     python infer/prompt_builder.py --interactive")
        print("  4. High-quality gen:   python generate_high_quality.py --lyrics song.lrc --genre Pop")
        print("  5. LoRA training:      python train/train_lora.py --verify-only")
    else:
        print(" ⚠ SOME CHECKS FAILED - See above for details")
        print("="*70)

    # Check if PEFT is available for LoRA
    print("\n[EXTRA] Checking LoRA dependencies...")
    try:
        import peft
        print(f"  ✓ PEFT library installed (version {peft.__version__})")
    except ImportError:
        print("  ⚠ PEFT library NOT installed")
        print("    Install with: pip install peft")
        print("    Required for LoRA fine-tuning")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
