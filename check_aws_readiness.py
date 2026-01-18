#!/usr/bin/env python3
"""
Check if DiffRhythm is ready for AWS deployment
"""
import os
from pathlib import Path

def check_symlinks():
    """Check for any remaining symlinks"""
    pretrained_path = Path("pretrained")
    symlinks = []
    
    for root, dirs, files in os.walk(pretrained_path):
        for file in files:
            file_path = Path(root) / file
            if file_path.is_symlink():
                symlinks.append(file_path)
    
    return symlinks

def check_model_files():
    """Check if all required model files exist and are actual files"""
    required_models = {
        "DiffRhythm-1_2": "pretrained/models--ASLP-lab--DiffRhythm-1_2/snapshots/*/cfm_model.pt",
        "DiffRhythm-vae": "pretrained/models--ASLP-lab--DiffRhythm-vae/snapshots/*/vae_model.pt", 
        "xlm-roberta": "pretrained/models--xlm-roberta-base/snapshots/*/model.safetensors",
        "MuQ-MuLan": "pretrained/models--OpenMuQ--MuQ-MuLan-large/snapshots/*/pytorch_model.bin"
    }
    
    results = {}
    pretrained_path = Path("pretrained")
    
    for name, pattern in required_models.items():
        # Convert glob pattern to actual search
        if "DiffRhythm-1_2" in pattern:
            matches = list(pretrained_path.glob("models--ASLP-lab--DiffRhythm-1_2/snapshots/*/cfm_model.pt"))
        elif "DiffRhythm-vae" in pattern:
            matches = list(pretrained_path.glob("models--ASLP-lab--DiffRhythm-vae/snapshots/*/vae_model.pt"))
        elif "xlm-roberta" in pattern:
            matches = list(pretrained_path.glob("models--xlm-roberta-base/snapshots/*/model.safetensors"))
        elif "MuQ-MuLan" in pattern:
            matches = list(pretrained_path.glob("models--OpenMuQ--MuQ-MuLan-large/snapshots/*/pytorch_model.bin"))
        else:
            matches = []
            
        if matches:
            file_path = matches[0]
            if file_path.exists() and not file_path.is_symlink():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                results[name] = {"status": "OK", "size_mb": size_mb, "path": str(file_path)}
            else:
                results[name] = {"status": "SYMLINK" if file_path.is_symlink() else "MISSING", "size_mb": 0, "path": str(file_path)}
        else:
            results[name] = {"status": "NOT_FOUND", "size_mb": 0, "path": "N/A"}
    
    return results

def main():
    print("=== DiffRhythm AWS Deployment Readiness Check ===")
    print()
    
    # Check for symlinks
    symlinks = check_symlinks()
    if symlinks:
        print(f"❌ Found {len(symlinks)} symlinks that need resolution:")
        for symlink in symlinks:
            print(f"   - {symlink}")
        print()
    else:
        print("✅ No symlinks found - good for AWS deployment!")
        print()
    
    # Check model files
    print("Model Files Status:")
    print("-" * 50)
    models = check_model_files()
    
    all_good = True
    for name, info in models.items():
        status_icon = "✅" if info["status"] == "OK" else "❌"
        print(f"{status_icon} {name}: {info['status']}")
        if info["status"] == "OK":
            print(f"   Size: {info['size_mb']:.1f} MB")
            print(f"   Path: {info['path']}")
        elif info["status"] == "SYMLINK":
            print(f"   ⚠️  This is a symlink - needs resolution")
            all_good = False
        elif info["status"] in ["MISSING", "NOT_FOUND"]:
            print(f"   ⚠️  File not found")
            all_good = False
        print()
    
    # Summary
    print("=" * 50)
    if all_good and not symlinks:
        print("🎉 READY FOR AWS DEPLOYMENT!")
        print("All models are actual files (no symlinks)")
    else:
        print("⚠️  NEEDS ATTENTION:")
        if symlinks:
            print("- Run symlink resolution script")
        if not all_good:
            print("- Some model files are missing or symlinked")
        print("\nRun: python3 run_symlink_fix.py")

if __name__ == "__main__":
    main()