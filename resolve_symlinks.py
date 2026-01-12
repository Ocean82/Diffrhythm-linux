#!/usr/bin/env python3
"""
Script to resolve HuggingFace symlinks to actual files for AWS deployment
"""
import os
import shutil
import sys
from pathlib import Path

def resolve_symlinks(pretrained_dir):
    """Replace all symlinks with actual files"""
    pretrained_path = Path(pretrained_dir)
    
    if not pretrained_path.exists():
        print(f"Error: {pretrained_dir} does not exist")
        return False
    
    print(f"Resolving symlinks in: {pretrained_path}")
    
    # Find all symlinks
    symlinks_found = []
    for root, dirs, files in os.walk(pretrained_path):
        for file in files:
            file_path = Path(root) / file
            if file_path.is_symlink():
                symlinks_found.append(file_path)
    
    print(f"Found {len(symlinks_found)} symlinks to resolve")
    
    # Resolve each symlink
    for symlink_path in symlinks_found:
        try:
            # Get the target file
            target_path = symlink_path.resolve()
            
            if not target_path.exists():
                print(f"Warning: Target does not exist for {symlink_path}")
                continue
            
            # Get file size for verification
            size_mb = target_path.stat().st_size / (1024 * 1024)
            
            print(f"Resolving: {symlink_path.name} ({size_mb:.1f} MB)")
            
            # Remove symlink and copy actual file
            symlink_path.unlink()
            shutil.copy2(target_path, symlink_path)
            
            print(f"[OK] Resolved: {symlink_path.name}")
            
        except Exception as e:
            print(f"Error resolving {symlink_path}: {e}")
            return False
    
    print(f"\n[SUCCESS] Successfully resolved {len(symlinks_found)} symlinks")
    return True

def verify_models(pretrained_dir):
    """Verify all required models are present and have correct sizes"""
    pretrained_path = Path(pretrained_dir)
    
    expected_models = {
        "models--ASLP-lab--DiffRhythm-1_2/snapshots/*/cfm_model.pt": (2000, 2500),  # 2.0-2.5 GB
        "models--ASLP-lab--DiffRhythm-1_2-full/snapshots/*/cfm_model.pt": (2000, 2500),  # 2.0-2.5 GB
        "models--ASLP-lab--DiffRhythm-vae/snapshots/*/vae_model.pt": (600, 700),  # 600-700 MB
        "models--OpenMuQ--MuQ-MuLan-large/snapshots/*/pytorch_model.bin": (2500, 3000),  # 2.5-3.0 GB
        "models--xlm-roberta-base/snapshots/*/model.safetensors": (1000, 1200),  # 1.0-1.2 GB
    }
    
    print("\nVerifying model files...")
    all_good = True
    
    for pattern, (min_mb, max_mb) in expected_models.items():
        matches = list(pretrained_path.glob(pattern))
        
        if not matches:
            print(f"[ERROR] Missing: {pattern}")
            all_good = False
            continue
        
        for model_path in matches:
            if not model_path.exists():
                print(f"[ERROR] File not found: {model_path}")
                all_good = False
                continue
            
            size_mb = model_path.stat().st_size / (1024 * 1024)
            
            if min_mb <= size_mb <= max_mb:
                print(f"[OK] {model_path.name}: {size_mb:.1f} MB (OK)")
            else:
                print(f"[WARNING] {model_path.name}: {size_mb:.1f} MB (Expected: {min_mb}-{max_mb} MB)")
                if size_mb < 10:  # Likely incomplete download
                    all_good = False
    
    return all_good

if __name__ == "__main__":
    pretrained_dir = "pretrained"
    
    if len(sys.argv) > 1:
        pretrained_dir = sys.argv[1]
    
    print("DiffRhythm Model Symlink Resolver")
    print("=" * 50)
    
    # Resolve symlinks
    if resolve_symlinks(pretrained_dir):
        # Verify models
        if verify_models(pretrained_dir):
            print("\n[SUCCESS] All models are ready for AWS deployment!")
        else:
            print("\n[WARNING] Some models may need re-downloading")
    else:
        print("\n[ERROR] Failed to resolve symlinks")
        sys.exit(1)