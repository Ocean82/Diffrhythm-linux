#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

def fix_symlinks():
    pretrained_dir = Path("pretrained")
    
    # Known symlink locations
    symlink_dirs = [
        "models--OpenMuQ--MuQ-MuLan-large/snapshots/2e01c796b71dca71b45251384c04cd7b237c9020",
        "models--xlm-roberta-base/snapshots/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
    ]
    
    for symlink_dir in symlink_dirs:
        dir_path = pretrained_dir / symlink_dir
        if not dir_path.exists():
            continue
            
        print(f"Processing: {symlink_dir}")
        
        for item in dir_path.iterdir():
            if item.is_symlink():
                try:
                    target = item.resolve()
                    if target.exists():
                        print(f"  Resolving: {item.name}")
                        item.unlink()
                        shutil.copy2(target, item)
                        print(f"  [OK] Fixed: {item.name}")
                    else:
                        print(f"  [ERROR] Target missing: {item.name}")
                except Exception as e:
                    print(f"  [ERROR] Error with {item.name}: {e}")

if __name__ == "__main__":
    fix_symlinks()
    print("Done!")