#!/usr/bin/env python3
"""
Fix remaining symlinks for AWS deployment
"""
import shutil
from pathlib import Path

def fix_symlinks():
    print("Fixing remaining symlinks for AWS deployment...")
    
    # Fix sentencepiece.bpe.model symlink
    symlink_path = Path("pretrained/models--xlm-roberta-base/snapshots/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089/sentencepiece.bpe.model")
    
    if symlink_path.is_symlink():
        print(f"Fixing symlink: {symlink_path}")
        
        # Find the target file
        target_path = symlink_path.resolve()
        
        if target_path.exists():
            # Remove symlink and copy actual file
            symlink_path.unlink()
            shutil.copy2(target_path, symlink_path)
            print(f"✅ Fixed: {symlink_path.name}")
        else:
            print(f"❌ Target not found: {target_path}")
    else:
        print(f"✅ Already fixed: {symlink_path.name}")

def verify_no_symlinks():
    """Verify no symlinks remain"""
    pretrained_path = Path("pretrained")
    symlinks = []
    
    for root, dirs, files in pretrained_path.rglob("*"):
        if root.is_symlink():
            symlinks.append(root)
    
    if symlinks:
        print(f"❌ Still found {len(symlinks)} symlinks:")
        for symlink in symlinks:
            print(f"   - {symlink}")
    else:
        print("✅ No symlinks found - ready for AWS!")
    
    return len(symlinks) == 0

if __name__ == "__main__":
    fix_symlinks()
    print()
    verify_no_symlinks()