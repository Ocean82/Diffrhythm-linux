#!/usr/bin/env python3
"""
Final DiffRhythm Deployment Readiness Verification
"""
import os
import sys
import torch
import json
from pathlib import Path

def verify_deployment_readiness():
    """Complete deployment readiness check"""
    print("DiffRhythm AWS Deployment Readiness Check")
    print("=" * 50)
    
    checks_passed = 0
    total_checks = 8
    
    try:
        # Check 1: Model files exist and are real files (not symlinks)
        print("1. Checking model files...")
        model_paths = [
            "pretrained/models--ASLP-lab--DiffRhythm-1_2/snapshots/*/cfm_model.pt",
            "pretrained/models--ASLP-lab--DiffRhythm-1_2-full/snapshots/*/cfm_model.pt", 
            "pretrained/models--ASLP-lab--DiffRhythm-vae/snapshots/*/vae_model.pt"
        ]
        
        for pattern in model_paths:
            matches = list(Path(".").glob(pattern))
            if matches and matches[0].exists() and not matches[0].is_symlink():
                size_gb = matches[0].stat().st_size / (1024**3)
                print(f"   [OK] {matches[0].name}: {size_gb:.2f} GB (real file)")
            else:
                print(f"   [ERROR] Missing or symlink: {pattern}")
                return False
        checks_passed += 1
        
        # Check 2: Core imports work
        print("2. Testing core imports...")
        from model import DiT, CFM
        from muq import MuQMuLan
        from huggingface_hub import hf_hub_download
        print("   [OK] All core imports successful")
        checks_passed += 1
        
        # Check 3: Model loading works
        print("3. Testing model loading...")
        from infer.infer_utils import prepare_model
        device = "cpu"
        max_frames = 2048
        cfm, tokenizer, muq, vae = prepare_model(max_frames, device)
        print("   [OK] All models loaded successfully")
        checks_passed += 1
        
        # Check 4: Text processing works
        print("4. Testing text processing...")
        with torch.no_grad():
            style_emb = muq(texts="folk acoustic guitar")
            print(f"   [OK] Style embedding: {style_emb.shape}")
        checks_passed += 1
        
        # Check 5: Configuration files exist
        print("5. Checking configuration files...")
        config_files = ["config/default.ini", "config/diffrhythm-1b.json"]
        for config_file in config_files:
            if Path(config_file).exists():
                print(f"   [OK] {config_file} exists")
            else:
                print(f"   [ERROR] Missing: {config_file}")
                return False
        checks_passed += 1
        
        # Check 6: G2P files exist
        print("6. Checking G2P files...")
        g2p_files = ["g2p/g2p/vocab.json", "g2p/sources/chinese_lexicon.txt"]
        for g2p_file in g2p_files:
            if Path(g2p_file).exists():
                print(f"   [OK] {g2p_file} exists")
            else:
                print(f"   [ERROR] Missing: {g2p_file}")
                return False
        checks_passed += 1
        
        # Check 7: Example files exist
        print("7. Checking example files...")
        example_files = ["infer/example/eg_en.lrc", "infer/example/vocal.npy"]
        for example_file in example_files:
            if Path(example_file).exists():
                print(f"   [OK] {example_file} exists")
            else:
                print(f"   [ERROR] Missing: {example_file}")
                return False
        checks_passed += 1
        
        # Check 8: Total size calculation
        print("8. Calculating total deployment size...")
        total_size = 0
        for root, dirs, files in os.walk("pretrained"):
            for file in files:
                file_path = Path(root) / file
                if file_path.exists() and not file_path.is_symlink():
                    total_size += file_path.stat().st_size
        
        total_gb = total_size / (1024**3)
        print(f"   [OK] Total model size: {total_gb:.2f} GB")
        checks_passed += 1
        
        # Final summary
        print(f"\n{'='*50}")
        print(f"DEPLOYMENT READINESS: {checks_passed}/{total_checks} CHECKS PASSED")
        
        if checks_passed == total_checks:
            print("\n🎉 READY FOR AWS DEPLOYMENT!")
            print("\nNext steps for AWS Linux deployment:")
            print("1. Upload your project to AWS")
            print("2. Install espeak-ng: sudo apt-get install espeak-ng")
            print("3. Install Python deps: pip install -r requirements.txt")
            print("4. Run inference: python infer/infer.py [options]")
            print("\nYour DiffRhythm setup is production-ready! 🎵")
            return True
        else:
            print("\n❌ DEPLOYMENT NOT READY")
            print("Please fix the issues above before deploying.")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_deployment_readiness()
    sys.exit(0 if success else 1)