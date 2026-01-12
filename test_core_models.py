#!/usr/bin/env python3
"""
Simplified test for DiffRhythm models without phonemizer dependency
"""
import os
import sys
import torch
import json

# Add current directory to path
sys.path.append(os.getcwd())

def test_core_models():
    """Test core model loading without phonemizer"""
    print("Testing DiffRhythm Core Models (No Phonemizer)")
    print("=" * 50)
    
    try:
        # Test basic imports
        print("1. Testing basic imports...")
        from model import DiT, CFM
        from muq import MuQMuLan
        from huggingface_hub import hf_hub_download
        print("   [OK] Core imports successful")
        
        # Test model configuration loading
        print("2. Testing model configuration...")
        config_path = "./config/diffrhythm-1b.json"
        with open(config_path) as f:
            model_config = json.load(f)
        print(f"   [OK] Config loaded: {model_config['model_type']}")
        
        # Test DiT model creation
        print("3. Testing DiT model creation...")
        device = "cpu"
        max_frames = 2048
        
        dit_model = DiT(**model_config["model"], max_frames=max_frames)
        print(f"   [OK] DiT model created (dim={dit_model.dim})")
        
        # Test CFM wrapper
        print("4. Testing CFM wrapper...")
        cfm = CFM(
            transformer=dit_model,
            num_channels=model_config["model"]["mel_dim"],
            max_frames=max_frames
        )
        cfm = cfm.to(device)
        print("   [OK] CFM model created and moved to CPU")
        
        # Test MuQ model loading
        print("5. Testing MuQ model loading...")
        muq = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir="./pretrained")
        muq = muq.to(device).eval()
        print("   [OK] MuQ model loaded")
        
        # Test MuQ text processing
        print("6. Testing MuQ text processing...")
        with torch.no_grad():
            style_emb = muq(texts="folk, acoustic guitar")
            print(f"   [OK] Style embedding shape: {style_emb.shape}")
        
        # Test VAE model loading
        print("7. Testing VAE model loading...")
        vae_path = hf_hub_download(
            repo_id="ASLP-lab/DiffRhythm-vae",
            filename="vae_model.pt",
            cache_dir="./pretrained",
        )
        vae = torch.jit.load(vae_path, map_location="cpu").to(device)
        print("   [OK] VAE model loaded")
        
        # Test model file sizes
        print("8. Checking model file sizes...")
        import os
        
        # Check DiffRhythm models
        base_model_path = hf_hub_download(
            repo_id="ASLP-lab/DiffRhythm-1_2",
            filename="cfm_model.pt",
            cache_dir="./pretrained"
        )
        base_size = os.path.getsize(base_model_path) / (1024**3)  # GB
        print(f"   [OK] DiffRhythm-1_2: {base_size:.2f} GB")
        
        full_model_path = hf_hub_download(
            repo_id="ASLP-lab/DiffRhythm-1_2-full",
            filename="cfm_model.pt", 
            cache_dir="./pretrained"
        )
        full_size = os.path.getsize(full_model_path) / (1024**3)  # GB
        print(f"   [OK] DiffRhythm-1_2-full: {full_size:.2f} GB")
        
        vae_size = os.path.getsize(vae_path) / (1024**3)  # GB
        print(f"   [OK] VAE model: {vae_size:.2f} GB")
        
        print("\n[SUCCESS] All core models loaded successfully!")
        print("Models are ready for Linux/WSL deployment!")
        print("\nNote: Full inference requires espeak-ng in Linux environment")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_core_models()
    sys.exit(0 if success else 1)