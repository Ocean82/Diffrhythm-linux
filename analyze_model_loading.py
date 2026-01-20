#!/usr/bin/env python3
"""
Analyze exactly where model loading gets stuck
"""
import os
import sys
import time
import torch

# Add current directory to path
sys.path.append(os.getcwd())


def time_model_loading():
    """Time each step of model loading to identify bottleneck"""
    print("=" * 60)
    print("ANALYZING MODEL LOADING BOTTLENECKS")
    print("=" * 60)

    device = "cpu"
    max_frames = 2048

    # Step 1: Import modules
    print("\n[1/6] Importing modules...")
    start = time.time()
    try:
        from model import DiT, CFM
        from huggingface_hub import hf_hub_download
        import json

        print(f"  ✓ Imports completed in {time.time()-start:.2f}s")
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False

    # Step 2: Load config
    print("\n[2/6] Loading config...")
    start = time.time()
    try:
        dit_config_path = "./config/diffrhythm-1b.json"
        with open(dit_config_path) as f:
            model_config = json.load(f)
        print(f"  ✓ Config loaded in {time.time()-start:.2f}s")
        print(f"  Model config keys: {list(model_config.keys())}")
    except Exception as e:
        print(f"  ✗ Config loading failed: {e}")
        return False

    # Step 3: Get model file path
    print("\n[3/6] Getting model file path...")
    start = time.time()
    try:
        dit_ckpt_path = hf_hub_download(
            repo_id="ASLP-lab/DiffRhythm-1_2",
            filename="cfm_model.pt",
            cache_dir="./pretrained",
        )
        print(f"  ✓ Model path obtained in {time.time()-start:.2f}s")
        print(f"  Path: {dit_ckpt_path}")

        # Check file size
        file_size = os.path.getsize(dit_ckpt_path)
        print(f"  File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
    except Exception as e:
        print(f"  ✗ Model path failed: {e}")
        return False

    # Step 4: Initialize DiT model (this might be slow)
    print("\n[4/6] Initializing DiT model...")
    start = time.time()
    try:
        dit_model_cls = DiT
        print(f"  Model params: {model_config['model']}")
        print("  Creating DiT instance...")

        # This is likely where it gets stuck
        dit_model = dit_model_cls(**model_config["model"], max_frames=max_frames)
        print(f"  ✓ DiT model initialized in {time.time()-start:.2f}s")
    except Exception as e:
        print(f"  ✗ DiT initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Step 5: Create CFM wrapper
    print("\n[5/6] Creating CFM wrapper...")
    start = time.time()
    try:
        cfm = CFM(
            transformer=dit_model,
            num_channels=model_config["model"]["mel_dim"],
            max_frames=max_frames,
        )
        cfm = cfm.to(device)
        print(f"  ✓ CFM created in {time.time()-start:.2f}s")
    except Exception as e:
        print(f"  ✗ CFM creation failed: {e}")
        return False

    # Step 6: Load checkpoint (this is definitely slow)
    print("\n[6/6] Loading checkpoint weights...")
    start = time.time()
    try:
        print("  Loading checkpoint file...")
        checkpoint = torch.load(dit_ckpt_path, weights_only=True, map_location="cpu")
        print(f"  Checkpoint loaded in {time.time()-start:.2f}s")

        print("  Applying weights to model...")
        start_apply = time.time()

        # This is the slow part - loading 2GB+ of weights
        cfm.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"  ✓ Weights applied in {time.time()-start_apply:.2f}s")
        print(f"  ✓ Total checkpoint loading: {time.time()-start:.2f}s")

    except Exception as e:
        print(f"  ✗ Checkpoint loading failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print(f"\n✓ Model loading analysis complete!")
    return True


def suggest_optimizations():
    """Suggest optimizations for CPU-only inference"""
    print("\n" + "=" * 60)
    print("CPU OPTIMIZATION SUGGESTIONS")
    print("=" * 60)

    print("Since you don't have GPU, here are CPU optimizations:")
    print("")
    print("1. **Model Loading Optimization:**")
    print("   - The 2GB+ model weights take time to load into RAM")
    print("   - This is a one-time cost per session")
    print("   - Consider keeping the process running for multiple generations")
    print("")
    print("2. **Inference Optimization:**")
    print("   - Use smaller batch sizes (batch_infer_num=1)")
    print("   - Use chunked decoding (--chunked flag)")
    print(
        "   - Reduce CFM steps from 32 to 16 for faster (but lower quality) generation"
    )
    print("")
    print("3. **Memory Optimization:**")
    print("   - Close other applications to free RAM")
    print("   - Use torch.set_num_threads(1) to avoid CPU oversubscription")
    print("")
    print("4. **Realistic Expectations:**")
    print("   - CPU inference: 10-20 minutes for 95s song")
    print("   - GPU inference: 1-3 minutes for 95s song")
    print("   - This is normal for large diffusion models")


def main():
    """Main analysis function"""
    print("DiffRhythm Model Loading Analysis")
    print("This will identify exactly where model loading gets stuck.")
    print("")

    success = time_model_loading()

    if success:
        print("\n✓ Model loading analysis successful!")
        print("The models can be loaded - the issue is just CPU performance.")
    else:
        print("\n✗ Model loading failed - this identifies the real issue.")

    suggest_optimizations()


if __name__ == "__main__":
    main()
