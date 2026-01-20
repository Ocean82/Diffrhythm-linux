#!/usr/bin/env python3
"""
ODE Integration Stall Diagnostic Tool

This script helps diagnose why ODE integration might be stalling during song generation.
Run this to identify bottlenecks and get recommendations.
"""

import os
import sys
import time
import torch

def print_section(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def check_system():
    print_section("SYSTEM CHECK")

    # Python version
    print(f"Python version: {sys.version}")

    # PyTorch version
    print(f"PyTorch version: {torch.__version__}")

    # CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("✗ CUDA not available - running on CPU")
        print("  ⚠ CPU inference is 10-50x slower than GPU!")

    # CPU info
    try:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        print(f"CPU cores: {cpu_count}")
    except:
        pass

    # Memory
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"RAM: {mem.total / 1e9:.1f} GB (available: {mem.available / 1e9:.1f} GB)")
    except ImportError:
        print("(Install psutil for memory info: pip install psutil)")

def check_model_size():
    print_section("MODEL SIZE ANALYSIS")

    # Load config
    try:
        import json
        with open("./config/diffrhythm-1b.json") as f:
            config = json.load(f)

        model_config = config["model"]
        dim = model_config["dim"]
        depth = model_config["depth"]
        heads = model_config["heads"]
        ff_mult = model_config["ff_mult"]

        # Estimate parameters
        # Each transformer layer has:
        # - Self-attention: 4 * dim * dim (Q, K, V, O projections)
        # - FFN: 2 * dim * (dim * ff_mult)
        params_per_layer = 4 * dim * dim + 2 * dim * (dim * ff_mult)
        total_params = params_per_layer * depth

        print(f"Model configuration:")
        print(f"  - Hidden dim: {dim}")
        print(f"  - Depth: {depth} layers")
        print(f"  - Heads: {heads}")
        print(f"  - FF multiplier: {ff_mult}")
        print(f"  - Estimated parameters: ~{total_params / 1e9:.2f}B")

        # Time estimates
        print(f"\nInference time estimates per ODE step:")
        print(f"  - GPU (RTX 3090): ~1-2 seconds")
        print(f"  - GPU (RTX 4090): ~0.5-1 second")
        print(f"  - CPU (modern): ~30-60 seconds")
        print(f"  - CPU (older): ~60-120 seconds")

    except Exception as e:
        print(f"Could not load model config: {e}")

def benchmark_transformer():
    print_section("TRANSFORMER BENCHMARK")

    if not torch.cuda.is_available():
        print("Running CPU benchmark (this may take a few minutes)...")

    try:
        from model import DiT
        import json

        with open("./config/diffrhythm-1b.json") as f:
            config = json.load(f)

        # Create a smaller test model
        test_config = config["model"].copy()
        test_config["depth"] = 2  # Reduce for quick test

        print("Creating test model (2 layers)...")
        model = DiT(**test_config, max_frames=2048)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()

        # Create dummy inputs
        batch_size = 1
        seq_len = 512  # Reduced for quick test
        x = torch.randn(batch_size, seq_len, 64, device=device)
        cond = torch.randn(batch_size, seq_len, 64, device=device)
        text = torch.randint(0, 100, (batch_size, seq_len), device=device)
        time_tensor = torch.tensor([0.5], device=device)
        style = torch.randn(batch_size, 512, device=device)
        start_time = torch.tensor([0.0], device=device)

        print(f"Running benchmark on {device}...")

        # Warmup
        with torch.no_grad():
            _ = model(x, cond, text, time_tensor, False, False,
                     style_prompt=style, start_time=start_time)

        # Benchmark
        times = []
        for i in range(3):
            start = time.time()
            with torch.no_grad():
                _ = model(x, cond, text, time_tensor, False, False,
                         style_prompt=style, start_time=start_time)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - start)

        avg_time = sum(times) / len(times)

        # Scale to full model (16 layers vs 2)
        full_model_estimate = avg_time * 8  # 16/2 layers
        # Scale to full sequence (2048 vs 512)
        full_seq_estimate = full_model_estimate * 4  # 2048/512

        print(f"\nBenchmark results (2-layer, seq_len=512):")
        print(f"  - Average time: {avg_time*1000:.1f}ms")
        print(f"\nEstimated full model (16 layers, seq_len=2048):")
        print(f"  - Single forward pass: {full_seq_estimate:.1f}s")
        print(f"  - Per ODE step (2 passes for CFG): {full_seq_estimate * 2:.1f}s")
        print(f"  - 8 steps total: {full_seq_estimate * 2 * 8 / 60:.1f} minutes")
        print(f"  - 16 steps total: {full_seq_estimate * 2 * 16 / 60:.1f} minutes")
        print(f"  - 32 steps total: {full_seq_estimate * 2 * 32 / 60:.1f} minutes")

    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

def provide_recommendations():
    print_section("RECOMMENDATIONS")

    has_gpu = torch.cuda.is_available()

    if has_gpu:
        print("✓ GPU detected - you should have reasonable performance")
        print("\nIf still stalling, check:")
        print("  1. GPU memory usage (nvidia-smi)")
        print("  2. Other GPU processes consuming resources")
        print("  3. Try reducing batch_infer_num to 1")
    else:
        print("CPU-only mode detected. Recommendations:")
        print("")
        print("1. REDUCE ODE STEPS (most impact):")
        print("   python -m infer.infer --steps 4 ...  # Quick preview")
        print("   python -m infer.infer --steps 8 ...  # Reasonable quality")
        print("")
        print("2. REDUCE AUDIO LENGTH:")
        print("   python -m infer.infer --audio-length 95 ...  # Max for faster model")
        print("")
        print("3. USE CHUNKED DECODING:")
        print("   python -m infer.infer --chunked ...  # Reduces memory usage")
        print("")
        print("4. SET EXPLICIT TIMEOUT:")
        print("   python -m infer.infer --timeout 1800 ...  # 30 minute timeout")
        print("")
        print("5. CONSIDER GPU OPTIONS:")
        print("   - Cloud GPU (Google Colab, RunPod, etc.)")
        print("   - Local GPU (even older GPUs are much faster)")
        print("")
        print("6. ENVIRONMENT VARIABLES:")
        print("   OMP_NUM_THREADS=4  # Limit CPU threads")
        print("   MKL_NUM_THREADS=4  # Limit MKL threads")

def main():
    print("\n" + "="*60)
    print(" DiffRhythm ODE Stall Diagnostic Tool")
    print("="*60)

    check_system()
    check_model_size()
    benchmark_transformer()
    provide_recommendations()

    print("\n" + "="*60)
    print(" Diagnostic Complete")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
