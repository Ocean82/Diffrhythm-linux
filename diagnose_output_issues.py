#!/usr/bin/env python3
"""
Comprehensive diagnostic script to check all potential output issues
"""
import os
import sys
import stat
import torch
import json
from pathlib import Path


def check_output_permissions():
    """Check output directory permissions and access"""
    print("=" * 60)
    print("CHECKING OUTPUT DIRECTORY PERMISSIONS")
    print("=" * 60)

    output_dirs = ["./output", "./infer/example/output", "output"]

    for output_dir in output_dirs:
        print(f"\nChecking directory: {output_dir}")

        # Check if directory exists
        if os.path.exists(output_dir):
            print(f"  ✓ Directory exists")

            # Check if it's actually a directory
            if os.path.isdir(output_dir):
                print(f"  ✓ Is a directory")

                # Check write permissions
                if os.access(output_dir, os.W_OK):
                    print(f"  ✓ Write permission granted")

                    # Test actual write capability
                    test_file = os.path.join(output_dir, "test_write.tmp")
                    try:
                        with open(test_file, "w") as f:
                            f.write("test")
                        os.remove(test_file)
                        print(f"  ✓ Write test successful")
                    except Exception as e:
                        print(f"  ✗ Write test failed: {e}")

                else:
                    print(f"  ✗ No write permission")

                # Show detailed permissions
                stat_info = os.stat(output_dir)
                permissions = stat.filemode(stat_info.st_mode)
                print(f"  Permissions: {permissions}")

            else:
                print(f"  ✗ Path exists but is not a directory")
        else:
            print(f"  ⚠ Directory does not exist")

            # Try to create it
            try:
                os.makedirs(output_dir, exist_ok=True)
                print(f"  ✓ Created directory successfully")
            except Exception as e:
                print(f"  ✗ Failed to create directory: {e}")


def check_model_files():
    """Check for required model files"""
    print("\n" + "=" * 60)
    print("CHECKING MODEL FILES")
    print("=" * 60)

    # Check cache directories
    cache_dirs = [
        "./pretrained",
        "/mnt/d/_hugging-face",
        "D:\\_hugging-face",
        os.path.expanduser("~/.cache/huggingface"),
        os.environ.get("HUGGINGFACE_HUB_CACHE", ""),
    ]

    required_models = {
        "CFM Model": ["cfm_model.pt", "cfm_full_model.pt"],
        "VAE Model": ["vae_model.pt"],
        "Config": ["diffrhythm-1b.json"],
    }

    for cache_dir in cache_dirs:
        if not cache_dir or not os.path.exists(cache_dir):
            continue

        print(f"\nChecking cache directory: {cache_dir}")

        # Look for model files recursively
        for root, dirs, files in os.walk(cache_dir):
            for model_type, model_files in required_models.items():
                for model_file in model_files:
                    if model_file in files:
                        full_path = os.path.join(root, model_file)
                        file_size = os.path.getsize(full_path)
                        print(
                            f"  ✓ Found {model_type}: {full_path} ({file_size:,} bytes)"
                        )

    # Check config directory
    config_dir = "./config"
    print(f"\nChecking config directory: {config_dir}")
    if os.path.exists(config_dir):
        for file in os.listdir(config_dir):
            if file.endswith(".json"):
                config_path = os.path.join(config_dir, file)
                print(f"  ✓ Found config: {config_path}")

                # Validate JSON
                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    print(f"    ✓ Valid JSON with {len(config)} keys")
                except Exception as e:
                    print(f"    ✗ Invalid JSON: {e}")
    else:
        print(f"  ✗ Config directory not found")


def check_vocoder_backend():
    """Check vocoder and audio backend availability"""
    print("\n" + "=" * 60)
    print("CHECKING VOCODER AND AUDIO BACKEND")
    print("=" * 60)

    # Check if VAE is being used as vocoder (DiffRhythm uses VAE for audio generation)
    print("DiffRhythm uses VAE for audio generation (not separate vocoder)")

    # Check audio libraries
    audio_libs = {
        "torchaudio": "torch audio processing",
        "librosa": "audio analysis",
        "soundfile": "audio file I/O",
    }

    for lib_name, description in audio_libs.items():
        try:
            __import__(lib_name)
            print(f"  ✓ {lib_name} available ({description})")
        except ImportError:
            print(f"  ✗ {lib_name} missing ({description})")

    # Check audio backends
    try:
        import torchaudio

        backends = torchaudio.list_audio_backends()
        print(f"  ✓ Available audio backends: {backends}")

        # Test audio save capability
        test_tensor = torch.randn(2, 1000)  # 2 channels, 1000 samples
        test_path = "./output/test_audio_backend.wav"
        os.makedirs("./output", exist_ok=True)

        try:
            torchaudio.save(test_path, test_tensor, sample_rate=44100)
            if os.path.exists(test_path):
                file_size = os.path.getsize(test_path)
                print(f"  ✓ Audio save test successful ({file_size} bytes)")
                os.remove(test_path)
            else:
                print(f"  ✗ Audio save test failed - file not created")
        except Exception as e:
            print(f"  ✗ Audio save test failed: {e}")

    except ImportError:
        print(f"  ✗ torchaudio not available")


def check_inference_script_flags():
    """Check inference script for output-related flags"""
    print("\n" + "=" * 60)
    print("CHECKING INFERENCE SCRIPT FLAGS")
    print("=" * 60)

    script_paths = ["infer/infer.py", "fix_infer.py", "api.py"]

    for script_path in script_paths:
        if not os.path.exists(script_path):
            continue

        print(f"\nAnalyzing script: {script_path}")

        try:
            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for output-related flags and settings
            output_indicators = {
                "--output-dir": "output directory argument",
                "torchaudio.save": "audio saving function",
                "sample_rate": "audio sample rate setting",
                ".wav": "WAV file extension",
                "output_path": "output path variable",
            }

            for indicator, description in output_indicators.items():
                if indicator in content:
                    print(f"  ✓ Found {description}")
                else:
                    print(f"  ⚠ Missing {description}")

            # Check for potential output format issues
            if "sample_rate=44100" in content:
                print(f"  ✓ Standard sample rate (44100 Hz)")
            elif "sample_rate" in content:
                print(f"  ⚠ Custom sample rate detected")
            else:
                print(f"  ✗ No sample rate specified")

        except Exception as e:
            print(f"  ✗ Error reading script: {e}")


def check_pytorch_audio_support():
    """Check PyTorch audio support and device"""
    print("\n" + "=" * 60)
    print("CHECKING PYTORCH AUDIO SUPPORT")
    print("=" * 60)

    try:
        import torch

        print(f"  ✓ PyTorch version: {torch.__version__}")

        # Check device availability
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print(f"  ⚠ CUDA not available (using CPU)")

        # Check tensor operations
        test_tensor = torch.randn(2, 1000)
        max_val = torch.max(torch.abs(test_tensor))
        print(f"  ✓ Tensor operations working (test max: {max_val:.4f})")

        # Test normalization (the problematic operation)
        if max_val > 0:
            normalized = (
                test_tensor.div(max_val).clamp(-1, 1).mul(32767).to(torch.int16)
            )
            print(
                f"  ✓ Normalization working (range: {normalized.min()} to {normalized.max()})"
            )
        else:
            print(f"  ⚠ Test tensor has zero max (would cause normalization issue)")

    except Exception as e:
        print(f"  ✗ PyTorch error: {e}")


def check_existing_outputs():
    """Check for any existing output files"""
    print("\n" + "=" * 60)
    print("CHECKING EXISTING OUTPUT FILES")
    print("=" * 60)

    search_dirs = [".", "./output", "./infer/example/output"]
    audio_extensions = [".wav", ".mp3", ".flac", ".m4a"]

    found_files = []

    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue

        print(f"\nSearching in: {search_dir}")

        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    full_path = os.path.join(root, file)
                    file_size = os.path.getsize(full_path)
                    mod_time = os.path.getmtime(full_path)

                    print(f"  Found: {full_path}")
                    print(f"    Size: {file_size:,} bytes")
                    print(f"    Modified: {mod_time}")

                    if file_size < 1000:
                        print(f"    ⚠ Very small file - likely silent or corrupted")
                    elif file_size > 1000000:  # > 1MB
                        print(f"    ✓ Reasonable size for audio file")
                    else:
                        print(f"    ? Small but potentially valid")

                    found_files.append(full_path)

    if not found_files:
        print("  No audio files found")
    else:
        print(f"\nTotal audio files found: {len(found_files)}")


def main():
    """Run all diagnostic checks"""
    print("DiffRhythm Output Issues Diagnostic")
    print("=" * 60)
    print(
        "This script will check for common issues that prevent audio output generation."
    )
    print("")

    try:
        check_output_permissions()
        check_model_files()
        check_vocoder_backend()
        check_inference_script_flags()
        check_pytorch_audio_support()
        check_existing_outputs()

        print("\n" + "=" * 60)
        print("DIAGNOSTIC COMPLETE")
        print("=" * 60)
        print("Review the results above to identify potential issues.")
        print("Common fixes:")
        print("1. Ensure output directory has write permissions")
        print("2. Verify all model files are downloaded and accessible")
        print("3. Check that torchaudio can save files")
        print("4. Install missing audio libraries if needed")
        print(
            "5. Use the fixed inference script (fix_infer.py) for better error handling"
        )

    except Exception as e:
        print(f"\nDiagnostic script error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
