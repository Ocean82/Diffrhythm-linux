#!/usr/bin/env python3
"""
Script to fix common output issues automatically
"""
import os
import sys
import stat
import json
from pathlib import Path


def fix_output_permissions():
    """Fix output directory permissions"""
    print("Fixing output directory permissions...")

    output_dirs = ["./output", "./infer/example/output"]

    for output_dir in output_dirs:
        try:
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            print(f"  ✓ Created/verified directory: {output_dir}")

            # Set proper permissions (read, write, execute for owner)
            os.chmod(
                output_dir,
                stat.S_IRWXU
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH,
            )
            print(f"  ✓ Set permissions for: {output_dir}")

            # Test write capability
            test_file = os.path.join(output_dir, "test_write.tmp")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            print(f"  ✓ Write test passed for: {output_dir}")

        except Exception as e:
            print(f"  ✗ Failed to fix {output_dir}: {e}")


def verify_model_paths():
    """Verify and report model file locations"""
    print("\nVerifying model file locations...")

    # Check if models are accessible
    try:
        from huggingface_hub import hf_hub_download

        # Test CFM model access
        try:
            cfm_path = hf_hub_download(
                repo_id="ASLP-lab/DiffRhythm-1_2",
                filename="cfm_model.pt",
                cache_dir="./pretrained",
            )
            print(f"  ✓ CFM model accessible: {cfm_path}")
        except Exception as e:
            print(f"  ✗ CFM model issue: {e}")

        # Test VAE model access
        try:
            vae_path = hf_hub_download(
                repo_id="ASLP-lab/DiffRhythm-vae",
                filename="vae_model.pt",
                cache_dir="./pretrained",
            )
            print(f"  ✓ VAE model accessible: {vae_path}")
        except Exception as e:
            print(f"  ✗ VAE model issue: {e}")

    except ImportError:
        print("  ✗ huggingface_hub not available")


def check_audio_format_support():
    """Check and fix audio format support"""
    print("\nChecking audio format support...")

    try:
        import torchaudio

        # Test basic audio tensor creation and saving
        import torch

        test_audio = torch.randn(2, 44100)  # 1 second of stereo audio
        test_path = "./output/format_test.wav"

        # Ensure output directory exists
        os.makedirs("./output", exist_ok=True)

        # Test WAV format (most compatible)
        torchaudio.save(test_path, test_audio, sample_rate=44100)

        if os.path.exists(test_path):
            file_size = os.path.getsize(test_path)
            print(f"  ✓ WAV format test passed ({file_size} bytes)")
            os.remove(test_path)
        else:
            print(f"  ✗ WAV format test failed - file not created")

        # Check available backends
        backends = torchaudio.list_audio_backends()
        print(f"  ✓ Available audio backends: {backends}")

    except Exception as e:
        print(f"  ✗ Audio format test failed: {e}")


def create_test_inference_script():
    """Create a minimal test inference script"""
    print("\nCreating test inference script...")

    test_script = """#!/usr/bin/env python3
import os
import sys
import torch
import torchaudio

# Test minimal audio generation and saving
def test_audio_output():
    print("Testing audio output capability...")
    
    # Create test audio (sine wave)
    sample_rate = 44100
    duration = 2  # seconds
    frequency = 440  # A4 note
    
    t = torch.linspace(0, duration, sample_rate * duration)
    audio = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)  # Add channel dimension
    audio = audio.repeat(2, 1)  # Make stereo
    
    # Normalize properly
    max_val = torch.max(torch.abs(audio))
    if max_val > 1e-8:
        audio = audio / max_val * 0.8  # Normalize to 80% amplitude
    else:
        print("ERROR: Generated audio is silent!")
        return False
    
    # Convert to int16
    audio_int16 = (audio * 32767).clamp(-32767, 32767).to(torch.int16)
    
    # Save
    output_path = "./output/test_sine_wave.wav"
    os.makedirs("./output", exist_ok=True)
    
    try:
        torchaudio.save(output_path, audio_int16.float(), sample_rate=sample_rate)
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✓ Test audio saved: {output_path} ({file_size} bytes)")
            return True
        else:
            print("✗ Test audio file was not created")
            return False
            
    except Exception as e:
        print(f"✗ Failed to save test audio: {e}")
        return False

if __name__ == "__main__":
    success = test_audio_output()
    sys.exit(0 if success else 1)
"""

    script_path = "./test_audio_output.py"
    try:
        with open(script_path, "w") as f:
            f.write(test_script)
        os.chmod(script_path, 0o755)  # Make executable
        print(f"  ✓ Created test script: {script_path}")
        print(f"    Run with: python {script_path}")
    except Exception as e:
        print(f"  ✗ Failed to create test script: {e}")


def check_inference_script_output_format():
    """Check and suggest fixes for inference script output format"""
    print("\nChecking inference script output format...")

    scripts_to_check = ["infer/infer.py", "fix_infer.py"]

    for script_path in scripts_to_check:
        if not os.path.exists(script_path):
            continue

        print(f"  Checking: {script_path}")

        try:
            with open(script_path, "r") as f:
                content = f.read()

            # Check for proper audio saving
            if "torchaudio.save" in content:
                print(f"    ✓ Uses torchaudio.save")
            else:
                print(f"    ⚠ Does not use torchaudio.save")

            # Check for sample rate
            if "sample_rate=44100" in content:
                print(f"    ✓ Uses standard sample rate (44100)")
            elif "sample_rate" in content:
                print(f"    ⚠ Uses custom sample rate")
            else:
                print(f"    ✗ No sample rate specified")

            # Check for proper normalization
            if "torch.max(torch.abs(" in content:
                print(f"    ⚠ Uses potentially problematic normalization")
                print(f"      Recommend using safe_normalize_audio function")

        except Exception as e:
            print(f"    ✗ Error reading {script_path}: {e}")


def main():
    """Run all fixes"""
    print("DiffRhythm Output Issues - Automatic Fixes")
    print("=" * 50)

    try:
        fix_output_permissions()
        verify_model_paths()
        check_audio_format_support()
        create_test_inference_script()
        check_inference_script_output_format()

        print("\n" + "=" * 50)
        print("FIXES COMPLETE")
        print("=" * 50)
        print("Next steps:")
        print("1. Run: python test_audio_output.py")
        print(
            "2. If that works, try: python fix_infer.py --lrc-path output/test.lrc --ref-prompt 'test song'"
        )
        print("3. Check the output directory for generated files")

    except Exception as e:
        print(f"Fix script error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
