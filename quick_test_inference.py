#!/usr/bin/env python3
"""
Quick test of inference pipeline - will timeout but should show where it fails
"""
import subprocess
import sys
import os


def test_inference():
    """Test inference with timeout to see where it fails"""
    print("Testing inference pipeline (will timeout after 2 minutes)...")

    cmd = [
        sys.executable,
        "fix_infer.py",
        "--lrc-path",
        "./output/test.lrc",
        "--ref-prompt",
        "simple test song",
        "--audio-length",
        "95",
        "--output-dir",
        "./output",
        "--chunked",
    ]

    print(f"Command: {' '.join(cmd)}")
    print("This will timeout after 2 minutes, but we can see the progress...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minutes
            cwd="/mnt/d/EMBERS-BANK/DiffRhythm-LINUX",
        )

        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        if result.returncode == 0:
            print("✓ Inference completed successfully!")
            return True
        else:
            print(f"✗ Inference failed with return code {result.returncode}")
            return False

    except subprocess.TimeoutExpired as e:
        print("⚠ Inference timed out after 2 minutes")
        print("STDOUT so far:")
        if e.stdout:
            print(e.stdout.decode())
        print("STDERR so far:")
        if e.stderr:
            print(e.stderr.decode())

        # Check if any output was created
        output_files = ["./output/output_fixed.wav", "./output/output.wav"]

        for output_file in output_files:
            full_path = f"/mnt/d/EMBERS-BANK/DiffRhythm-LINUX/{output_file}"
            if os.path.exists(full_path):
                file_size = os.path.getsize(full_path)
                print(f"✓ Found output file: {output_file} ({file_size} bytes)")
                return True

        print("No output files found yet")
        return False

    except Exception as e:
        print(f"Error running inference: {e}")
        return False


if __name__ == "__main__":
    success = test_inference()
    sys.exit(0 if success else 1)
