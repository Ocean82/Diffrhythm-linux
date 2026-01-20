#!/usr/bin/env python3
"""
Patient CPU test - run inference with proper timeout and monitoring
"""
import subprocess
import sys
import os
import time
import threading


def monitor_progress():
    """Monitor progress by checking log files and output"""
    while True:
        time.sleep(30)  # Check every 30 seconds

        # Check if any output files exist
        output_files = [
            "/mnt/d/EMBERS-BANK/DiffRhythm-LINUX/output/output_fixed.wav",
            "/mnt/d/EMBERS-BANK/DiffRhythm-LINUX/output/output.wav",
        ]

        for output_file in output_files:
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"[MONITOR] Found output: {output_file} ({file_size} bytes)")
                return

        print("[MONITOR] Still processing... (this can take 15-30 minutes on CPU)")


def run_patient_inference():
    """Run inference with 30-minute timeout"""
    print("Starting PATIENT CPU inference test...")
    print("This will take 15-30 minutes on CPU. Please be patient.")
    print("=" * 60)

    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
    monitor_thread.start()

    cmd = [
        "/mnt/d/EMBERS-BANK/DiffRhythm-LINUX/.venv/bin/python",
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

    print(f"Command: {' '.join(cmd[1:])}")  # Skip long python path
    print("Starting inference... (timeout: 30 minutes)")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd="/mnt/d/EMBERS-BANK/DiffRhythm-LINUX",
            timeout=1800,  # 30 minutes
            capture_output=True,
            text=True,
        )

        elapsed = time.time() - start_time
        print(
            f"\nInference completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)"
        )

        print("\nSTDOUT:")
        print(result.stdout)

        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)

        if result.returncode == 0:
            print("✓ SUCCESS! Inference completed.")

            # Check output
            output_file = "/mnt/d/EMBERS-BANK/DiffRhythm-LINUX/output/output_fixed.wav"
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"✓ Output file: {file_size:,} bytes")

                if file_size > 100000:  # > 100KB
                    print("✓ File size looks good - likely contains audio!")
                    return True
                else:
                    print("⚠ File size is small - may be silent")
                    return False
            else:
                print("✗ No output file found")
                return False
        else:
            print(f"✗ Inference failed with return code {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(
            f"\n⚠ Inference timed out after {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)"
        )
        print("This suggests the CPU inference is too slow.")
        print("Recommendation: Install GPU-enabled PyTorch")
        return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    print("DiffRhythm Patient CPU Test")
    print("=" * 40)
    print("This test will run inference with a 30-minute timeout.")
    print("CPU inference is very slow - please be patient.")
    print("")

    choice = input("Continue with patient test? (y/n): ").lower().strip()
    if choice != "y":
        print("Test cancelled.")
        sys.exit(0)

    success = run_patient_inference()
    sys.exit(0 if success else 1)
