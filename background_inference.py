#!/usr/bin/env python3
"""
Background inference that saves progress and can be monitored
"""
import os
import sys
import time
import json
import subprocess
from pathlib import Path


def start_background_inference():
    """Start inference in background and monitor progress"""
    print("Starting DiffRhythm inference in background...")
    print("This will run independently and save progress to files.")
    print("")

    # Create progress tracking files
    progress_file = "./output/inference_progress.json"
    log_file = "./output/inference.log"

    # Ensure output directory exists
    os.makedirs("./output", exist_ok=True)

    # Initialize progress tracking
    progress = {
        "status": "starting",
        "start_time": time.time(),
        "current_step": "initializing",
        "estimated_completion": None,
        "output_file": "./output/optimized_output.wav",
    }

    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)

    # Start the inference process
    cmd = [
        "/mnt/d/EMBERS-BANK/DiffRhythm-LINUX/.venv/bin/python",
        "optimized_cpu_inference.py",
    ]

    print(f"Starting background process...")
    print(f"Progress file: {progress_file}")
    print(f"Log file: {log_file}")
    print(f"Expected output: ./output/optimized_output.wav")
    print("")

    # Start process in background
    with open(log_file, "w") as log:
        process = subprocess.Popen(
            cmd,
            cwd="/mnt/d/EMBERS-BANK/DiffRhythm-LINUX",
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
            text=True,
        )

        # Send 'y' to confirm start
        process.stdin.write("y\n")
        process.stdin.flush()
        process.stdin.close()

    print(f"✓ Background process started (PID: {process.pid})")
    print("")
    print("Monitor progress with:")
    print(f"  tail -f {log_file}")
    print(f"  cat {progress_file}")
    print("")
    print("Expected timeline:")
    print("  0-8 minutes: Model loading")
    print("  8-20 minutes: CFM sampling (the slow part)")
    print("  20-22 minutes: VAE decoding and saving")
    print("")
    print("Check for output file periodically:")
    print("  ls -la ./output/optimized_output.wav")

    return process


def monitor_progress():
    """Monitor the progress of background inference"""
    progress_file = "./output/inference_progress.json"
    log_file = "./output/inference.log"
    output_file = "./output/optimized_output.wav"

    print("Monitoring inference progress...")
    print("Press Ctrl+C to stop monitoring (inference will continue)")
    print("")

    try:
        while True:
            # Check if output file exists
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"✓ OUTPUT FOUND: {output_file} ({file_size:,} bytes)")
                if file_size > 100000:
                    print("✓ File size looks good - generation likely successful!")
                else:
                    print("⚠ File size is small - may be silent audio")
                break

            # Show recent log entries
            if os.path.exists(log_file):
                try:
                    with open(log_file, "r") as f:
                        lines = f.readlines()
                        if lines:
                            recent_lines = lines[-3:]  # Last 3 lines
                            print("Recent progress:")
                            for line in recent_lines:
                                print(f"  {line.strip()}")
                except:
                    pass

            print(f"Checking again in 30 seconds... (or Ctrl+C to stop monitoring)")
            time.sleep(30)
            print("-" * 50)

    except KeyboardInterrupt:
        print("\n⚠ Monitoring stopped (inference continues in background)")
        print(f"Check output manually: ls -la {output_file}")


def check_existing_inference():
    """Check if there's already an inference running"""
    output_file = "./output/optimized_output.wav"
    log_file = "./output/inference.log"

    if os.path.exists(log_file):
        # Check if log file is recent (within last hour)
        log_age = time.time() - os.path.getmtime(log_file)
        if log_age < 3600:  # 1 hour
            print(f"Found recent inference log (age: {log_age/60:.1f} minutes)")

            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"✓ Output file exists: {file_size:,} bytes")
                return True
            else:
                print("⚠ Inference may still be running...")
                choice = input("Monitor existing inference? (y/n): ").lower().strip()
                if choice == "y":
                    monitor_progress()
                    return True

    return False


def main():
    """Main function"""
    print("DiffRhythm Background Inference Manager")
    print("=" * 45)
    print("This will run inference in the background so you can:")
    print("- Close this terminal")
    print("- Monitor progress periodically")
    print("- Let it run for the full 15-20 minutes needed")
    print("")

    # Check if there's already an inference running
    if check_existing_inference():
        return

    print("Starting new background inference...")
    choice = input("Continue? (y/n): ").lower().strip()
    if choice != "y":
        print("Cancelled.")
        return

    try:
        process = start_background_inference()

        print("Background inference started!")
        print("")
        choice = input("Monitor progress now? (y/n): ").lower().strip()
        if choice == "y":
            monitor_progress()
        else:
            print("Inference running in background.")
            print("Check later with: ls -la ./output/optimized_output.wav")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
