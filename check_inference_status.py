#!/usr/bin/env python3
"""
Quick status check for inference progress
"""
import os
import time


def check_status():
    """Check current inference status"""
    print("DiffRhythm Inference Status Check")
    print("=" * 35)

    # Check key files
    files_to_check = {
        "./output/optimized_output.wav": "Final output",
        "./output/inference.log": "Process log",
        "./output/test_sine_wave.wav": "Audio test file",
        "./output/test.lrc": "Test lyrics",
    }

    found_files = []

    for file_path, description in files_to_check.items():
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            mod_time = os.path.getmtime(file_path)
            age_minutes = (time.time() - mod_time) / 60

            print(f"✓ {description}: {file_path}")
            print(f"  Size: {file_size:,} bytes")
            print(f"  Age: {age_minutes:.1f} minutes")

            if file_path.endswith(".wav") and file_size > 100000:
                print(f"  Status: ✓ Good size - likely contains audio")
            elif file_path.endswith(".wav") and file_size < 1000:
                print(f"  Status: ⚠ Very small - likely silent")
            elif file_path.endswith(".wav"):
                print(f"  Status: ? Small but might be valid")

            found_files.append(file_path)
            print()
        else:
            print(f"✗ {description}: Not found")

    # Check if inference is likely running
    log_file = "./output/inference.log"
    if log_file in found_files:
        log_age = (time.time() - os.path.getmtime(log_file)) / 60
        if log_age < 30:  # Less than 30 minutes old
            print("🔄 Inference may still be running (recent log activity)")
        else:
            print("⏸ Inference likely completed or stopped (old log)")

    # Show recent log if available
    if os.path.exists(log_file):
        print("\nRecent log entries:")
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
                recent_lines = lines[-5:] if lines else []
                for line in recent_lines:
                    print(f"  {line.strip()}")
        except Exception as e:
            print(f"  Error reading log: {e}")

    # Summary
    print("\n" + "=" * 35)
    if "./output/optimized_output.wav" in found_files:
        output_size = os.path.getsize("./output/optimized_output.wav")
        if output_size > 100000:
            print("🎵 SUCCESS: Audio output found and looks good!")
            print("   Try playing: ./output/optimized_output.wav")
        else:
            print("⚠ Audio output found but very small (may be silent)")
    else:
        print("⏳ No audio output yet - inference may still be running")
        print("   Expected file: ./output/optimized_output.wav")

    print("\nNext steps:")
    if "./output/optimized_output.wav" not in found_files:
        print("- Wait for inference to complete (15-20 minutes total)")
        print("- Run this script again to check progress")
        print("- Or start new inference: python background_inference.py")
    else:
        print("- Listen to the generated audio file")
        print(
            "- If silent, the normalization fix is working but models may need debugging"
        )


if __name__ == "__main__":
    check_status()
