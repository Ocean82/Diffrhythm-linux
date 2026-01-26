"""
Test script for file cleanup functionality
"""
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from backend.config import Config
from backend.cleanup import get_files_to_delete, cleanup_old_files

def test_cleanup():
    """Test cleanup functionality"""
    
    print("=" * 80)
    print("File Cleanup Test")
    print("=" * 80)
    print()
    
    # Ensure output directory exists
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create test files with different ages
    test_job_id = "test-cleanup-job"
    test_dir = Config.OUTPUT_DIR / test_job_id
    test_dir.mkdir(exist_ok=True)
    
    # Create a file that should be kept (recent)
    recent_file = test_dir / "recent.wav"
    recent_file.touch()
    print(f"Created recent file: {recent_file}")
    
    # Create a file that should be deleted (old)
    old_file = test_dir / "old.wav"
    old_file.touch()
    # Set modification time to 35 days ago (older than 30 day retention)
    old_time = time.time() - (35 * 24 * 60 * 60)
    os.utime(old_file, (old_time, old_time))
    print(f"Created old file: {old_file} (age: 35 days)")
    
    print()
    print("Files to delete:")
    print("-" * 80)
    
    files_to_delete = get_files_to_delete()
    print(f"Found {len(files_to_delete)} files to delete")
    
    for file_info in files_to_delete:
        print(f"  - {file_info['file_path']} (age: {file_info['age_days']} days)")
        if file_info.get('s3_url'):
            print(f"    S3: {file_info['s3_url']}")
    
    print()
    print("Running cleanup...")
    print("-" * 80)
    
    result = cleanup_old_files()
    
    print(f"Cleanup result: {result}")
    print()
    
    # Verify files
    print("Verifying cleanup...")
    print("-" * 80)
    
    if recent_file.exists():
        print(f"[SUCCESS] Recent file still exists: {recent_file}")
    else:
        print(f"[ERROR] Recent file was deleted (should be kept)")
    
    if old_file.exists():
        print(f"[ERROR] Old file still exists (should be deleted): {old_file}")
    else:
        print(f"[SUCCESS] Old file was deleted: {old_file}")
    
    # Clean up test directory
    if test_dir.exists():
        try:
            for file in test_dir.iterdir():
                file.unlink()
            test_dir.rmdir()
            print(f"[INFO] Cleaned up test directory: {test_dir}")
        except Exception as e:
            print(f"[WARNING] Could not clean up test directory: {e}")
    
    print()
    print("=" * 80)
    print("Test Complete")
    print("=" * 80)

if __name__ == "__main__":
    test_cleanup()
