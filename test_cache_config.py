#!/usr/bin/env python3
"""
Test cache directory configuration
"""

import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Check environment variables
    print("\nEnvironment Variables:")
    huggingface_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    hf_home = os.environ.get("HF_HOME")
    print(f"HUGGINGFACE_HUB_CACHE: {huggingface_cache}")
    print(f"HF_HOME: {hf_home}")
    
    # Check DEFAULT_CACHE_DIR from infer_utils
    print("\nChecking infer_utils:")
    try:
        from infer.infer_utils import DEFAULT_CACHE_DIR
        print(f"DEFAULT_CACHE_DIR: {DEFAULT_CACHE_DIR}")
        
        # Check if directory exists
        print(f"Directory exists: {os.path.exists(DEFAULT_CACHE_DIR)}")
        
        if os.path.exists(DEFAULT_CACHE_DIR):
            print(f"Directory is readable: {os.access(DEFAULT_CACHE_DIR, os.R_OK)}")
            print(f"Directory is writable: {os.access(DEFAULT_CACHE_DIR, os.W_OK)}")
            
            # List contents
            try:
                contents = os.listdir(DEFAULT_CACHE_DIR)
                print(f"Contents ({len(contents)} items):")
                for item in contents:
                    item_path = os.path.join(DEFAULT_CACHE_DIR, item)
                    if os.path.isdir(item_path):
                        print(f"  [DIR] {item}")
                    else:
                        print(f"  [FILE] {item}")
            except Exception as e:
                print(f"Error listing directory contents: {e}")
        else:
            print("Directory does not exist. Attempting to create...")
            try:
                os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
                print("Directory created successfully")
                print(f"New directory exists: {os.path.exists(DEFAULT_CACHE_DIR)}")
            except Exception as e:
                print(f"Error creating directory: {e}")
                
    except Exception as e:
        print(f"Error importing infer_utils: {type(e).__name__}: {e}")
        import traceback
        print(traceback.format_exc())

    # Test hf_hub_download directly
    print("\nTesting hf_hub_download:")
    try:
        from huggingface_hub import hf_hub_download
        
        # Test with a small file that's likely to be cached
        cached_path = hf_hub_download(
            repo_id="ASLP-lab/DiffRhythm-1_2",
            filename="cfm_model.pt",
            cache_dir="D:\\_hugging-face"
        )
        
        print(f"Model cached at: {cached_path}")
        print(f"File exists: {os.path.exists(cached_path)}")
        if os.path.exists(cached_path):
            print(f"File size: {os.path.getsize(cached_path) / (1024*1024):.2f} MB")
            
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
