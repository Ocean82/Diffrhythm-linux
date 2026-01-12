#!/usr/bin/env python3
"""
Simple script to run symlink resolution for AWS deployment
"""
import subprocess
import sys
import os

def main():
    print("=== DiffRhythm AWS Deployment Setup ===")
    print("Resolving symlinks for AWS compatibility...")
    
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run the symlink resolution
    try:
        result = subprocess.run([sys.executable, "resolve_symlinks.py"], 
                              capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        if result.returncode == 0:
            print("\n✅ SUCCESS: All symlinks resolved for AWS deployment!")
        else:
            print(f"\n❌ ERROR: Script failed with return code {result.returncode}")
            
    except Exception as e:
        print(f"❌ ERROR: Failed to run symlink resolution: {e}")

if __name__ == "__main__":
    main()