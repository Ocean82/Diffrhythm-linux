#!/usr/bin/env python3
"""
Test the fixed inference with simple lyrics
"""
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

def test_fixed_inference():
    """Test the fixed inference pipeline"""
    print("="*60)
    print("Testing Fixed Inference")
    print("="*60)
    
    # Create simple test lyrics
    test_lyrics = """[00:00.00]Hello world
[00:05.00]This is a test song
[00:10.00]Testing the fixed version
[00:15.00]Hope it works now"""
    
    # Save test lyrics
    lyrics_path = "./output/test_fix.lrc"
    os.makedirs("./output", exist_ok=True)
    with open(lyrics_path, 'w', encoding='utf-8') as f:
        f.write(test_lyrics)
    
    print(f"✓ Test lyrics saved to {lyrics_path}")
    
    # Import and run the fixed inference
    try:
        import subprocess
        import sys
        
        # Run the fixed inference script
        cmd = [
            sys.executable, "fix_infer.py",
            "--lrc-path", lyrics_path,
            "--ref-prompt", "upbeat pop song with clear vocals",
            "--audio-length", "95",
            "--output-dir", "./output",
            "--chunked"
        ]
        
        print("Running fixed inference...")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✓ Fixed inference completed successfully!")
            
            # Check if output file exists
            output_path = "./output/output_fixed.wav"
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"✓ Output file created: {output_path}")
                print(f"✓ File size: {file_size:,} bytes")
                
                if file_size > 10000:  # At least 10KB
                    print("✓ File size looks reasonable - likely contains audio!")
                    return True
                else:
                    print("⚠ File size is very small - may be silent")
                    return False
            else:
                print("✗ Output file was not created")
                return False
        else:
            print(f"✗ Fixed inference failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠ Inference timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"✗ Error running fixed inference: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_inference()
    sys.exit(0 if success else 1)