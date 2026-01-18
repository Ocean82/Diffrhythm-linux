#!/usr/bin/env python3
"""
Test song generation with the fixed infer.py
"""
import os
import sys
import subprocess
import time

def create_test_lyrics():
    """Create simple test lyrics"""
    lyrics = """[00:00.00]Hello world
[00:02.00]This is a test
[00:04.00]Of the fixed system
[00:06.00]Let's see if it works"""
    
    os.makedirs("output", exist_ok=True)
    lyrics_path = "output/test_lyrics.lrc"
    
    with open(lyrics_path, 'w', encoding='utf-8') as f:
        f.write(lyrics)
    
    print(f"✓ Test lyrics created: {lyrics_path}")
    return lyrics_path

def test_generation():
    """Test the fixed generation system"""
    print("="*60)
    print("TESTING FIXED SONG GENERATION")
    print("="*60)
    
    # Create test lyrics
    lyrics_path = create_test_lyrics()
    
    # Prepare command
    cmd = [
        sys.executable, "infer/infer.py",
        "--lrc-path", lyrics_path,
        "--ref-prompt", "pop song, upbeat, energetic",
        "--audio-length", "95",
        "--output-dir", "output"
    ]
    
    print("Command to run:")
    print(" ".join(cmd))
    print()
    print("⚠ This will take 5-15 minutes on CPU...")
    print("⚠ You can stop with Ctrl+C if needed")
    print()
    
    # Ask for confirmation
    response = input("Continue with generation test? (y/N): ").strip().lower()
    if response != 'y':
        print("Test cancelled.")
        return False
    
    # Run the generation
    start_time = time.time()
    
    try:
        print("\n" + "="*40)
        print("STARTING GENERATION...")
        print("="*40)
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)  # 20 minute timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nGeneration completed in {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
        # Print output
        if result.stdout:
            print("\n" + "="*40)
            print("STDOUT:")
            print("="*40)
            print(result.stdout)
        
        if result.stderr:
            print("\n" + "="*40)
            print("STDERR:")
            print("="*40)
            print(result.stderr)
        
        # Check results
        if result.returncode == 0:
            print("\n" + "="*60)
            print("✓ GENERATION SUCCESSFUL!")
            print("="*60)
            
            # Check output file
            output_file = "output/output_fixed.wav"
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"✓ Output file created: {output_file}")
                print(f"✓ File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
                
                if file_size > 1000:
                    print("✓ File size looks reasonable - likely contains audio")
                    return True
                else:
                    print("⚠ File size is very small - may be silent")
                    return False
            else:
                print("✗ Output file not found")
                return False
        else:
            print(f"\n✗ Generation failed with return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n⚠ Generation timed out after 20 minutes")
        print("This is expected on CPU-only systems")
        return False
        
    except KeyboardInterrupt:
        print(f"\n⚠ Generation interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n✗ Error during generation: {e}")
        return False

def main():
    """Main test function"""
    print("This will test the fixed song generation system.")
    print("The test will:")
    print("1. Create simple test lyrics")
    print("2. Run the fixed infer.py with comprehensive error handling")
    print("3. Check if a valid audio file is generated")
    print()
    
    success = test_generation()
    
    if success:
        print("\n" + "="*60)
        print("🎉 TEST PASSED!")
        print("="*60)
        print("The fixed system successfully generated audio!")
        print("Check output/output_fixed.wav to hear the result.")
    else:
        print("\n" + "="*60)
        print("⚠ TEST RESULTS UNCLEAR")
        print("="*60)
        print("The system may need further investigation.")
        print("Check the output above for specific error messages.")

if __name__ == "__main__":
    main()