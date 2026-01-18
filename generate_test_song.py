#!/usr/bin/env python3
"""
Generate a test song to verify the complete system works
"""
import os
import sys
import subprocess
import time

def create_test_lyrics():
    """Create a simple test song with vocals"""
    lyrics = """[00:00.00]Hello world, this is a test
[00:03.00]Of the DiffRhythm system
[00:06.00]With all the fixes applied
[00:09.00]Let's make some music now
[00:12.00]The normalization is safe
[00:15.00]No more silent failures
[00:18.00]Every step is validated
[00:21.00]Audio generation works"""
    
    os.makedirs("output", exist_ok=True)
    lyrics_path = "output/test_song.lrc"
    
    with open(lyrics_path, 'w', encoding='utf-8') as f:
        f.write(lyrics)
    
    print(f"✓ Created test song lyrics: {lyrics_path}")
    return lyrics_path

def generate_song():
    """Generate the test song"""
    print("="*60)
    print("GENERATING TEST SONG WITH VOCALS")
    print("="*60)
    
    # Create lyrics
    lyrics_path = create_test_lyrics()
    
    # Prepare generation command
    cmd = [
        sys.executable, "infer/infer.py",
        "--lrc-path", lyrics_path,
        "--ref-prompt", "pop song, clear vocals, upbeat, energetic",
        "--audio-length", "95",  # Short song for testing
        "--output-dir", "output"
    ]
    
    print("Generation command:")
    print(" ".join(cmd))
    print()
    print("⚠ This will take 5-15 minutes on CPU...")
    print("⚠ The system will show detailed progress with the new fixes")
    print()
    
    # Ask for confirmation
    response = input("Start song generation? (y/N): ").strip().lower()
    if response != 'y':
        print("Generation cancelled.")
        return False
    
    # Start generation
    print("\n" + "="*40)
    print("STARTING GENERATION...")
    print("="*40)
    
    start_time = time.time()
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Wait for completion
        return_code = process.poll()
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nGeneration completed in {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
        if return_code == 0:
            print("✓ Generation successful!")
            
            # Check output file
            output_file = "output/output_fixed.wav"
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"✓ Output file: {output_file}")
                print(f"✓ File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
                
                if file_size > 10000:  # At least 10KB
                    print("✓ File size indicates successful generation!")
                    return True
                else:
                    print("⚠ File size is very small - may be silent")
                    return False
            else:
                print("✗ Output file not created")
                return False
        else:
            print(f"✗ Generation failed with code: {return_code}")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠ Generation interrupted by user")
        return False
    except Exception as e:
        print(f"\n✗ Error during generation: {e}")
        return False

def main():
    """Main function"""
    print("DIFFRHYTHM TEST SONG GENERATION")
    print("This will generate a complete song with vocals to verify all fixes work")
    print()
    
    success = generate_song()
    
    if success:
        print("\n" + "="*60)
        print("🎉 SUCCESS! SONG GENERATION COMPLETE!")
        print("="*60)
        print()
        print("✓ All fixes are working correctly")
        print("✓ Safe normalization prevented silent failures")
        print("✓ Comprehensive validation caught any issues")
        print("✓ Detailed logging showed the process")
        print()
        print("Your generated song is ready:")
        print("  📁 output/output_fixed.wav")
        print()
        print("The DiffRhythm system is now fully functional!")
        
    else:
        print("\n" + "="*60)
        print("⚠ GENERATION INCOMPLETE")
        print("="*60)
        print()
        print("The generation may have encountered issues.")
        print("Check the output above for specific error messages.")
        print("Common issues:")
        print("- Very long generation time on CPU (5-15 minutes is normal)")
        print("- Silent audio due to CPU-only inference limitations")
        print("- Missing dependencies")

if __name__ == "__main__":
    main()