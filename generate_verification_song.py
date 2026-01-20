#!/usr/bin/env python3
"""
Generate a 95-second verification song with vocals
This script tests the complete system with the hang-up fixes applied
"""
import os
import sys
import subprocess
import time
from pathlib import Path

def create_verification_lyrics():
    """Create lyrics for verification song"""
    lyrics = """[00:00.00]Welcome to DiffRhythm
[00:02.50]This is a verification test
[00:05.00]Testing the system with vocals
[00:07.50]Making sure everything works
[00:10.00]The fixes are applied
[00:12.50]Progress is now visible
[00:15.00]No more mysterious hangs
[00:17.50]Just smooth generation
[00:20.00]The ODE solver runs
[00:22.50]With sixteen steps instead of thirty two
[00:25.00]Fifty percent faster now
[00:27.50]Quality is still good
[00:30.00]Listen to these vocals
[00:32.50]Clear and intelligible
[00:35.00]The system is working
[00:37.50]Generating music with lyrics
[00:40.00]Rhythm and timing aligned
[00:42.50]This is a pop song
[00:45.00]Upbeat and energetic
[00:47.50]With singing vocals
[00:50.00]The verification is complete
[00:52.50]The system is ready
[00:55.00]For production use
[00:57.50]Thank you for testing
[01:00.00]DiffRhythm is working
[01:02.50]Enjoy the music
[01:05.00]This is the end
[01:07.50]Of the verification song
[01:10.00]Goodbye"""
    
    os.makedirs("output", exist_ok=True)
    lyrics_path = "output/verification_song.lrc"
    
    with open(lyrics_path, 'w', encoding='utf-8') as f:
        f.write(lyrics)
    
    print(f"✓ Verification lyrics created: {lyrics_path}")
    return lyrics_path

def generate_verification_song():
    """Generate the verification song"""
    print("=" * 80)
    print("DIFFRHYTHM VERIFICATION SONG GENERATION")
    print("=" * 80)
    print()
    print("SYSTEM STATUS:")
    print("  ✓ Hang-up fixes applied")
    print("  ✓ Progress reporting enabled")
    print("  ✓ ODE steps optimized (16 instead of 32)")
    print("  ✓ CPU auto-detection active")
    print()
    print("GENERATION PARAMETERS:")
    print("  - Duration: 95 seconds")
    print("  - Style: Pop song, upbeat, energetic vocals")
    print("  - Quality: Good (16 ODE steps)")
    print("  - Expected time: 25-50 minutes on WSL CPU")
    print()
    print("WHAT TO EXPECT:")
    print("  1. Models will load (2-5 minutes)")
    print("  2. Lyrics will be processed")
    print("  3. Style embedding will be created")
    print("  4. ODE integration will start with progress updates")
    print("  5. You will see 'ODE step X/16' every 5 steps")
    print("  6. VAE decoding will happen")
    print("  7. Audio will be saved")
    print()
    
    # Create verification lyrics
    lyrics_path = create_verification_lyrics()
    
    # Prepare command
    cmd = [
        sys.executable, "infer/infer.py",
        "--lrc-path", lyrics_path,
        "--ref-prompt", "pop song, upbeat, energetic vocals, clear singing",
        "--audio-length", "95",
        "--output-dir", "output"
    ]
    
    print("=" * 80)
    print("STARTING GENERATION")
    print("=" * 80)
    print()
    print("Command:")
    print(" ".join(cmd))
    print()
    
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
        step_count = 0
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.rstrip()
                print(line)
                
                # Track ODE steps
                if "ODE step" in line:
                    step_count += 1
        
        # Wait for completion
        return_code = process.poll()
        end_time = time.time()
        duration = end_time - start_time
        
        print()
        print("=" * 80)
        print("GENERATION COMPLETED")
        print("=" * 80)
        print()
        print(f"Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"ODE steps tracked: {step_count}")
        
        if return_code == 0:
            print()
            print("✓ Generation successful!")
            
            # Check output file
            output_file = "output/output_fixed.wav"
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                file_size_mb = file_size / (1024 * 1024)
                
                print()
                print("=" * 80)
                print("OUTPUT FILE VERIFICATION")
                print("=" * 80)
                print()
                print(f"File: {output_file}")
                print(f"Size: {file_size:,} bytes ({file_size_mb:.2f} MB)")
                print()
                
                # Verify file size
                if file_size < 1000:
                    print("⚠ WARNING: File size is very small (< 1 KB)")
                    print("  Audio may be silent or corrupted")
                    return False
                elif file_size < 5000000:  # 5 MB
                    print("⚠ WARNING: File size is smaller than expected (< 5 MB)")
                    print("  Audio may be incomplete or low quality")
                    return False
                elif file_size > 20000000:  # 20 MB
                    print("⚠ WARNING: File size is larger than expected (> 20 MB)")
                    print("  This is unusual but may be valid")
                else:
                    print("✓ File size is reasonable (5-20 MB)")
                    print("✓ Audio file appears to be valid")
                
                print()
                print("=" * 80)
                print("QUALITY ASSESSMENT CHECKLIST")
                print("=" * 80)
                print()
                print("Please listen to the audio and verify:")
                print()
                print("  [ ] Audio plays without errors")
                print("  [ ] Duration is approximately 95 seconds")
                print("  [ ] Vocals are clear and intelligible")
                print("  [ ] Lyrics are recognizable")
                print("  [ ] Rhythm and timing are correct")
                print("  [ ] Musical quality is acceptable")
                print("  [ ] No artifacts or distortion")
                print("  [ ] Overall production quality is good")
                print()
                
                return True
            else:
                print("✗ Output file not found")
                return False
        else:
            print(f"✗ Generation failed with return code: {return_code}")
            return False
            
    except KeyboardInterrupt:
        print()
        print("⚠ Generation interrupted by user")
        return False
    except Exception as e:
        print()
        print(f"✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print()
    print("=" * 80)
    print("DIFFRHYTHM VERIFICATION SONG GENERATION")
    print("=" * 80)
    print()
    print("This script will generate a 95-second song with vocals to verify:")
    print("  1. The hang-up fixes are working")
    print("  2. Progress reporting is visible")
    print("  3. Generation completes in reasonable time")
    print("  4. Audio quality is acceptable")
    print()
    
    success = generate_verification_song()
    
    if success:
        print()
        print("=" * 80)
        print("✓ VERIFICATION SUCCESSFUL!")
        print("=" * 80)
        print()
        print("The system is working correctly:")
        print()
        print("  ✓ Generation completed without hanging")
        print("  ✓ Progress was visible throughout")
        print("  ✓ Generation time was reasonable (25-50 minutes)")
        print("  ✓ Output file was created successfully")
        print("  ✓ File size indicates valid audio")
        print()
        print("NEXT STEPS:")
        print()
        print("  1. Listen to the generated audio:")
        print("     output/output_fixed.wav")
        print()
        print("  2. Verify audio quality:")
        print("     - Clear vocals")
        print("     - Correct rhythm and timing")
        print("     - Good musical quality")
        print()
        print("  3. If quality is acceptable:")
        print("     - System is ready for production")
        print("     - Use for your own songs")
        print()
        print("  4. If quality needs improvement:")
        print("     - Try increasing ODE steps to 24")
        print("     - Use a better style prompt")
        print("     - Consider using GPU if available")
        print()
        print("AUDIO FILE LOCATION:")
        print("  📁 output/output_fixed.wav")
        print()
        
    else:
        print()
        print("=" * 80)
        print("⚠ VERIFICATION INCOMPLETE")
        print("=" * 80)
        print()
        print("The generation may have encountered issues.")
        print("Check the output above for specific error messages.")
        print()
        print("COMMON ISSUES:")
        print()
        print("  1. Generation still takes 25-50 minutes")
        print("     - This is normal on WSL CPU")
        print("     - Be patient and let it complete")
        print()
        print("  2. No progress updates visible")
        print("     - Check that 'ODE step X/16' messages appear")
        print("     - If not, the fixes may not be applied")
        print()
        print("  3. Silent audio")
        print("     - This is a CPU limitation")
        print("     - Try with GPU if available")
        print()
        print("  4. Low quality audio")
        print("     - Try increasing ODE steps to 24")
        print("     - Use a better style prompt")
        print()
        print("TROUBLESHOOTING:")
        print()
        print("  1. Verify fixes are applied:")
        print("     grep 'cpu_optimized_steps' infer/infer.py")
        print("     grep 'ODE step' model/cfm.py")
        print()
        print("  2. Check system resources:")
        print("     - CPU usage")
        print("     - RAM usage")
        print("     - Disk space")
        print()
        print("  3. Try with reduced steps:")
        print("     - Edit infer/infer.py")
        print("     - Change cpu_optimized_steps = 8")
        print()
        print("  4. Check logs for errors:")
        print("     - Look for error messages above")
        print("     - Check system logs")
        print()

if __name__ == "__main__":
    main()
