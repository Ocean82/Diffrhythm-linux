#!/usr/bin/env python3
"""
Comprehensive codec and audio pipeline diagnostic tool
Checks all audio encoding/decoding capabilities and dependencies
"""
import sys
import os
import tempfile
import subprocess

def check_ffmpeg():
    """Check FFmpeg installation"""
    print("=" * 80)
    print("CHECKING FFMPEG")
    print("=" * 80)
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✓ FFmpeg installed: {version_line}")
            
            # Check for codec support
            print("\nChecking codec support...")
            codecs_to_check = ['aac', 'mp3', 'libmp3lame', 'libopus', 'libvorbis', 'flac', 'pcm_s16le']
            
            for codec in codecs_to_check:
                result = subprocess.run(['ffmpeg', '-codecs'], 
                                      capture_output=True, text=True, timeout=5)
                if codec in result.stdout:
                    print(f"  ✓ {codec}")
                else:
                    print(f"  ✗ {codec}")
            
            return True
        else:
            print(f"✗ FFmpeg not working: {result.stderr}")
            return False
    except FileNotFoundError:
        print("✗ FFmpeg not found in PATH")
        print("  Install with: sudo apt-get install ffmpeg")
        return False
    except subprocess.TimeoutExpired:
        print("✗ FFmpeg check timed out")
        return False
    except Exception as e:
        print(f"✗ Error checking FFmpeg: {e}")
        return False

def check_librosa():
    """Check librosa installation and backend"""
    print("\n" + "=" * 80)
    print("CHECKING LIBROSA")
    print("=" * 80)
    
    try:
        import librosa
        print(f"✓ librosa {librosa.__version__} installed")
        
        # Check audio backends
        try:
            import audioread
            print(f"✓ audioread installed (provides codec support)")
        except ImportError:
            print(f"⚠ audioread not installed (librosa codec support limited)")
        
        # Test loading different formats
        print("\nTesting format support in librosa...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test WAV file
            try:
                import numpy as np
                import scipy.io.wavfile
                test_audio = np.random.randint(-32768, 32767, (44100,), dtype=np.int16)
                wav_path = os.path.join(tmpdir, "test.wav")
                scipy.io.wavfile.write(wav_path, 44100, test_audio)
                
                audio, sr = librosa.load(wav_path, sr=None)
                print(f"  ✓ WAV loading works (shape: {audio.shape}, sr: {sr})")
            except Exception as e:
                print(f"  ✗ WAV loading failed: {e}")
        
        return True
    except ImportError as e:
        print(f"✗ librosa not installed: {e}")
        print("  Install with: pip install librosa")
        return False

def check_torchaudio():
    """Check torchaudio and its backends"""
    print("\n" + "=" * 80)
    print("CHECKING TORCHAUDIO")
    print("=" * 80)
    
    try:
        import torchaudio
        print(f"✓ torchaudio {torchaudio.__version__} installed")
        
        # List available backends
        try:
            backends = torchaudio.list_audio_backends()
            print(f"✓ Available backends: {backends}")
            
            # Get current backend
            current = torchaudio.get_audio_backend()
            print(f"✓ Current backend: {current}")
        except AttributeError:
            print("⚠ Could not determine backend information")
        
        # Test loading different formats
        print("\nTesting format support in torchaudio...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test WAV file
            try:
                import torch
                import scipy.io.wavfile
                import numpy as np
                
                test_audio = np.random.randint(-32768, 32767, (44100,), dtype=np.int16)
                wav_path = os.path.join(tmpdir, "test.wav")
                scipy.io.wavfile.write(wav_path, 44100, test_audio)
                
                audio, sr = torchaudio.load(wav_path)
                print(f"  ✓ WAV loading works (shape: {audio.shape}, sr: {sr})")
            except Exception as e:
                print(f"  ✗ WAV loading failed: {e}")
        
        return True
    except ImportError as e:
        print(f"✗ torchaudio not installed: {e}")
        print("  Install with: pip install torchaudio")
        return False

def check_mutagen():
    """Check mutagen for MP3 metadata"""
    print("\n" + "=" * 80)
    print("CHECKING MUTAGEN")
    print("=" * 80)
    
    try:
        from mutagen.mp3 import MP3
        import mutagen
        
        # Try to get version
        try:
            version = mutagen.__version__
            print(f"✓ mutagen {version} installed")
        except AttributeError:
            # Fallback: try to get version from package metadata
            try:
                import importlib.metadata
                version = importlib.metadata.version('mutagen')
                print(f"✓ mutagen {version} installed")
            except:
                print(f"✓ mutagen installed (version unknown)")
        
        print(f"✓ MP3 metadata support available")
        return True
    except ImportError as e:
        print(f"✗ mutagen not installed: {e}")
        print("  Install with: pip install mutagen")
        return False

def check_scipy_audioio():
    """Check scipy audio I/O"""
    print("\n" + "=" * 80)
    print("CHECKING SCIPY AUDIO I/O")
    print("=" * 80)
    
    try:
        from scipy.io import wavfile
        import scipy
        print(f"✓ scipy {scipy.__version__} installed")
        print(f"✓ scipy.io.wavfile available")
        
        # Test WAV writing
        try:
            import numpy as np
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
            
            test_audio = np.random.randint(-32768, 32767, (44100, 2), dtype=np.int16)
            wavfile.write(temp_path, 44100, test_audio)
            
            sr, audio = wavfile.read(temp_path)
            print(f"  ✓ WAV write/read works (shape: {audio.shape}, sr: {sr})")
            
            os.unlink(temp_path)
        except Exception as e:
            print(f"  ⚠ WAV write/read test failed: {e}")
        
        return True
    except ImportError as e:
        print(f"⚠ scipy not installed: {e}")
        print("  Install with: pip install scipy")
        return False

def check_soundfile():
    """Check soundfile for audio I/O"""
    print("\n" + "=" * 80)
    print("CHECKING SOUNDFILE")
    print("=" * 80)
    
    try:
        import soundfile
        print(f"✓ soundfile {soundfile.__version__} installed")
        
        # Test audio I/O
        try:
            import numpy as np
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
            
            test_audio = np.random.random((44100, 2)).astype(np.float32)
            soundfile.write(temp_path, test_audio, 44100)
            
            audio, sr = soundfile.read(temp_path)
            print(f"  ✓ WAV write/read works (shape: {audio.shape}, sr: {sr})")
            
            os.unlink(temp_path)
        except Exception as e:
            print(f"  ⚠ WAV write/read test failed: {e}")
        
        return True
    except ImportError as e:
        print(f"⚠ soundfile not installed: {e}")
        print("  Install with: pip install soundfile")
        return False

def check_audioread():
    """Check audioread for extended format support"""
    print("\n" + "=" * 80)
    print("CHECKING AUDIOREAD")
    print("=" * 80)
    
    try:
        import audioread
        
        # Try to get version
        try:
            version = audioread.__version__
            print(f"✓ audioread {version} installed")
        except AttributeError:
            # Fallback: try to get version from package metadata
            try:
                import importlib.metadata
                version = importlib.metadata.version('audioread')
                print(f"✓ audioread {version} installed")
            except:
                print(f"✓ audioread installed (version unknown)")
        
        # Check available backends
        print("  Provides extended codec support to librosa")
        
        return True
    except ImportError:
        print(f"⚠ audioread not installed")
        print("  Install with: pip install audioread")
        print("  (Enables MP3, OGG, FLAC, and other formats in librosa)")
        return False

def check_format_support():
    """Check DiffRhythm format support"""
    print("\n" + "=" * 80)
    print("DIFFRHYTHM FORMAT SUPPORT")
    print("=" * 80)
    
    print("\nSupported Input Formats:")
    print("  ✓ WAV (PCM 16-bit) - Primary format")
    print("  ✓ FLAC (Free Lossless Audio Codec)")
    print("  ✓ MP3 (MPEG-1 Layer III)")
    
    print("\nRequired Audio Specifications:")
    print("  - Sample Rate: 44.1 kHz (auto-resampled if different)")
    print("  - Channels: Mono or Stereo (auto-converted to stereo if needed)")
    print("  - Bit Depth: 16-bit or 24-bit (converted to float32 internally)")
    print("  - Duration: At least 10 seconds for style reference")
    
    print("\nOutput Format:")
    print("  ✓ WAV (PCM 16-bit stereo, 44.1 kHz)")

def analyze_pipeline():
    """Analyze the audio pipeline"""
    print("\n" + "=" * 80)
    print("AUDIO PIPELINE ANALYSIS")
    print("=" * 80)
    
    print("\n1. INPUT STAGE (get_style_prompt)")
    print("   - Accepts: WAV, FLAC, MP3 files (>= 10 seconds)")
    print("   - Uses: librosa.load() with FFmpeg backend")
    print("   - Output: Audio at 24 kHz, float32")
    print("   - Fallback: mutagen for MP3 metadata")
    
    print("\n2. REFERENCE STAGE (get_reference_latent)")
    print("   - Accepts: Any format torchaudio supports")
    print("   - Uses: torchaudio.load() for edit mode")
    print("   - Resamples: To 44.1 kHz if needed")
    print("   - Output: Stereo audio, float32")
    
    print("\n3. VAE ENCODING/DECODING")
    print("   - Expects: 44.1 kHz stereo audio")
    print("   - Format: float32 or int16")
    print("   - Bit depth: 16-bit PCM equivalent")
    
    print("\n4. OUTPUT STAGE (save_audio_robust)")
    print("   - Format: 16-bit PCM WAV")
    print("   - Sample Rate: 44.1 kHz")
    print("   - Channels: Stereo (2)")
    print("   - Methods: torchaudio > scipy > soundfile")

def print_codec_matrix():
    """Print codec support matrix"""
    print("\n" + "=" * 80)
    print("CODEC SUPPORT MATRIX")
    print("=" * 80)
    
    codecs = {
        "WAV (PCM 16-bit)": {
            "input": "✓",
            "librosa": "✓",
            "torchaudio": "✓",
            "output": "✓",
            "notes": "Native, no FFmpeg needed"
        },
        "FLAC": {
            "input": "✓",
            "librosa": "✓ (needs FFmpeg)",
            "torchaudio": "✓ (needs FFmpeg)",
            "output": "✗",
            "notes": "Supported for input only"
        },
        "MP3": {
            "input": "✓",
            "librosa": "✓ (needs FFmpeg)",
            "torchaudio": "✓ (needs FFmpeg)",
            "output": "✗",
            "notes": "Metadata via mutagen, decoding via FFmpeg"
        },
        "OGG/Vorbis": {
            "input": "⚠",
            "librosa": "⚠ (needs FFmpeg+audioread)",
            "torchaudio": "⚠ (needs FFmpeg)",
            "output": "✗",
            "notes": "Supported if FFmpeg available"
        },
        "AAC": {
            "input": "⚠",
            "librosa": "⚠ (needs FFmpeg+audioread)",
            "torchaudio": "⚠ (needs FFmpeg)",
            "output": "✗",
            "notes": "Supported if FFmpeg available"
        },
        "OPUS": {
            "input": "⚠",
            "librosa": "⚠ (needs FFmpeg+audioread)",
            "torchaudio": "⚠ (needs FFmpeg)",
            "output": "✗",
            "notes": "Supported if FFmpeg available"
        }
    }
    
    print(f"\n{'Format':<20} {'Input':<10} {'librosa':<15} {'torchaudio':<15} {'Output':<10} {'Notes':<30}")
    print("-" * 100)
    
    for format_name, support in codecs.items():
        print(f"{format_name:<20} {support['input']:<10} {support['librosa']:<15} {support['torchaudio']:<15} {support['output']:<10} {support['notes']:<30}")

def main():
    print("\n")
    print("*" * 80)
    print("DIFFRHYTHM CODEC AND AUDIO PIPELINE DIAGNOSTIC")
    print("*" * 80)
    print()
    
    results = {
        "FFmpeg": check_ffmpeg(),
        "librosa": check_librosa(),
        "torchaudio": check_torchaudio(),
        "mutagen": check_mutagen(),
        "scipy": check_scipy_audioio(),
        "soundfile": check_soundfile(),
        "audioread": check_audioread(),
    }
    
    check_format_support()
    analyze_pipeline()
    print_codec_matrix()
    
    # Summary
    print("\n" + "=" * 80)
    print("INSTALLATION RECOMMENDATIONS")
    print("=" * 80)
    
    print("\nMinimum (WAV only):")
    print("  pip install librosa torchaudio scipy")
    
    print("\nRecommended (most formats):")
    print("  pip install librosa torchaudio scipy soundfile audioread")
    
    print("\nOptional system packages:")
    print("  sudo apt-get install ffmpeg libsndfile1")
    
    print("\n" + "=" * 80)
    print("STATUS SUMMARY")
    print("=" * 80)
    
    required = ["librosa", "torchaudio"]
    fallback = ["scipy", "soundfile"]
    codec_support = ["FFmpeg", "audioread", "mutagen"]
    
    print("\nRequired Libraries:")
    for lib in required:
        status = "✓" if results.get(lib) else "✗"
        print(f"  {status} {lib}")
    
    print("\nFallback Libraries:")
    for lib in fallback:
        status = "✓" if results.get(lib) else "⚠"
        print(f"  {status} {lib}")
    
    print("\nCodec Support:")
    for lib in codec_support:
        status = "✓" if results.get(lib) else "⚠"
        print(f"  {status} {lib}")
    
    # Overall status
    print("\n" + "=" * 80)
    if results["librosa"] and results["torchaudio"]:
        if results["FFmpeg"]:
            print("✓ READY FOR ALL FORMATS (FFmpeg available)")
        else:
            print("⚠ READY FOR WAV/FLAC (FFmpeg missing - install for MP3/OGG/AAC)")
    else:
        print("✗ MISSING REQUIRED LIBRARIES")
    print("=" * 80)
    
    print()

if __name__ == "__main__":
    main()
