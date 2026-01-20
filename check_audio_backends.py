#!/usr/bin/env python3
"""
Check audio codec and backend compatibility
"""
import sys
import os

def check_torchaudio():
    """Check torchaudio installation and backends"""
    print("=" * 70)
    print("CHECKING TORCHAUDIO")
    print("=" * 70)
    
    try:
        import torchaudio
        print(f"✓ torchaudio {torchaudio.__version__} installed")
        
        # Check backends
        try:
            backends = torchaudio.list_audio_backends()
            print(f"✓ Available backends: {backends}")
        except AttributeError:
            print("⚠ Could not list backends")
        
        # Check if current backend is available
        try:
            current_backend = torchaudio.get_audio_backend()
            print(f"✓ Current backend: {current_backend}")
        except AttributeError:
            print("⚠ Could not determine current backend")
        
        return True
    except ImportError as e:
        print(f"✗ torchaudio not installed: {e}")
        return False

def check_scipy():
    """Check scipy.io.wavfile"""
    print("\n" + "=" * 70)
    print("CHECKING SCIPY")
    print("=" * 70)
    
    try:
        import scipy.io.wavfile as wavfile
        import scipy
        print(f"✓ scipy {scipy.__version__} installed")
        print(f"✓ scipy.io.wavfile available")
        
        # Test basic functionality
        import tempfile
        import numpy as np
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        try:
            # Create test audio
            test_audio = np.random.randint(-32768, 32767, (44100, 2), dtype=np.int16)
            
            # Write test audio
            wavfile.write(temp_path, 44100, test_audio)
            
            # Read back
            sr, audio = wavfile.read(temp_path)
            print(f"✓ WAV write/read test successful (shape: {audio.shape}, sr: {sr})")
            
            os.unlink(temp_path)
        except Exception as e:
            print(f"⚠ WAV test failed: {e}")
        
        return True
    except ImportError as e:
        print(f"⚠ scipy not installed: {e}")
        print("  Install with: pip install scipy")
        return False

def check_soundfile():
    """Check soundfile"""
    print("\n" + "=" * 70)
    print("CHECKING SOUNDFILE")
    print("=" * 70)
    
    try:
        import soundfile
        print(f"✓ soundfile {soundfile.__version__} installed")
        
        # Test basic functionality
        import tempfile
        import numpy as np
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        try:
            # Create test audio
            test_audio = np.random.random((44100, 2)).astype(np.float32)
            
            # Write test audio
            soundfile.write(temp_path, test_audio, 44100)
            
            # Read back
            audio, sr = soundfile.read(temp_path)
            print(f"✓ WAV write/read test successful (shape: {audio.shape}, sr: {sr})")
            
            os.unlink(temp_path)
        except Exception as e:
            print(f"⚠ WAV test failed: {e}")
        
        return True
    except ImportError as e:
        print(f"⚠ soundfile not installed: {e}")
        print("  Install with: pip install soundfile")
        return False

def check_numpy():
    """Check numpy"""
    print("\n" + "=" * 70)
    print("CHECKING NUMPY")
    print("=" * 70)
    
    try:
        import numpy as np
        print(f"✓ numpy {np.__version__} installed")
        return True
    except ImportError as e:
        print(f"✗ numpy not installed: {e}")
        return False

def check_torch():
    """Check torch"""
    print("\n" + "=" * 70)
    print("CHECKING TORCH")
    print("=" * 70)
    
    try:
        import torch
        print(f"✓ torch {torch.__version__} installed")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        print(f"✓ Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        return True
    except ImportError as e:
        print(f"✗ torch not installed: {e}")
        return False

def test_wav_saving():
    """Test WAV saving with all available methods"""
    print("\n" + "=" * 70)
    print("TESTING WAV SAVING WITH ALL METHODS")
    print("=" * 70)
    
    import tempfile
    import numpy as np
    import torch
    
    # Create test audio
    test_audio_int16 = torch.randint(-32768, 32767, (2, 44100), dtype=torch.int16)
    test_audio_float32 = torch.randn(2, 44100)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test 1: torchaudio
        print("\nTest 1: torchaudio with int16")
        try:
            import torchaudio
            path = os.path.join(temp_dir, "test_ta_int16.wav")
            torchaudio.save(path, test_audio_int16, sample_rate=44100)
            size = os.path.getsize(path)
            print(f"  ✓ Success ({size} bytes)")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        
        print("Test 2: torchaudio with float32")
        try:
            import torchaudio
            path = os.path.join(temp_dir, "test_ta_float32.wav")
            torchaudio.save(path, test_audio_float32, sample_rate=44100)
            size = os.path.getsize(path)
            print(f"  ✓ Success ({size} bytes)")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        
        # Test 3: scipy
        print("Test 3: scipy with int16")
        try:
            import scipy.io.wavfile as wavfile
            path = os.path.join(temp_dir, "test_scipy_int16.wav")
            wavfile.write(path, 44100, test_audio_int16.cpu().numpy().T.astype(np.int16))
            size = os.path.getsize(path)
            print(f"  ✓ Success ({size} bytes)")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        
        # Test 4: soundfile
        print("Test 4: soundfile with float32")
        try:
            import soundfile
            path = os.path.join(temp_dir, "test_sf_float32.wav")
            soundfile.write(path, test_audio_float32.cpu().numpy().T, 44100)
            size = os.path.getsize(path)
            print(f"  ✓ Success ({size} bytes)")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

def main():
    print("\n")
    print("*" * 70)
    print("AUDIO CODEC AND BACKEND COMPATIBILITY CHECK")
    print("*" * 70)
    print()
    
    results = {
        "torch": check_torch(),
        "numpy": check_numpy(),
        "torchaudio": check_torchaudio(),
        "scipy": check_scipy(),
        "soundfile": check_soundfile(),
    }
    
    test_wav_saving()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nInstalled libraries:")
    for lib, available in results.items():
        status = "✓" if available else "✗"
        print(f"  {status} {lib}")
    
    print("\nRecommendations:")
    if not results["torchaudio"]:
        print("  ✗ torchaudio is not installed")
    else:
        print("  ✓ torchaudio is installed (primary method)")
    
    if not results["scipy"]:
        print("  ⚠ scipy is not installed - install for fallback: pip install scipy")
    else:
        print("  ✓ scipy is installed (fallback method)")
    
    if not results["soundfile"]:
        print("  ⚠ soundfile is not installed - install for fallback: pip install soundfile")
    else:
        print("  ✓ soundfile is installed (fallback method)")
    
    print("\nFor audio saving to work reliably, you should have at least one of:")
    print("  - torchaudio (primary)")
    print("  - scipy (fallback)")
    print("  - soundfile (fallback)")
    
    if results["torchaudio"] and (results["scipy"] or results["soundfile"]):
        print("\n✓ GOOD: Multiple methods available")
    elif results["torchaudio"]:
        print("\n⚠ WARNING: Only torchaudio available, no fallbacks")
        print("  Recommended: pip install scipy soundfile")
    else:
        print("\n✗ ERROR: No audio saving methods available")
    
    print()

if __name__ == "__main__":
    main()
