# DiffRhythm Audio Codec Compatibility Investigation

## Issue Description

Song generation appears to hang before output, likely during or after the audio saving phase. This could be due to:
1. Codec compatibility issues
2. WAV format problems
3. torchaudio backend issues in WSL
4. Tensor shape/dtype incompatibilities
5. Audio library conflicts

---

## Audio Pipeline Analysis

### Current Pipeline (infer.py)

```
CFM Sampling → VAE Decoding → Rearrange → Normalize → torchaudio.save()
                                          ↓
                              [int16, shape=(channels, samples)]
                                          ↓
                                  output_fixed.wav
```

### Potential Issues

#### 1. Tensor Shape Issue (Line 212)
```python
output = rearrange(output, "b d n -> d (b n)")
```
- Expected output from VAE: [batch=1, channels=2, samples=N]
- After rearrange: [channels=2, samples=N]
- **Issue**: If batch > 1, this could create very large tensors

#### 2. Audio Format Issue (Line 433)
```python
torchaudio.save(output_path, generated_song, sample_rate=44100)
```
- torchaudio uses backend (libsndfile or FFmpeg)
- WAV saving can hang if:
  - Tensor dtype is incompatible (should be float32 or int16)
  - Shape is wrong (should be [channels, samples])
  - Backend is not available in WSL

#### 3. Normalization Issue (Line 71)
```python
normalized = normalized.clamp(-1, 1).mul(32767).to(torch.int16)
```
- Converting to int16 for WAV saving
- **Issue**: int16 range is -32768 to 32767, multiplying by 32767 might overflow

#### 4. Backend Issues in WSL
- torchaudio might use FFmpeg which could hang
- No GUI available, might block on resource allocation
- Slow disk I/O in WSL

---

## Diagnostic Checks

### Check 1: Audio Tensor Properties
```python
# Should verify:
- Tensor dtype: float32 or int16
- Tensor shape: [channels, samples]
- Tensor values: -1 to 1 (float) or -32768 to 32767 (int16)
- Tensor size: reasonable (95s @ 44100Hz = ~4.2M samples)
```

### Check 2: Backend Availability
```python
# torchaudio backends:
- libsndfile: native WAV support
- sox: audio effects
- ffmpeg: codec support
```

### Check 3: File I/O
```python
# Verify:
- Output directory exists and is writable
- Sufficient disk space
- File permissions
```

---

## Identified Problems and Solutions

### Problem 1: torchaudio.save() Hanging

**Cause**: WAV encoding in WSL with FFmpeg backend might be slow or hang

**Solution**: Use numpy/scipy for WAV saving instead
```python
import scipy.io.wavfile as wavfile
wavfile.write(output_path, 44100, audio_np)
```

### Problem 2: Incorrect Audio Dtype

**Cause**: Audio might be wrong dtype for saving

**Solution**: Explicitly ensure float32 before saving
```python
audio_float = generated_song.to(torch.float32)
# Normalize to -1 to 1 range
audio_float = audio_float / 32768.0
torchaudio.save(output_path, audio_float, sample_rate=44100)
```

### Problem 3: Tensor Shape Mismatch

**Cause**: Shape might be [samples, channels] instead of [channels, samples]

**Solution**: Verify and transpose if needed
```python
if audio.shape[0] > audio.shape[1]:
    audio = audio.T  # Transpose if needed
```

### Problem 4: Memory Pressure

**Cause**: Large int16 tensor holding memory, slow disk write

**Solution**: Write in chunks or use memory mapping

---

## Recommended Fixes

### Fix 1: Add Audio Format Validation
```python
def validate_audio_tensor(audio, expected_shape_dim=2):
    """Validate audio tensor before saving"""
    # Check dtype
    if audio.dtype not in [torch.float32, torch.int16]:
        raise ValueError(f"Invalid dtype: {audio.dtype}")
    
    # Check shape
    if audio.dim() != expected_shape_dim:
        raise ValueError(f"Invalid shape dims: {audio.dim()}")
    
    # Check size
    if audio.shape[0] > audio.shape[1]:  # More channels than samples
        return audio.T
    
    return audio
```

### Fix 2: Use Alternative Save Method
```python
def save_audio_scipy(audio, output_path, sample_rate=44100):
    """Save audio using scipy instead of torchaudio"""
    import scipy.io.wavfile as wavfile
    
    # Convert to numpy
    audio_np = audio.cpu().numpy()
    
    # Ensure int16 range
    if audio_np.dtype == np.float32:
        audio_np = (audio_np * 32767).astype(np.int16)
    
    # Transpose if needed
    if audio_np.shape[0] > audio_np.shape[1]:
        audio_np = audio_np.T
    
    # Write
    wavfile.write(output_path, sample_rate, audio_np)
```

### Fix 3: Add Timeout Protection
```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Audio saving timed out after 60 seconds")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(60)  # 60 second timeout

try:
    torchaudio.save(output_path, generated_song, sample_rate=44100)
finally:
    signal.alarm(0)  # Cancel alarm
```

### Fix 4: Fallback Chain
```python
def save_audio_robust(audio, output_path, sample_rate=44100):
    """Save audio with fallback methods"""
    
    # Method 1: Try torchaudio (fastest)
    try:
        print("Attempting save with torchaudio...")
        torchaudio.save(output_path, audio, sample_rate=sample_rate)
        print("✓ Successfully saved with torchaudio")
        return True
    except Exception as e:
        print(f"⚠ torchaudio failed: {e}")
    
    # Method 2: Try scipy
    try:
        print("Attempting save with scipy...")
        import scipy.io.wavfile as wavfile
        audio_np = audio.cpu().numpy()
        if audio_np.dtype == np.float32:
            audio_np = (audio_np * 32767).astype(np.int16)
        wavfile.write(output_path, sample_rate, audio_np)
        print("✓ Successfully saved with scipy")
        return True
    except Exception as e:
        print(f"⚠ scipy failed: {e}")
    
    # Method 3: Try soundfile
    try:
        print("Attempting save with soundfile...")
        import soundfile as sf
        audio_np = audio.cpu().numpy()
        sf.write(output_path, audio_np.T, sample_rate)
        print("✓ Successfully saved with soundfile")
        return True
    except Exception as e:
        print(f"⚠ soundfile failed: {e}")
    
    raise RuntimeError("All save methods failed")
```

---

## Implementation Plan

### Step 1: Add Audio Validation
- Validate tensor shape before saving
- Check dtype compatibility
- Print audio statistics for debugging

### Step 2: Add Timeout Protection
- Set 60-second timeout on save operation
- Provide clear error if timeout occurs
- Allow user to interrupt

### Step 3: Implement Fallback Methods
- Try torchaudio first
- Fall back to scipy if needed
- Fall back to soundfile as last resort

### Step 4: Test Each Method
- Verify each save method works
- Check output file integrity
- Verify audio playback

---

## Dependencies Check

### Required for Default Method
```
torchaudio - already installed
```

### Required for Fallback Methods
```
scipy - audio.io.wavfile for WAV
soundfile - advanced audio I/O
numpy - array operations
```

### Check Installation
```bash
python -c "import scipy; print(scipy.__version__)"
python -c "import soundfile; print(soundfile.__version__)"
python -c "import numpy; print(numpy.__version__)"
```

---

## Testing Procedure

### Test 1: Verify Audio Tensor
```python
# Generate dummy audio tensor
audio = torch.randn(2, 4410000).to(torch.int16)  # 100 seconds, 44.1kHz

# Try saving
save_audio_robust(audio, "test_output.wav", 44100)

# Verify file
print(f"File size: {os.path.getsize('test_output.wav')} bytes")
```

### Test 2: Check Each Backend
```bash
python -c "import torchaudio; print(torchaudio.list_audio_backends())"
```

### Test 3: Real Generation Test
```bash
python infer/infer.py \
  --lrc-path output/test.lrc \
  --ref-prompt "pop song" \
  --audio-length 95 \
  --output-dir output
```

---

## Expected Outcomes

### Before Fix
- Generation hangs during audio saving
- No output file created
- Process appears frozen
- No clear error message

### After Fix
- Audio saves successfully in <5 seconds
- Output file created and valid
- Clear progress messages
- Works even if preferred backend unavailable

---

## Codec Compatibility Matrix

| Codec | Format | torchaudio | scipy | soundfile | WSL |
|-------|--------|-----------|-------|-----------|-----|
| PCM | WAV | ✓ | ✓ | ✓ | ✓ |
| FLAC | FLAC | ✓ | ✗ | ✓ | ✓ |
| MP3 | MP3 | ✓ | ✗ | ✗ | ? |
| OGG | OGG | ✓ | ✗ | ✓ | ? |

**Recommendation**: Use WAV with PCM codec for maximum compatibility

---

## Summary

**Root Cause**: Likely torchaudio.save() hanging due to backend issues in WSL

**Solution**: Implement robust multi-method audio saving with fallbacks

**Expected Result**: Audio saves reliably in <5 seconds regardless of backend

---

## Next Steps

1. Implement save_audio_robust() function
2. Add audio tensor validation
3. Add timeout protection
4. Test each backend
5. Update infer.py to use new saving method
