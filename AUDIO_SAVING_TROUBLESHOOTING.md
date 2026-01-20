# Audio Saving Troubleshooting Guide

## Quick Diagnosis

If generation is hanging before producing output, it's likely due to audio saving issues.

### Step 1: Check Backend Compatibility
```bash
python check_audio_backends.py
```

This will show:
- Which audio libraries are installed
- Which backends are available
- Test results for each method

### Step 2: Install Fallback Methods
```bash
pip install scipy soundfile
```

### Step 3: Run Generation
```bash
python infer/infer.py \
  --lrc-path output/test.lrc \
  --ref-prompt "pop song" \
  --audio-length 95 \
  --output-dir output
```

---

## Common Issues and Solutions

### Issue 1: Generation Hangs Before "GENERATION COMPLETE"
**Symptom**: Hangs after "Generation completed in X minutes" message

**Cause**: `torchaudio.save()` hanging during WAV encoding

**Solution**:
```bash
# Install scipy and soundfile for fallback methods
pip install scipy soundfile

# Run again - will use fallback if torchaudio hangs
python infer/infer.py --lrc-path output/test.lrc --ref-prompt "pop song" --audio-length 95 --output-dir output
```

### Issue 2: "Audio saving timed out"
**Symptom**: Error message says "Audio saving timed out after 60 seconds"

**Cause**: WAV backend extremely slow or unavailable

**Solution**:
```bash
# Check which backends are available
python check_audio_backends.py

# If only torchaudio shows, install fallback methods
pip install scipy soundfile

# Run generation again
python infer/infer.py --lrc-path output/test.lrc --ref-prompt "pop song" --audio-length 95 --output-dir output
```

### Issue 3: "Failed to save audio with any available method"
**Symptom**: All saving methods failed

**Cause**: No working audio backends available

**Solution**:
```bash
# Check what's installed
python check_audio_backends.py

# Install missing backends
pip install scipy soundfile

# Verify installation
python -c "import scipy; print('scipy ok')"
python -c "import soundfile; print('soundfile ok')"

# Run again
python infer/infer.py --lrc-path output/test.lrc --ref-prompt "pop song" --audio-length 95 --output-dir output
```

---

## Understanding Audio Saving Methods

### Method 1: torchaudio (Primary)
**Pros**:
- Native PyTorch integration
- Fastest
- Handles multiple formats

**Cons**:
- Requires backend (libsndfile or FFmpeg)
- Can hang in WSL on slow systems
- May have version conflicts

**Used when**: Available and working

### Method 2: scipy.io.wavfile (Fallback 1)
**Pros**:
- Pure Python implementation
- Lightweight
- Works reliably in WSL

**Cons**:
- Only supports WAV
- Slightly slower than torchaudio
- Requires scipy installation

**Used when**: torchaudio times out or fails

### Method 3: soundfile (Fallback 2)
**Pros**:
- Modern audio library
- Works with many formats
- Good WSL support

**Cons**:
- Requires external installation
- Not in standard library

**Used when**: scipy unavailable or fails

---

## Detailed Troubleshooting Steps

### Step 1: Test Each Backend Individually

**Test torchaudio:**
```python
import torch
import torchaudio

# Create test audio
audio = torch.randn(2, 44100)  # 1 second of audio

# Try saving
torchaudio.save("test_ta.wav", audio, sample_rate=44100)
print("torchaudio: OK")
```

**Test scipy:**
```python
import numpy as np
import scipy.io.wavfile as wavfile

# Create test audio
audio = np.random.randint(-32768, 32767, (44100, 2), dtype=np.int16)

# Try saving
wavfile.write("test_scipy.wav", 44100, audio)
print("scipy: OK")
```

**Test soundfile:**
```python
import numpy as np
import soundfile

# Create test audio
audio = np.random.random((44100, 2)).astype(np.float32)

# Try saving
soundfile.write("test_sf.wav", audio, 44100)
print("soundfile: OK")
```

### Step 2: Check WSL-Specific Issues

**Verify WSL disk access:**
```bash
# Check output directory is accessible
ls -la output/

# Try creating a test file
touch output/test.txt && rm output/test.txt
echo "Disk access: OK"
```

**Check file system:**
```bash
# Check disk space
df -h /mnt/d/

# Check filesystem type
mount | grep "/mnt/d"
```

**Check permissions:**
```bash
# Test write permission
touch output/test_permission.txt
rm output/test_permission.txt
echo "Permissions: OK"
```

### Step 3: Monitor System Resources

**During generation, monitor:**
```bash
# In separate terminal
watch -n 1 'ps aux | grep python && free -h && df -h /mnt/d'
```

Check for:
- High memory usage
- Disk space filling up
- CPU throttling
- Slow I/O

---

## Installation Guide

### Install scipy
```bash
pip install scipy

# Verify
python -c "import scipy; print(f'scipy {scipy.__version__}')"
```

### Install soundfile
```bash
pip install soundfile

# If that fails, try:
pip install soundfile --no-binary :all:

# Verify
python -c "import soundfile; print(f'soundfile {soundfile.__version__}')"
```

### Install both
```bash
pip install scipy soundfile

# Verify
python -c "import scipy, soundfile; print('Both installed')"
```

---

## Expected Behavior After Fix

### Before (Hanging)
```
✓ Generation completed in 28.45 seconds (28.5 minutes)
✓ Generated 1 song(s)
Selected song tensor: shape=torch.Size([2, 4189488]), dtype=torch.int16
[HANGS HERE FOR SEVERAL MINUTES]
```

### After (Working)
```
✓ Generation completed in 28.45 seconds (28.5 minutes)
✓ Generated 1 song(s)
Selected song tensor: shape=torch.Size([2, 4189488]), dtype=torch.int16
   Saving audio to output/output_fixed.wav...
   Validating audio tensor...
     Shape: torch.Size([2, 4189488])
     Dtype: torch.int16
     ✓ Audio tensor is valid
   Method 1: Attempting save with torchaudio...
     ✓ Saved with torchaudio (8377952 bytes)
✓ Audio saved: output/output_fixed.wav
✓ File size: 8,377,952 bytes (7.99 MB)
✓ File size is reasonable for 95-second song

============================================================
GENERATION COMPLETE!
============================================================
```

---

## Testing After Fix

### Test 1: Quick Audio Save Test
```bash
python -c "
import torch
import torchaudio
import time

audio = torch.randint(-32768, 32767, (2, 4189488), dtype=torch.int16)
start = time.time()
torchaudio.save('test_output.wav', audio, sample_rate=44100)
elapsed = time.time() - start
print(f'Saved 95s audio in {elapsed:.1f}s')
"
```

### Test 2: Full Generation
```bash
python infer/infer.py \
  --lrc-path output/test.lrc \
  --ref-prompt "pop song, upbeat" \
  --audio-length 95 \
  --output-dir output
```

### Test 3: Verify Output
```bash
# Check file exists and has reasonable size
ls -lh output/output_fixed.wav

# Play the audio (on system with audio support)
# aplay output/output_fixed.wav
```

---

## Debug Output Interpretation

### Success Messages
```
✓ Audio tensor is valid          → Tensor format is correct
✓ Saved with torchaudio          → torchaudio worked
✓ Audio saved: output/output_fixed.wav  → File created
✓ File size is reasonable        → File size is normal
```

### Warning Messages
```
⚠ torchaudio save timed out      → torchaudio too slow, trying fallback
⚠ torchaudio failed              → torchaudio failed, trying fallback
⚠ scipy not available            → scipy not installed
```

### Error Messages
```
✗ Audio validation failed         → Tensor format wrong
✗ ERROR: Audio saving timed out   → All methods too slow
✗ Failed to save audio            → All methods failed
```

---

## Performance Expectations

### Audio Saving Time
- Ideal: < 1 second
- Acceptable: 1-5 seconds
- Slow: 5-30 seconds
- Hanging: > 30 seconds

### File Size
- 95s @ 44.1kHz stereo int16: ~8 MB
- 285s @ 44.1kHz stereo int16: ~25 MB

---

## Reporting Issues

If audio saving still fails after trying all these steps:

1. Run diagnostic:
   ```bash
   python check_audio_backends.py > audio_diagnostics.txt
   ```

2. Collect logs:
   ```bash
   python infer/infer.py \
     --lrc-path output/test.lrc \
     --ref-prompt "pop song" \
     --audio-length 95 \
     --output-dir output 2>&1 | tee generation.log
   ```

3. Include in bug report:
   - `audio_diagnostics.txt`
   - `generation.log`
   - `uname -a` (system info)
   - `python --version` (Python version)

---

## Summary

**If generation is hanging during audio saving:**

1. Run: `python check_audio_backends.py`
2. Install: `pip install scipy soundfile`
3. Run generation again

The new code will automatically try multiple saving methods and use the first one that works.

**Expected result**: Audio saves in < 5 seconds instead of hanging indefinitely.
