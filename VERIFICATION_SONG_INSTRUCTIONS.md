# Generate 95-Second Verification Song - Complete Instructions

## 🎯 Objective

Generate a complete 95-second song with vocals to verify the entire DiffRhythm system works correctly end-to-end.

---

## ✅ Pre-Generation Checklist

Before running generation, ensure:

### 1. Check Codec Support
```bash
python check_codec_pipeline.py
```

**Expected output:**
```
✓ librosa X.X.X installed
✓ torchaudio X.X.X installed
✓ scipy X.X.X installed
✓ FFmpeg installed
✓ Available backends: [list]
✓ Current backend: sox_io (or similar)
```

### 2. Verify Required Packages
```bash
python -c "import librosa, torchaudio, torch, scipy; print('All packages OK')"
```

**Expected output:**
```
All packages OK
```

### 3. Check Disk Space
```bash
df -h /mnt/d
```

**Required:** At least 5 GB free space (for models + output)

### 4. Check Memory
```bash
free -h
```

**Required:** At least 8 GB available RAM

---

## 🚀 Generate 95-Second Song

### Step 1: Run Verification Script
```bash
python generate_verification_95s_song.py
```

### Step 2: Confirm Prerequisites Check

The script will:
- ✓ Check all required libraries (librosa, torchaudio, torch, scipy)
- ✓ Check optional libraries (soundfile, audioread, mutagen)
- ✓ Exit with error if anything missing

**If error occurs:**
```bash
pip install librosa torchaudio scipy soundfile audioread mutagen
```

### Step 3: Create Test Lyrics

The script will:
- ✓ Create 24 lines of test lyrics
- ✓ Save to `output/verification_95s.lrc`
- ✓ Confirm duration ~95 seconds

**Example lyrics:**
```
[00:00.00]Welcome to DiffRhythm
[00:02.50]This is a verification test
[00:05.00]Testing song generation with vocals
...
[01:05.00]Verification complete
```

### Step 4: Start Generation

When prompted:
```
Start generation? (y/N):
```

**Type:** `y` and press Enter

### Step 5: Monitor Progress

The script will show real-time output with:

**First 5 minutes (Model Loading):**
```
DEBUG: Using cache directory: ./pretrained
DEBUG: Preparing CFM model...
DEBUG: Initializing DiT model...
DEBUG: Loading CFM checkpoint...
DEBUG: Preparing MuQMuLan...
DEBUG: All models prepared successfully.
```

**Next 30-50 minutes (ODE Sampling):**
```
Starting ODE integration with 16 steps...
Progress will be shown every 5 steps
     ODE step 5/16
     ODE step 10/16
     ODE step 15/16
     ODE step 16/16
```

**Next 5-10 minutes (VAE Decoding):**
```
Processing latent 1/1...
Latent prepared for VAE: shape=torch.Size([1, 128, 2048])
✓ VAE decode successful: shape=torch.Size([1, 2, 4189248])
```

**Finally (Audio Saving):**
```
Saving audio to output/output_fixed.wav...
✓ Audio tensor is valid
Method 1: Attempting save with torchaudio...
✓ Saved with torchaudio (8377952 bytes)
✓ Audio saved: output/output_fixed.wav
```

---

## ⏱️ Timeline Expectations

### Total Generation Time

| Component | Estimated Time |
|-----------|-----------------|
| Codec check | < 1 minute |
| Lyrics creation | < 1 minute |
| Model loading | 2-5 minutes |
| Lyrics processing | 1 minute |
| Style embedding | 1-2 minutes |
| ODE sampling | 15-30 minutes |
| VAE decoding | 5-10 minutes |
| Audio saving | < 5 seconds |
| **Total** | **25-50 minutes** |

### What to Watch For

- First 10 minutes: Initial setup and model loading
- Next 20-40 minutes: ODE sampling (main computation)
- Last 5-10 minutes: VAE decoding and audio saving
- Should complete in 25-50 minutes on WSL CPU

---

## 📊 Output Verification

### Check File Was Created
```bash
ls -lh output/output_fixed.wav
```

**Expected:**
```
-rw-r--r-- 1 user group 8.4M Jan 18 12:34 output/output_fixed.wav
```

**File size range:** 8-15 MB (95-second stereo 16-bit WAV)

### Check File Can Be Played
```bash
# On system with audio player:
aplay output/output_fixed.wav
# or
ffplay output/output_fixed.wav
# or open in audio editor
```

### Check Audio Properties
```bash
ffprobe output/output_fixed.wav
```

**Expected:**
```
Duration: 00:01:35.00 (95 seconds)
Channel(s): 2 (stereo)
Sample rate: 44100 Hz
Bit depth: 16-bit
Format: PCM
```

---

## 🎵 Quality Assessment

Listen to `output/output_fixed.wav` and check:

### Audio Presence
- [ ] Audio plays (not silent)
- [ ] Duration is ~95 seconds
- [ ] No static or noise at start/end

### Vocal Quality
- [ ] Vocals are present
- [ ] Singing is intelligible
- [ ] Lyrics are recognizable
- [ ] Vocal tone is natural

### Music Quality
- [ ] Background music is present
- [ ] Rhythm is consistent
- [ ] Timing is correct
- [ ] Musical quality is acceptable

### Technical Quality
- [ ] No artifacts or distortion
- [ ] No glitches or clicks
- [ ] Audio levels are good (not too quiet/loud)
- [ ] Stereo balance is correct

---

## ⚠️ Possible Issues & Solutions

### Issue 1: Audio is Silent
**Symptom:** File plays but contains no audio

**Causes:**
- CPU-only inference limitation
- Poor style prompt
- Model issues

**Solutions:**
1. Try different style prompt: `"pop song with powerful vocals"`
2. Check lyrics make sense
3. Try with GPU if available

### Issue 2: Very Small File (<1 MB)
**Symptom:** File size too small for 95-second song

**Causes:**
- Audio generation failed silently
- VAE decoding issue
- Model inference failed

**Solutions:**
1. Check full log output for errors
2. Verify models loaded correctly
3. Check audio tensor validation passed

### Issue 3: Generation Hangs
**Symptom:** Process stops responding

**Causes:**
- ODE solver slow on system
- Memory pressure
- I/O bottleneck

**Solutions:**
1. Check system resources: `free -h`, `df -h`
2. Close other applications
3. Try with reduced steps (modify code)

### Issue 4: Error During Saving
**Symptom:** Generation completes but audio not saved

**Causes:**
- Codec issues
- File permissions
- Disk space

**Solutions:**
```bash
# Check disk space
df -h /mnt/d

# Check output directory permissions
chmod 777 output

# Check available codecs
python check_codec_pipeline.py
```

### Issue 5: Memory Error
**Symptom:** "CUDA out of memory" or similar

**Causes:**
- Insufficient RAM
- Other processes using memory

**Solutions:**
1. Close other applications
2. Check free memory: `free -h`
3. Try chunked processing

---

## 🔍 Debugging if Generation Fails

### 1. Enable Verbose Output
The script already shows all output. Check for:
- Model loading errors
- CUDA/device issues
- File I/O errors
- Audio validation failures

### 2. Check Logs
```bash
# Capture full output
python generate_verification_95s_song.py 2>&1 | tee verification.log
```

Then examine `verification.log` for errors.

### 3. Test Components Separately
```bash
# Test codec support
python check_codec_pipeline.py

# Test model loading
python -c "from infer.infer_utils import prepare_model; prepare_model(2048, 'cpu')"

# Test audio saving
python -c "
import torch, scipy.io.wavfile, numpy as np
audio = torch.randn(2, 44100).numpy()
scipy.io.wavfile.write('test.wav', 44100, audio)
print('Audio save test OK')
"
```

### 4. Check System Resources
```bash
# Monitor while running
watch -n 1 'free -h; df -h /mnt/d; ps aux | grep python'
```

---

## ✅ Success Criteria

Generation is successful when:

1. **File Created**
   - ✓ `output/output_fixed.wav` exists
   - ✓ Size 8-15 MB
   - ✓ File type is WAV

2. **Audio Valid**
   - ✓ Plays without errors
   - ✓ Duration ~95 seconds
   - ✓ Stereo (2 channels)
   - ✓ 44.1 kHz sample rate

3. **Content Present**
   - ✓ Not completely silent
   - ✓ Vocals present
   - ✓ Music present
   - ✓ Intelligible lyrics

4. **Quality Acceptable**
   - ✓ No major artifacts
   - ✓ No distortion
   - ✓ Reasonable loudness
   - ✓ Good timing/rhythm

---

## 📝 Documentation Reference

| Document | Purpose |
|----------|---------|
| `check_codec_pipeline.py` | Check codec support |
| `CODEC_VALIDATION_SUMMARY.md` | Codec overview |
| `CODEC_AND_FORMAT_TROUBLESHOOTING.md` | Troubleshooting |
| `QUICK_START_FIXED.md` | Quick start guide |
| `README_HANG_UP_FIXES.md` | System overview |

---

## 🎯 Next Steps After Verification

### If Successful
1. ✓ System is fully functional
2. ✓ Ready for production use
3. ✓ Generate songs with your own lyrics
4. ✓ Adjust parameters as needed

### If Issues Found
1. Review troubleshooting section above
2. Check codec support with `python check_codec_pipeline.py`
3. Review logs for specific errors
4. Try alternative solutions listed

---

## 📞 Getting Help

### Quick Diagnostics
```bash
# Check everything
python check_codec_pipeline.py

# Test codec support
python -c "import librosa, torchaudio; print('OK')"

# Check system resources
free -h && df -h /mnt/d && ps aux | grep python
```

### Common Commands

```bash
# Generate with custom settings
python infer/infer.py \
  --lrc-path your_lyrics.lrc \
  --ref-prompt "your style description" \
  --audio-length 95 \
  --output-dir output

# Check output file
ls -lh output/output_fixed.wav

# Play audio
ffplay output/output_fixed.wav

# Get file info
ffprobe output/output_fixed.wav
```

---

## 📊 Expected Output

### Successful Generation Output

```
================================================================================
DIFFRHYTHM 95-SECOND VERIFICATION SONG
================================================================================

This script will:
  1. Check all prerequisites
  2. Create test lyrics
  3. Generate a 95-second song with vocals
  4. Verify the output file
  5. Guide quality assessment

⏱ Estimated time: 30-60 minutes on WSL CPU

────────────────────────────────────────────────────────────────────────────────
► CHECKING PREREQUISITES
────────────────────────────────────────────────────────────────────────────────

Required Libraries:
  ✓ librosa          - Audio loading
  ✓ torchaudio       - Audio I/O
  ✓ torch            - PyTorch
  ✓ scipy            - WAV file writing

Optional Libraries:
  ✓ soundfile        - Audio fallback
  ✓ audioread        - Extended codec support
  ✓ mutagen          - MP3 metadata

✓ All prerequisites met!

[... generation progress ...]

================================================================================
VERIFICATION COMPLETE
================================================================================

✓ SUCCESS - 95-SECOND SONG GENERATED WITH VOCALS

System Status:
  ✓ Codec validation working
  ✓ Audio loading working
  ✓ Model loading working
  ✓ Lyrics processing working
  ✓ Style embedding working
  ✓ ODE sampling working
  ✓ VAE decoding working
  ✓ Audio saving working

Output:
  Location: output/output_fixed.wav
  Format: 16-bit WAV, 44.1 kHz, stereo
  Duration: 95 seconds

🎵 DiffRhythm is fully functional! 🎵
```

---

## Summary

**To verify the complete system:**

1. Run: `python generate_verification_95s_song.py`
2. Answer: `y` when asked to start
3. Wait: 25-50 minutes for generation
4. Listen: `output/output_fixed.wav`
5. Verify: Audio contains vocals and music

**Expected result:** 95-second song with clear vocals, lyrics, and background music!

---

**Status: ✅ READY TO VERIFY**
**Expected Result: WORKING 95-SECOND SONG**
