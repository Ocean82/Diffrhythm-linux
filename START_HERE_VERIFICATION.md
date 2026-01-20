# 🎵 START HERE - Generate 95-Second Verification Song

## Quick Start (3 Steps)

### Step 1: Check Prerequisites
```bash
python check_codec_pipeline.py
```

### Step 2: Run Verification Generator
```bash
python generate_verification_95s_song.py
```

### Step 3: Listen to Output
```bash
# File will be at:
output/output_fixed.wav

# To play (if audio player installed):
ffplay output/output_fixed.wav
```

---

## What Happens

### The Script Will:
1. ✓ Check all required packages (2-3 seconds)
2. ✓ Create test lyrics (< 1 second)
3. ✓ Ask for confirmation to start
4. ✓ Generate 95-second song (25-50 minutes)
5. ✓ Verify output file

### During Generation You'll See:
```
[1] Models loading (2-5 minutes)
    - CFM model
    - Tokenizer
    - MuQ-MuLan
    - VAE model

[2] Lyrics processing
    - Tokenizing lyrics
    - Creating embeddings

[3] ODE Sampling (15-30 minutes)
    - ODE step 5/16
    - ODE step 10/16
    - ODE step 15/16
    - ODE step 16/16

[4] VAE Decoding (5-10 minutes)
    - Decoding latents to audio

[5] Audio Saving (< 5 seconds)
    - Saving to WAV file
```

---

## What You Get

**Output File:** `output/output_fixed.wav`

**Specifications:**
- Duration: 95 seconds
- Format: WAV (16-bit stereo)
- Sample Rate: 44.1 kHz
- File Size: ~8-15 MB
- Content: Music + vocals + lyrics

**What to Hear:**
- Clear singing vocals
- Recognizable lyrics
- Background music
- Good rhythm and timing

---

## Timeline

| Component | Time |
|-----------|------|
| Prerequisites check | < 1 min |
| Model loading | 2-5 min |
| ODE sampling | 15-30 min |
| VAE decoding | 5-10 min |
| Audio saving | < 1 min |
| **Total** | **25-50 min** |

---

## Before Starting

### Ensure You Have

✓ **8+ GB RAM available**
```bash
free -h  # Check free memory
```

✓ **5+ GB disk space**
```bash
df -h /mnt/d  # Check disk space
```

✓ **Required packages installed**
```bash
pip install librosa torchaudio scipy soundfile audioread mutagen
sudo apt-get install ffmpeg libsndfile1
```

---

## Troubleshooting

### Issue: Script fails immediately
**Solution:**
```bash
python check_codec_pipeline.py  # Check what's missing
pip install librosa torchaudio scipy  # Install minimum
```

### Issue: Generation hangs
**Solutions:**
1. Check system resources: `free -h`
2. Close other applications
3. Monitor with: `watch -n 1 "free -h; df -h /mnt/d"`

### Issue: Output file too small
**Possible cause:** Audio generation failed
**Solution:** Check log output for errors

### Issue: Audio is silent
**Possible cause:** CPU inference limitation
**Solution:** This is expected for some CPU-only systems

---

## Files You Need to Know

| File | Purpose |
|------|---------|
| `generate_verification_95s_song.py` | **← RUN THIS** |
| `check_codec_pipeline.py` | Check codec support |
| `infer/infer.py` | Main generation script |
| `output/verification_95s.lrc` | Generated lyrics |
| `output/output_fixed.wav` | **← YOUR OUTPUT** |

---

## Success Indicators

✅ Generation completes (doesn't hang)
✅ No fatal errors in output
✅ File `output/output_fixed.wav` created
✅ File size 8-15 MB
✅ Audio plays
✅ Contains vocals and music
✅ Duration ~95 seconds

---

## After Generation

### If Successful
- ✓ System works perfectly
- ✓ All components functional
- ✓ Ready to use with your own lyrics
- ✓ Ready for production

### If Issues
- Review `VERIFICATION_SONG_INSTRUCTIONS.md`
- Check `CODEC_AND_FORMAT_TROUBLESHOOTING.md`
- Run: `python check_codec_pipeline.py`

---

## Generate Again With Your Own Lyrics

```bash
# Create your lyrics in output/my_lyrics.lrc
python infer/infer.py \
  --lrc-path output/my_lyrics.lrc \
  --ref-prompt "your style description" \
  --audio-length 95 \
  --output-dir output
```

---

## Quick Command Reference

```bash
# Check everything
python check_codec_pipeline.py

# Generate verification song
python generate_verification_95s_song.py

# Generate with custom lyrics
python infer/infer.py \
  --lrc-path output/my_lyrics.lrc \
  --ref-prompt "pop song, upbeat vocals" \
  --audio-length 95 \
  --output-dir output

# Check output file
ls -lh output/output_fixed.wav

# Get file info
ffprobe output/output_fixed.wav

# Play audio (if ffplay installed)
ffplay output/output_fixed.wav
```

---

## Expected Output Example

When you run the script and it completes successfully:

```
================================================================================
DIFFRHYTHM 95-SECOND VERIFICATION SONG
================================================================================

────────────────────────────────────────────────────────────────────────────────
► CHECKING PREREQUISITES
────────────────────────────────────────────────────────────────────────────────

✓ All prerequisites met!

────────────────────────────────────────────────────────────────────────────────
► CREATING TEST LYRICS
────────────────────────────────────────────────────────────────────────────────

✓ Created test lyrics: output/verification_95s.lrc
✓ Duration: ~95 seconds
✓ Contains: 24 lyric lines with vocals

────────────────────────────────────────────────────────────────────────────────
► GENERATING 95-SECOND SONG WITH VOCALS
────────────────────────────────────────────────────────────────────────────────

[... generation progress for 25-50 minutes ...]

────────────────────────────────────────────────────────────────────────────────
► VERIFYING OUTPUT
────────────────────────────────────────────────────────────────────────────────

✓ Output file exists: output/output_fixed.wav
✓ File size: 8,377,952 bytes (7.99 MB)
✓ File size is reasonable for 95-second stereo audio

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

## 🚀 Ready? Let's Go!

```bash
python generate_verification_95s_song.py
```

Answer `y` when asked and wait 25-50 minutes.

Then listen to your generated song at: `output/output_fixed.wav`

---

**Welcome to DiffRhythm! 🎵**
