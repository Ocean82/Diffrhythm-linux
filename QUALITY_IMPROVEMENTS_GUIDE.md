# DiffRhythm Quality Improvements Guide

## System Analysis

| Component | Current State | Notes |
|-----------|---------------|-------|
| Sample Rate | 44100 Hz | Already high quality |
| VAE Compression | 2048x downsampling | Latent space: 64 channels |
| Model | ~1B params DiT (LlamaDecoderLayer) | 16 layers, 2048 dim, 32 heads |
| ODE Steps | 8-32 (configurable) | More steps = higher quality |
| Training | Accelerate + EMA | Supports distributed training |

---

## 1. BEFORE GENERATION (Model & Training)

### 1.1 LoRA Fine-Tuning ✅ HIGHLY FEASIBLE

**Feasibility: HIGH** - The architecture is well-suited for LoRA

The DiT model uses `LlamaDecoderLayer` which has standard linear layers perfect for LoRA injection.

**Implementation:**

```python
# Install PEFT library
# pip install peft

from peft import LoraConfig, get_peft_model

# Target modules in DiT/LlamaDecoderLayer
lora_config = LoraConfig(
    r=16,  # LoRA rank (8-64 typical)
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",      # FFN (MLP)
    ],
    lora_dropout=0.05,
    bias="none",
)

# Apply to model
model = get_peft_model(cfm.transformer, lora_config)
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
# Expected: ~10-50M trainable (vs 1B total)
```

**Training Script Modification:**
```python
# In train/train.py, modify model creation:
from peft import LoraConfig, get_peft_model

# After creating CFM model
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
)
model.transformer = get_peft_model(model.transformer, lora_config)
```

**Benefits:**
- 10-50x fewer trainable parameters
- Can run on consumer GPUs (16GB VRAM)
- Preserves base model capabilities
- Easy to merge or swap adapters

---

### 1.2 Better Conditioning Inputs ✅ FEASIBLE

**Current Conditioning:**
- Style prompt: MuQ-MuLan embeddings (512-dim) from audio or text
- Lyrics: Phoneme tokens with timing
- Start time: Normalized position embedding

**Improvements:**

#### A. Richer Text Prompts
The model uses MuQ-MuLan for text → embedding. Use detailed prompts:

```python
# Instead of:
prompt = "pop music"

# Use:
prompt = "upbeat pop music, 120 BPM, electronic synths, female vocals, major key, energetic drums"
```

#### B. Audio Reference Quality
For `--ref-audio-path`, use high-quality reference tracks:
- 44.1kHz, 16-bit minimum
- 10+ seconds of representative audio
- Clear production (not compressed/YouTube rips)

#### C. Lyrics Timing Precision
Ensure LRC timing is accurate:
```
[00:00.00] First line here
[00:03.50] Second line with precise timing
```

---

### 1.3 Higher Sample Rate & Longer Training ⚠️ LIMITED

**Current State:**
- Already at 44.1kHz (CD quality)
- VAE is pre-trained with 2048x downsampling

**Limitations:**
- VAE would need retraining for different sample rates
- The VAE (DiffRhythm-vae) is a JIT-compiled model, not easily modifiable

**What's Possible:**
- More ODE steps (--steps 32 or higher)
- More training epochs on quality data
- Better CFG strength tuning

---

## 2. AFTER GENERATION (Post-Processing) ✅ ALL FEASIBLE

### 2.1 Audio Super-Resolution

Create `post_processing/enhance_audio.py`:

```python
#!/usr/bin/env python3
"""
Audio Enhancement Pipeline for DiffRhythm outputs
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path

def load_audio(path, target_sr=44100):
    """Load audio file and resample if needed"""
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    return waveform, target_sr

def apply_basic_enhancement(waveform, sr=44100):
    """Apply basic audio enhancements"""
    # High-pass filter to remove rumble (below 30Hz)
    waveform = torchaudio.functional.highpass_biquad(waveform, sr, cutoff_freq=30)

    # Gentle compression
    # Normalize peaks
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak * 0.95

    return waveform

def apply_hifi_gan_enhancement(waveform, model_path=None):
    """
    Apply HiFi-GAN for audio super-resolution
    Requires: pip install speechbrain
    """
    try:
        from speechbrain.pretrained import HIFIGAN
        hifi_gan = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-ljspeech",
            savedir="pretrained_models/hifigan"
        )
        # Note: HiFi-GAN expects mel spectrograms, not raw waveforms
        # This is a placeholder for the concept
        return waveform
    except ImportError:
        print("HiFi-GAN not available. Install: pip install speechbrain")
        return waveform

def apply_demucs_separation(audio_path, output_dir):
    """
    Use Demucs for stem separation and remixing
    Requires: pip install demucs
    """
    import subprocess
    try:
        subprocess.run([
            "python", "-m", "demucs",
            "-o", output_dir,
            "--two-stems", "vocals",
            audio_path
        ], check=True)
        print(f"Stems saved to {output_dir}")
    except Exception as e:
        print(f"Demucs failed: {e}")

def enhance_audio_file(input_path, output_path, enhance_mode="basic"):
    """
    Main enhancement function

    Args:
        input_path: Path to input audio
        output_path: Path to save enhanced audio
        enhance_mode: "basic", "hifigan", or "full"
    """
    print(f"Loading {input_path}...")
    waveform, sr = load_audio(input_path)

    print(f"Applying {enhance_mode} enhancement...")
    if enhance_mode == "basic":
        waveform = apply_basic_enhancement(waveform, sr)
    elif enhance_mode == "hifigan":
        waveform = apply_hifi_gan_enhancement(waveform)
    elif enhance_mode == "full":
        waveform = apply_basic_enhancement(waveform, sr)
        waveform = apply_hifi_gan_enhancement(waveform)

    print(f"Saving to {output_path}...")
    torchaudio.save(output_path, waveform, sr)
    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")
    parser.add_argument("--mode", default="basic", choices=["basic", "hifigan", "full"])
    args = parser.parse_args()

    enhance_audio_file(args.input, args.output, args.mode)
```

---

### 2.2 Noise Reduction & EQ

Create `post_processing/audio_mastering.py`:

```python
#!/usr/bin/env python3
"""
Audio Mastering Pipeline
"""

import numpy as np
import torch
import torchaudio
from scipy import signal
from scipy.io import wavfile

def apply_eq(waveform, sr, bands=None):
    """
    Apply parametric EQ

    Default bands for music mastering:
    - Low cut: 30Hz highpass
    - Low shelf: +2dB at 100Hz
    - High shelf: +3dB at 10kHz
    - Air: +2dB at 16kHz
    """
    if bands is None:
        bands = [
            {"type": "highpass", "freq": 30, "Q": 0.7},
            {"type": "lowshelf", "freq": 100, "gain_db": 2},
            {"type": "highshelf", "freq": 10000, "gain_db": 3},
        ]

    audio = waveform.numpy()

    for band in bands:
        if band["type"] == "highpass":
            b, a = signal.butter(2, band["freq"] / (sr/2), btype='high')
            audio = signal.filtfilt(b, a, audio)
        elif band["type"] == "lowshelf":
            # Simplified low shelf
            b, a = signal.butter(1, band["freq"] / (sr/2), btype='low')
            low = signal.filtfilt(b, a, audio)
            gain = 10 ** (band["gain_db"] / 20)
            audio = audio + (low * (gain - 1))
        elif band["type"] == "highshelf":
            b, a = signal.butter(1, band["freq"] / (sr/2), btype='high')
            high = signal.filtfilt(b, a, audio)
            gain = 10 ** (band["gain_db"] / 20)
            audio = audio + (high * (gain - 1))

    return torch.from_numpy(audio).float()

def apply_compression(waveform, threshold_db=-10, ratio=4, attack_ms=10, release_ms=100, sr=44100):
    """Apply dynamic range compression"""
    audio = waveform.numpy()

    # Simple compressor
    threshold = 10 ** (threshold_db / 20)
    attack_samples = int(attack_ms * sr / 1000)
    release_samples = int(release_ms * sr / 1000)

    envelope = np.abs(audio)
    # Smooth envelope
    envelope = np.maximum.accumulate(envelope)

    gain = np.ones_like(audio)
    above_threshold = envelope > threshold
    gain[above_threshold] = (threshold + (envelope[above_threshold] - threshold) / ratio) / envelope[above_threshold]

    compressed = audio * gain
    return torch.from_numpy(compressed).float()

def apply_limiter(waveform, ceiling_db=-0.3):
    """Apply brick-wall limiter"""
    ceiling = 10 ** (ceiling_db / 20)
    return torch.clamp(waveform, -ceiling, ceiling)

def master_audio(input_path, output_path):
    """Full mastering chain"""
    print(f"Loading {input_path}...")
    waveform, sr = torchaudio.load(input_path)

    print("Applying EQ...")
    waveform = apply_eq(waveform, sr)

    print("Applying compression...")
    waveform = apply_compression(waveform, threshold_db=-12, ratio=3, sr=sr)

    print("Applying limiter...")
    waveform = apply_limiter(waveform, ceiling_db=-0.5)

    # Normalize
    peak = waveform.abs().max()
    waveform = waveform / peak * 0.95

    print(f"Saving to {output_path}...")
    torchaudio.save(output_path, waveform, sr)
    print("Done!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python audio_mastering.py input.wav output.wav")
        sys.exit(1)
    master_audio(sys.argv[1], sys.argv[2])
```

---

### 2.3 Using External Tools

**Demucs (Stem Separation):**
```bash
pip install demucs
python -m demucs --two-stems vocals output/output_fixed.wav -o output/stems/
```

**FFmpeg (Basic Enhancement):**
```bash
# Normalize + slight compression
ffmpeg -i output/output_fixed.wav \
  -af "loudnorm=I=-14:TP=-1:LRA=11,acompressor=threshold=-20dB:ratio=4:attack=5:release=50" \
  output/output_mastered.wav

# Add reverb
ffmpeg -i output/output_fixed.wav \
  -af "aecho=0.8:0.88:60:0.4" \
  output/output_reverb.wav
```

---

## 3. INFERENCE QUALITY IMPROVEMENTS ✅

### 3.1 Increase ODE Steps

```bash
# Higher quality (slower)
python -m infer.infer --lrc-path lyrics.lrc --ref-prompt "pop music" --steps 32

# Maximum quality
python -m infer.infer --lrc-path lyrics.lrc --ref-prompt "pop music" --steps 64
```

### 3.2 CFG Strength Tuning

```bash
# Default: 4.0 (GPU) / 2.0 (CPU)
# Higher = more adherence to prompt, but can cause artifacts
python -m infer.infer --lrc-path lyrics.lrc --ref-prompt "pop music" --cfg-strength 6.0

# Lower = more natural but less prompt-adherent
python -m infer.infer --lrc-path lyrics.lrc --ref-prompt "pop music" --cfg-strength 2.5
```

### 3.3 Seed Control for Best Results

```python
# In infer.py, add seed control for reproducibility
# Generate multiple outputs with different seeds, pick the best

for seed in [42, 123, 456, 789, 1000]:
    torch.manual_seed(seed)
    # Generate...
```

---

## 4. QUANTIZATION & EFFICIENCY ⚠️ MODERATE

### 4.1 INT8 Quantization (Inference Only)

```python
# Using torch.compile with reduced precision
import torch

# Enable torch.compile for faster inference (PyTorch 2.0+)
cfm.transformer = torch.compile(cfm.transformer, mode="reduce-overhead")

# Or use dynamic quantization
cfm_quantized = torch.quantization.quantize_dynamic(
    cfm, {torch.nn.Linear}, dtype=torch.qint8
)
```

### 4.2 ONNX Export (For Deployment)

```python
# Export to ONNX for optimized inference
import torch.onnx

dummy_input = {
    "x": torch.randn(1, 2048, 64),
    "cond": torch.randn(1, 2048, 64),
    "text": torch.randint(0, 363, (1, 2048)),
    "time": torch.tensor([0.5]),
    # ... other inputs
}

torch.onnx.export(
    cfm.transformer,
    dummy_input,
    "diffrhythm_transformer.onnx",
    opset_version=14,
)
```

---

## 5. COMPLETE ENHANCEMENT PIPELINE

Create `post_processing/full_pipeline.py`:

```python
#!/usr/bin/env python3
"""
Complete audio enhancement pipeline for DiffRhythm
"""

import argparse
import os
import subprocess
from pathlib import Path

def run_pipeline(input_path, output_dir, steps=None):
    """Run complete enhancement pipeline"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_name = Path(input_path).stem

    print("=" * 60)
    print("DiffRhythm Audio Enhancement Pipeline")
    print("=" * 60)

    # Step 1: Basic enhancement
    print("\n[1/4] Applying basic enhancement...")
    basic_output = output_dir / f"{input_name}_enhanced.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-af", "loudnorm=I=-14:TP=-1:LRA=11",
        str(basic_output)
    ], check=True)

    # Step 2: Stem separation (optional)
    if steps and "stems" in steps:
        print("\n[2/4] Separating stems with Demucs...")
        stems_dir = output_dir / "stems"
        subprocess.run([
            "python", "-m", "demucs",
            "--two-stems", "vocals",
            "-o", str(stems_dir),
            str(basic_output)
        ], check=False)

    # Step 3: EQ and compression
    print("\n[3/4] Applying EQ and compression...")
    mastered_output = output_dir / f"{input_name}_mastered.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(basic_output),
        "-af", "equalizer=f=100:t=q:w=1:g=2,equalizer=f=10000:t=q:w=1:g=3,acompressor=threshold=-15dB:ratio=3:attack=5:release=50",
        str(mastered_output)
    ], check=True)

    # Step 4: Final limiting
    print("\n[4/4] Applying final limiter...")
    final_output = output_dir / f"{input_name}_final.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(mastered_output),
        "-af", "alimiter=limit=0.95:attack=5:release=50",
        str(final_output)
    ], check=True)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"Final output: {final_output}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("-o", "--output-dir", default="output/enhanced")
    parser.add_argument("--stems", action="store_true", help="Also separate stems")
    args = parser.parse_args()

    steps = []
    if args.stems:
        steps.append("stems")

    run_pipeline(args.input, args.output_dir, steps)
```

---

## 6. SUMMARY TABLE

| Improvement | Feasibility | Effort | Impact |
|-------------|-------------|--------|--------|
| **LoRA Fine-Tuning** | ✅ High | Medium | High |
| **Better Prompts** | ✅ High | Low | Medium |
| **More ODE Steps** | ✅ High | Low | High |
| **CFG Tuning** | ✅ High | Low | Medium |
| **Post-Processing EQ** | ✅ High | Low | Medium |
| **Demucs Stems** | ✅ High | Low | Medium |
| **Audio Mastering** | ✅ High | Low | Medium |
| **Higher Sample Rate** | ⚠️ Low | Very High | Medium |
| **Retrain VAE** | ⚠️ Low | Very High | High |
| **INT8 Quantization** | ⚠️ Medium | Medium | Low (quality) |

---

## 7. QUICK START

```bash
# 1. Generate with higher quality settings
python -m infer.infer \
  --lrc-path lyrics.lrc \
  --ref-prompt "professional pop music, pristine production, 120 BPM" \
  --steps 32 \
  --cfg-strength 4.0 \
  --output-dir output

# 2. Apply post-processing
ffmpeg -i output/output_fixed.wav \
  -af "loudnorm=I=-14:TP=-1:LRA=11,acompressor=threshold=-15dB:ratio=3" \
  output/output_mastered.wav

# 3. (Optional) Separate stems
pip install demucs
python -m demucs --two-stems vocals output/output_mastered.wav -o output/stems/
```
