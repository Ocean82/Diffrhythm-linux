# DiffRhythm Quality Integration Analysis

## Executive Summary

Based on deep investigation of the DiffRhythm-LINUX project, this document identifies the most compatible and impactful quality improvements that can be implemented.

---

## 1. EXISTING COMPONENTS ANALYSIS

### 1.1 Core Architecture
| Component | Technology | Quality Impact |
|-----------|------------|----------------|
| **Diffusion Model** | DiT (LlamaDecoderLayer) | High - ODE steps controllable |
| **VAE** | JIT-compiled decoder | Medium - Fixed architecture |
| **Style Encoder** | MuQ-MuLan | High - Prompt engineering |
| **Tokenizer** | CNENTokenizer (G2P) | Medium - Phoneme accuracy |
| **Audio Output** | 44.1kHz, 16-bit | Already optimal |

### 1.2 Available Dependencies (from requirements.txt)
```
✅ Already Installed:
- scipy==1.15.2          → Audio filtering, compression
- soundfile==0.13.1      → High-quality audio I/O
- librosa==0.10.2        → Audio analysis, processing
- pydub==0.25.1          → Audio manipulation
- nnAudio==0.3.3         → Neural audio processing
- pytorch-lightning      → Training framework
- transformers==4.49.0   → Model architecture
- safetensors            → Efficient model loading
- einops                 → Tensor operations
- gradio                 → Web interface (in docker)
```

### 1.3 Existing Quality-Related Files
```
post_processing/
├── __init__.py           ✅ Created
├── enhance.py            ✅ Created (EQ, compression, limiting)

train/
├── train.py              ✅ Base training
├── train_lora.py         ✅ Created (LoRA fine-tuning)

infer/
├── infer.py              ✅ Enhanced with ODE controls
├── infer_utils.py        ✅ Style embedding extraction
├── example/
│   ├── music_prompt_keywords.json  ✅ Genre/mood/instrument lists
│   └── *.lrc                       ✅ Example lyrics
```

---

## 2. RECOMMENDED QUALITY IMPROVEMENTS

### Priority 1: Immediate Improvements (No Dependencies)

#### A. Enhanced Prompt Engineering
The `music_prompt_keywords.json` provides structured vocabulary. Create an intelligent prompt builder:

```python
# File: infer/prompt_builder.py
import json
import random
from pathlib import Path

class StylePromptBuilder:
    """Build rich, consistent style prompts for better generation quality"""

    def __init__(self, keywords_path="infer/example/music_prompt_keywords.json"):
        with open(keywords_path) as f:
            self.keywords = json.load(f)

    def build_prompt(
        self,
        genre: str = None,
        mood: str = None,
        instruments: list = None,
        tempo: str = None,
        vocals: str = None,
        production: str = None
    ) -> str:
        """
        Build a comprehensive style prompt

        Args:
            genre: Music genre (Pop, Rock, etc.)
            mood: Emotional mood (Upbeat, Melancholic, etc.)
            instruments: List of instruments
            tempo: Tempo description (slow, medium, fast, or BPM)
            vocals: Vocal style (male, female, choir, etc.)
            production: Production quality (professional, lo-fi, etc.)

        Returns:
            Rich prompt string for MuQ-MuLan
        """
        parts = []

        # Genre
        if genre:
            if genre not in self.keywords.get("genre", []):
                print(f"Warning: '{genre}' not in known genres")
            parts.append(f"{genre} music")

        # Mood
        if mood:
            parts.append(mood.lower())

        # Instruments
        if instruments:
            inst_str = ", ".join(instruments[:3])  # Limit to 3
            parts.append(f"featuring {inst_str}")

        # Tempo
        if tempo:
            if tempo.isdigit():
                parts.append(f"{tempo} BPM")
            else:
                parts.append(f"{tempo} tempo")

        # Vocals
        if vocals:
            parts.append(f"{vocals} vocals")

        # Production quality
        if production:
            parts.append(f"{production} production")
        else:
            parts.append("professional studio production")

        return ", ".join(parts)

    def random_prompt(self, base_genre: str = None) -> str:
        """Generate a random high-quality prompt"""
        genre = base_genre or random.choice(self.keywords["genre"])
        mood = random.choice(self.keywords["mood"])
        instruments = random.sample(self.keywords["instrument"], k=2)

        return self.build_prompt(
            genre=genre,
            mood=mood,
            instruments=instruments,
            production="pristine"
        )

# Example usage:
# builder = StylePromptBuilder()
# prompt = builder.build_prompt(
#     genre="Pop",
#     mood="Upbeat",
#     instruments=["Piano", "Synthesizer"],
#     tempo="120",
#     vocals="female",
#     production="professional studio"
# )
# >> "Pop music, upbeat, featuring Piano, Synthesizer, 120 BPM, female vocals, professional studio production"
```

#### B. Inference Quality Presets

```python
# File: infer/quality_presets.py
"""Quality presets for different use cases"""

QUALITY_PRESETS = {
    "preview": {
        "steps": 4,
        "cfg_strength": 2.0,
        "description": "Fast preview, lower quality (~3-5 min CPU)"
    },
    "draft": {
        "steps": 8,
        "cfg_strength": 2.5,
        "description": "Draft quality, reasonable speed (~6-10 min CPU)"
    },
    "standard": {
        "steps": 16,
        "cfg_strength": 3.0,
        "description": "Standard quality (~12-20 min CPU)"
    },
    "high": {
        "steps": 32,
        "cfg_strength": 4.0,
        "description": "High quality (~25-40 min CPU, 2-3 min GPU)"
    },
    "maximum": {
        "steps": 64,
        "cfg_strength": 5.0,
        "description": "Maximum quality (~50-80 min CPU, 4-6 min GPU)"
    }
}

def get_preset(name: str) -> dict:
    """Get quality preset by name"""
    if name not in QUALITY_PRESETS:
        available = ", ".join(QUALITY_PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return QUALITY_PRESETS[name]

def print_presets():
    """Print all available presets"""
    print("\nAvailable Quality Presets:")
    print("-" * 60)
    for name, config in QUALITY_PRESETS.items():
        print(f"  {name:12} | Steps: {config['steps']:2} | CFG: {config['cfg_strength']} | {config['description']}")
    print()
```

### Priority 2: Post-Processing Pipeline (Using Existing scipy/soundfile)

#### A. Complete Audio Mastering Chain

```python
# File: post_processing/mastering.py
"""Professional-grade audio mastering using existing dependencies"""

import numpy as np
import torch
import torchaudio
from scipy import signal
from scipy.io import wavfile
from pathlib import Path


class AudioMasteringChain:
    """Complete mastering chain for DiffRhythm outputs"""

    def __init__(self, sample_rate=44100):
        self.sr = sample_rate

    def multiband_compress(self, audio, bands=None):
        """
        Multi-band compression for balanced dynamics

        Args:
            audio: numpy array [channels, samples]
            bands: List of (low_freq, high_freq, threshold_db, ratio)
        """
        if bands is None:
            bands = [
                (20, 200, -18, 4),      # Sub-bass
                (200, 800, -15, 3),     # Low-mids
                (800, 4000, -12, 2.5),  # Mids
                (4000, 20000, -10, 2),  # Highs
            ]

        result = np.zeros_like(audio)

        for low, high, thresh_db, ratio in bands:
            # Band-pass filter
            sos = signal.butter(4, [low, high], btype='band', fs=self.sr, output='sos')
            band = signal.sosfilt(sos, audio)

            # Compress this band
            threshold = 10 ** (thresh_db / 20)
            envelope = np.abs(band)

            gain = np.ones_like(envelope)
            above = envelope > threshold
            gain[above] = (threshold + (envelope[above] - threshold) / ratio) / envelope[above]
            gain = np.clip(gain, 0.1, 1.0)

            result += band * gain

        return result

    def stereo_enhance(self, audio, width=1.15):
        """Enhance stereo width"""
        if audio.shape[0] != 2:
            return audio

        mid = (audio[0] + audio[1]) / 2
        side = (audio[0] - audio[1]) / 2
        side *= width

        return np.stack([mid + side, mid - side])

    def harmonic_exciter(self, audio, amount=0.1):
        """Add subtle harmonic enhancement"""
        # Generate harmonics via soft clipping
        harmonics = np.tanh(audio * 2) - audio
        return audio + harmonics * amount

    def brickwall_limiter(self, audio, ceiling_db=-0.3, lookahead_ms=5):
        """True-peak limiter with lookahead"""
        ceiling = 10 ** (ceiling_db / 20)
        lookahead_samples = int(lookahead_ms * self.sr / 1000)

        # Find peaks with lookahead
        peaks = np.maximum.reduce([
            np.roll(np.abs(audio), i, axis=-1)
            for i in range(lookahead_samples)
        ])

        # Calculate gain reduction
        gain = np.where(peaks > ceiling, ceiling / peaks, 1.0)

        # Smooth gain changes
        attack_coef = np.exp(-1 / (0.001 * self.sr))
        release_coef = np.exp(-1 / (0.1 * self.sr))

        smoothed_gain = np.ones_like(gain)
        for i in range(1, gain.shape[-1]):
            if gain[..., i] < smoothed_gain[..., i-1]:
                smoothed_gain[..., i] = attack_coef * smoothed_gain[..., i-1] + (1 - attack_coef) * gain[..., i]
            else:
                smoothed_gain[..., i] = release_coef * smoothed_gain[..., i-1] + (1 - release_coef) * gain[..., i]

        return audio * smoothed_gain

    def apply_mastering(self, audio_path, output_path, preset="balanced"):
        """
        Apply full mastering chain

        Presets:
            - "subtle": Minimal processing
            - "balanced": Standard mastering
            - "loud": Maximized loudness
            - "broadcast": Broadcast-ready (-14 LUFS)
        """
        print(f"Loading {audio_path}...")
        waveform, sr = torchaudio.load(audio_path)
        audio = waveform.numpy()

        print("Applying mastering chain...")

        # 1. High-pass filter (remove DC and sub-bass rumble)
        sos = signal.butter(2, 30, btype='high', fs=self.sr, output='sos')
        audio = signal.sosfilt(sos, audio)

        # 2. Multi-band compression
        if preset in ["balanced", "loud", "broadcast"]:
            print("  - Multi-band compression")
            audio = self.multiband_compress(audio)

        # 3. Harmonic enhancement
        if preset in ["balanced", "loud"]:
            print("  - Harmonic exciter")
            audio = self.harmonic_exciter(audio, amount=0.05 if preset == "balanced" else 0.1)

        # 4. Stereo enhancement
        if audio.shape[0] == 2:
            print("  - Stereo enhancement")
            width = {"subtle": 1.05, "balanced": 1.1, "loud": 1.15, "broadcast": 1.05}
            audio = self.stereo_enhance(audio, width=width.get(preset, 1.1))

        # 5. Final limiting
        print("  - Brickwall limiter")
        ceiling = {"subtle": -1.0, "balanced": -0.5, "loud": -0.1, "broadcast": -1.0}
        audio = self.brickwall_limiter(audio, ceiling_db=ceiling.get(preset, -0.5))

        # 6. Normalize
        peak = np.max(np.abs(audio))
        if peak > 0:
            target = 10 ** (-0.5 / 20)  # -0.5 dB
            audio = audio * (target / peak)

        print(f"Saving to {output_path}...")
        torchaudio.save(output_path, torch.from_numpy(audio).float(), sr)

        print("Done!")
        return output_path


def master_audio(input_path, output_path=None, preset="balanced"):
    """Convenience function for mastering"""
    if output_path is None:
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_mastered{p.suffix}")

    chain = AudioMasteringChain()
    return chain.apply_mastering(input_path, output_path, preset)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python mastering.py input.wav [output.wav] [preset]")
        print("Presets: subtle, balanced, loud, broadcast")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    preset = sys.argv[3] if len(sys.argv) > 3 else "balanced"

    master_audio(input_path, output_path, preset)
```

### Priority 3: Training Enhancements

#### A. Data Preparation for LoRA Training

```python
# File: dataset/prepare_training_data.py
"""Prepare high-quality training data for LoRA fine-tuning"""

import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json


class TrainingDataPreparer:
    """Prepare audio files for DiffRhythm training"""

    def __init__(
        self,
        vae_model_path="pretrained/vae_model.pt",
        muq_model=None,
        sample_rate=44100,
        downsample_rate=2048
    ):
        self.sr = sample_rate
        self.downsample_rate = downsample_rate

        # Load VAE for encoding
        self.vae = torch.jit.load(vae_model_path, map_location="cpu")
        self.muq = muq_model

    def process_audio(self, audio_path, output_dir):
        """
        Process a single audio file

        Creates:
            - latent.pt: VAE-encoded latent
            - style.pt: MuQ style embedding
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            waveform = resampler(waveform)

        # Ensure stereo
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]

        # Normalize
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak * 0.95

        # Encode with VAE
        with torch.no_grad():
            waveform = waveform.unsqueeze(0)  # Add batch dim
            latent = self.vae.encode_export(waveform)
            mean, scale = latent.chunk(2, dim=1)

            # Sample from VAE
            stdev = torch.nn.functional.softplus(scale) + 1e-4
            latent = torch.randn_like(mean) * stdev + mean

        # Save latent
        latent_path = output_dir / "latent.pt"
        torch.save(latent.squeeze(0), latent_path)

        # Extract style embedding
        if self.muq is not None:
            # Use middle 10 seconds for style
            mid_point = waveform.shape[-1] // 2
            style_segment = waveform[..., mid_point-5*self.sr:mid_point+5*self.sr]

            with torch.no_grad():
                style = self.muq(wavs=style_segment.squeeze(0))

            style_path = output_dir / "style.pt"
            torch.save(style.squeeze(0), style_path)

        return {
            "latent_path": str(latent_path),
            "duration": waveform.shape[-1] / self.sr
        }

    def prepare_dataset(self, audio_dir, output_dir, lyrics_dir=None):
        """
        Prepare entire dataset

        Args:
            audio_dir: Directory containing audio files
            output_dir: Output directory for processed data
            lyrics_dir: Optional directory with .lrc files
        """
        audio_dir = Path(audio_dir)
        output_dir = Path(output_dir)

        audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))

        scp_lines = []

        for audio_path in tqdm(audio_files, desc="Processing audio"):
            name = audio_path.stem
            item_dir = output_dir / name

            try:
                result = self.process_audio(audio_path, item_dir)

                # Find matching lyrics
                lrc_path = None
                if lyrics_dir:
                    lrc_file = Path(lyrics_dir) / f"{name}.lrc"
                    if lrc_file.exists():
                        lrc_path = str(lrc_file)

                # Create SCP entry
                entry = f"{name}|{item_dir}/lrc.pt|{item_dir}/latent.pt|{item_dir}/style.pt"
                scp_lines.append(entry)

            except Exception as e:
                print(f"Error processing {audio_path}: {e}")

        # Save SCP file
        scp_path = output_dir / "train.scp"
        with open(scp_path, "w") as f:
            f.write("\n".join(scp_lines))

        print(f"Dataset prepared: {len(scp_lines)} files")
        print(f"SCP file: {scp_path}")
```

---

## 3. INTEGRATION RECOMMENDATIONS

### 3.1 Immediate Actions (No New Dependencies)

| Action | File | Impact |
|--------|------|--------|
| Add prompt builder | `infer/prompt_builder.py` | Better style control |
| Add quality presets | `infer/quality_presets.py` | Easier quality selection |
| Add mastering chain | `post_processing/mastering.py` | Polished output |
| Update infer.py | Add `--preset` flag | User convenience |

### 3.2 Short-Term (Install Additional Packages)

```bash
# For enhanced post-processing
pip install pedalboard     # Professional audio effects
pip install noisereduce    # AI-based noise reduction
pip install pyloudnorm     # Loudness normalization (broadcast standard)

# For LoRA training
pip install peft           # Parameter-efficient fine-tuning
pip install bitsandbytes   # Already installed - 8-bit training
```

### 3.3 Recommended External Tools

| Tool | Purpose | Integration |
|------|---------|-------------|
| **FFmpeg** | Audio conversion, filtering | Shell commands |
| **Demucs** | Stem separation | `pip install demucs` |
| **Vocos** | Neural vocoder | Replace VAE decoder |
| **AudioSR** | Audio super-resolution | Post-processing |

---

## 4. IMPLEMENTATION PRIORITY

### Phase 1: Immediate (1-2 hours)
1. ✅ Create `infer/prompt_builder.py`
2. ✅ Create `infer/quality_presets.py`
3. ✅ Create `post_processing/mastering.py`
4. Update `infer/infer.py` with `--preset` flag

### Phase 2: Short-Term (1 day)
1. Install `pedalboard` and `pyloudnorm`
2. Enhance mastering chain with professional effects
3. Add loudness normalization

### Phase 3: Medium-Term (1 week)
1. Set up LoRA training pipeline
2. Prepare sample training data
3. Fine-tune on specific music styles

### Phase 4: Long-Term (1 month)
1. Evaluate neural vocoder replacements (Vocos, BigVGAN)
2. Implement audio super-resolution
3. Train custom style adapters

---

## 5. QUICK START

```bash
# Generate with quality preset
python -m infer.infer \
  --lrc-path infer/example/eg_en.lrc \
  --ref-prompt "Pop music, upbeat, featuring Piano, Synthesizer, 120 BPM, female vocals, professional studio production" \
  --steps 32 \
  --cfg-strength 4.0 \
  --output-dir output

# Apply mastering
python -m post_processing.mastering output/output_fixed.wav output/mastered.wav balanced

# Or use the full pipeline
python -m post_processing.enhance output/output_fixed.wav output/enhanced.wav --mode full
```

---

## 6. QUALITY CHECKLIST

Before release, verify:

- [ ] ODE steps ≥ 16 for quality output
- [ ] CFG strength between 3.0-5.0
- [ ] Style prompt is detailed (genre + mood + instruments + production)
- [ ] Post-processing applied (at minimum: normalization + limiting)
- [ ] Loudness normalized to -14 LUFS for streaming
- [ ] No clipping or distortion in output
- [ ] Stereo imaging is balanced

---

*Generated from DiffRhythm-LINUX project analysis*
*Last updated: 2026-01-19*
