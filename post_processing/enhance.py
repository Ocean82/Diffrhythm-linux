#!/usr/bin/env python3
"""
Audio Enhancement Pipeline for DiffRhythm

This module provides various audio enhancement techniques to improve
the quality of generated music.

Usage:
    python -m post_processing.enhance input.wav output.wav --mode full

Modes:
    - basic: Normalization and basic EQ
    - denoise: Add noise reduction
    - master: Full mastering chain (EQ, compression, limiting)
    - full: All enhancements
"""

import argparse
import numpy as np
import torch
import torchaudio
from pathlib import Path

try:
    from scipy import signal
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Note: scipy not available. Some features disabled.")


def load_audio(path, target_sr=44100):
    """Load audio file and resample if needed"""
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    return waveform, target_sr


def save_audio(waveform, path, sr=44100):
    """Save audio file"""
    torchaudio.save(path, waveform, sr)


class AudioEnhancer:
    """Audio enhancement pipeline"""

    def __init__(self, sample_rate=44100):
        self.sr = sample_rate

    def normalize(self, audio, target_db=-1.0):
        """Normalize audio to target dB level"""
        peak = audio.abs().max()
        if peak > 0:
            target_amplitude = 10 ** (target_db / 20)
            audio = audio * (target_amplitude / peak)
        return audio

    def highpass_filter(self, audio, cutoff_hz=30):
        """Remove low frequency rumble"""
        return torchaudio.functional.highpass_biquad(
            audio, self.sr, cutoff_freq=cutoff_hz
        )

    def lowpass_filter(self, audio, cutoff_hz=18000):
        """Remove high frequency noise"""
        return torchaudio.functional.lowpass_biquad(
            audio, self.sr, cutoff_freq=cutoff_hz
        )

    def apply_eq(self, audio, bands=None):
        """
        Apply parametric EQ

        Args:
            audio: Audio tensor
            bands: List of EQ band dicts with keys:
                   - freq: Center frequency
                   - gain_db: Gain in dB
                   - q: Q factor (bandwidth)
        """
        if bands is None:
            # Default mastering EQ curve
            bands = [
                {"freq": 80, "gain_db": 1.5, "q": 1.0},    # Bass warmth
                {"freq": 250, "gain_db": -1.0, "q": 1.5},  # Reduce mud
                {"freq": 3000, "gain_db": 1.0, "q": 1.0},  # Presence
                {"freq": 10000, "gain_db": 2.0, "q": 0.7}, # Air/brightness
            ]

        for band in bands:
            audio = torchaudio.functional.equalizer_biquad(
                audio,
                self.sr,
                center_freq=band["freq"],
                gain=band["gain_db"],
                Q=band["q"]
            )
        return audio

    def apply_compression(self, audio, threshold_db=-15, ratio=3.0,
                         attack_ms=10, release_ms=100):
        """
        Apply dynamic range compression

        Note: This is a simplified compressor. For professional results,
        use external tools like FFmpeg or dedicated audio software.
        """
        if not SCIPY_AVAILABLE:
            print("Warning: scipy required for compression")
            return audio

        audio_np = audio.numpy()
        threshold = 10 ** (threshold_db / 20)

        # Calculate envelope
        envelope = np.abs(audio_np)
        attack_coef = np.exp(-1 / (attack_ms * self.sr / 1000))
        release_coef = np.exp(-1 / (release_ms * self.sr / 1000))

        # Smooth envelope
        smoothed = np.zeros_like(envelope)
        for i in range(1, len(smoothed[0])):
            for ch in range(len(smoothed)):
                if envelope[ch, i] > smoothed[ch, i-1]:
                    smoothed[ch, i] = attack_coef * smoothed[ch, i-1] + (1 - attack_coef) * envelope[ch, i]
                else:
                    smoothed[ch, i] = release_coef * smoothed[ch, i-1] + (1 - release_coef) * envelope[ch, i]

        # Calculate gain
        gain = np.ones_like(smoothed)
        above = smoothed > threshold
        gain[above] = threshold + (smoothed[above] - threshold) / ratio
        gain[above] /= smoothed[above]

        compressed = audio_np * gain
        return torch.from_numpy(compressed).float()

    def apply_limiter(self, audio, ceiling_db=-0.3):
        """Apply brick-wall limiter"""
        ceiling = 10 ** (ceiling_db / 20)
        return torch.clamp(audio, -ceiling, ceiling)

    def stereo_enhance(self, audio, width=1.2):
        """
        Enhance stereo width

        Args:
            audio: Stereo audio tensor [2, samples]
            width: Stereo width multiplier (1.0 = no change, >1 = wider)
        """
        if audio.shape[0] != 2:
            return audio  # Not stereo

        mid = (audio[0] + audio[1]) / 2
        side = (audio[0] - audio[1]) / 2

        # Enhance side signal
        side = side * width

        # Recombine
        left = mid + side
        right = mid - side

        return torch.stack([left, right])

    def enhance(self, audio, mode="full"):
        """
        Apply enhancement pipeline

        Args:
            audio: Audio tensor
            mode: Enhancement mode
                - "basic": Normalization, highpass
                - "eq": Add EQ
                - "master": Add compression and limiting
                - "full": All enhancements
        """
        print(f"  Applying {mode} enhancement...")

        # Always apply basic cleanup
        audio = self.highpass_filter(audio, cutoff_hz=30)

        if mode in ["eq", "master", "full"]:
            audio = self.apply_eq(audio)

        if mode in ["master", "full"]:
            audio = self.apply_compression(audio)

        if mode == "full":
            audio = self.stereo_enhance(audio, width=1.1)

        # Always normalize and limit at the end
        audio = self.apply_limiter(audio, ceiling_db=-0.5)
        audio = self.normalize(audio, target_db=-1.0)

        return audio


def enhance_audio_file(input_path, output_path, mode="full"):
    """
    Enhance audio file

    Args:
        input_path: Path to input audio
        output_path: Path to save enhanced audio
        mode: Enhancement mode
    """
    print(f"Loading {input_path}...")
    waveform, sr = load_audio(input_path)
    print(f"  Shape: {waveform.shape}, Sample rate: {sr}Hz")

    enhancer = AudioEnhancer(sample_rate=sr)
    enhanced = enhancer.enhance(waveform, mode=mode)

    print(f"Saving to {output_path}...")
    save_audio(enhanced, output_path, sr)

    # Print stats
    orig_peak = waveform.abs().max().item()
    new_peak = enhanced.abs().max().item()
    print(f"  Original peak: {20*np.log10(orig_peak):.1f} dB")
    print(f"  Enhanced peak: {20*np.log10(new_peak):.1f} dB")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Audio enhancement for DiffRhythm outputs"
    )
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")
    parser.add_argument(
        "--mode",
        default="full",
        choices=["basic", "eq", "master", "full"],
        help="Enhancement mode"
    )
    args = parser.parse_args()

    enhance_audio_file(args.input, args.output, args.mode)


if __name__ == "__main__":
    main()
