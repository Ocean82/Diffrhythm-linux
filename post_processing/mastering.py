#!/usr/bin/env python3
"""
Professional Audio Mastering Chain for DiffRhythm

This module provides a complete mastering chain using scipy and torchaudio,
without requiring additional dependencies.

Usage:
    python -m post_processing.mastering input.wav output.wav --preset balanced

Presets:
    - subtle: Minimal processing, preserves dynamics
    - balanced: Standard mastering, good for most music
    - loud: Maximized loudness, reduced dynamics
    - broadcast: Broadcast-ready, normalized to -14 LUFS
"""

import argparse
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Tuple, Optional

try:
    from scipy import signal
    from scipy.ndimage import uniform_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Some features disabled.")


class AudioMasteringChain:
    """Complete professional mastering chain for DiffRhythm outputs"""

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize mastering chain

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sr = sample_rate

    def analyze_audio(self, audio: np.ndarray) -> dict:
        """
        Analyze audio characteristics

        Args:
            audio: Audio array [channels, samples]

        Returns:
            Dictionary with analysis results
        """
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio ** 2))
        crest_factor = peak / (rms + 1e-10)
        dc_offset = np.mean(audio)

        # Estimate loudness (simplified LUFS)
        # True LUFS requires K-weighting, this is an approximation
        loudness_lufs = -0.691 + 10 * np.log10(rms ** 2 + 1e-10)

        return {
            "peak_db": 20 * np.log10(peak + 1e-10),
            "rms_db": 20 * np.log10(rms + 1e-10),
            "crest_factor_db": 20 * np.log10(crest_factor),
            "dc_offset": dc_offset,
            "loudness_lufs_approx": loudness_lufs,
            "is_stereo": audio.shape[0] == 2,
        }

    def remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """Remove DC offset from audio"""
        return audio - np.mean(audio, axis=-1, keepdims=True)

    def high_pass_filter(
        self,
        audio: np.ndarray,
        cutoff_hz: float = 30,
        order: int = 2
    ) -> np.ndarray:
        """
        Apply high-pass filter to remove subsonic content

        Args:
            audio: Audio array
            cutoff_hz: Filter cutoff frequency
            order: Filter order
        """
        if not SCIPY_AVAILABLE:
            return audio

        sos = signal.butter(order, cutoff_hz, btype='high', fs=self.sr, output='sos')
        return signal.sosfilt(sos, audio)

    def low_pass_filter(
        self,
        audio: np.ndarray,
        cutoff_hz: float = 18000,
        order: int = 2
    ) -> np.ndarray:
        """Apply low-pass filter"""
        if not SCIPY_AVAILABLE:
            return audio

        sos = signal.butter(order, cutoff_hz, btype='low', fs=self.sr, output='sos')
        return signal.sosfilt(sos, audio)

    def parametric_eq(
        self,
        audio: np.ndarray,
        bands: list = None
    ) -> np.ndarray:
        """
        Apply parametric EQ

        Args:
            audio: Audio array
            bands: List of (freq, gain_db, Q) tuples

        Default bands optimized for music mastering:
            - 80 Hz: +1.5 dB (bass warmth)
            - 250 Hz: -1 dB (reduce mud)
            - 3000 Hz: +1 dB (presence)
            - 10000 Hz: +2 dB (air/brightness)
        """
        if not SCIPY_AVAILABLE:
            return audio

        if bands is None:
            bands = [
                (80, 1.5, 1.0),    # Bass warmth
                (250, -1.0, 1.5),  # Reduce mud
                (3000, 1.0, 1.0),  # Presence
                (10000, 2.0, 0.7), # Air
            ]

        result = audio.copy()

        for freq, gain_db, Q in bands:
            # Peak/notch filter
            w0 = 2 * np.pi * freq / self.sr
            A = 10 ** (gain_db / 40)
            alpha = np.sin(w0) / (2 * Q)

            b0 = 1 + alpha * A
            b1 = -2 * np.cos(w0)
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha / A

            b = np.array([b0/a0, b1/a0, b2/a0])
            a = np.array([1, a1/a0, a2/a0])

            result = signal.lfilter(b, a, result)

        return result

    def multiband_compress(
        self,
        audio: np.ndarray,
        bands: list = None
    ) -> np.ndarray:
        """
        Apply multi-band compression

        Args:
            audio: Audio array
            bands: List of (low_freq, high_freq, threshold_db, ratio, attack_ms, release_ms)
        """
        if not SCIPY_AVAILABLE:
            return audio

        if bands is None:
            bands = [
                (20, 150, -18, 4, 10, 100),     # Sub-bass
                (150, 500, -15, 3, 5, 80),      # Bass
                (500, 2000, -12, 2.5, 3, 60),   # Low-mids
                (2000, 6000, -10, 2, 2, 50),    # High-mids
                (6000, 20000, -8, 2, 1, 40),    # Highs
            ]

        result = np.zeros_like(audio)

        for low, high, thresh_db, ratio, attack_ms, release_ms in bands:
            # Band-pass filter
            sos = signal.butter(4, [low, high], btype='band', fs=self.sr, output='sos')
            band = signal.sosfilt(sos, audio)

            # Compress this band
            band = self._compress_signal(
                band, thresh_db, ratio, attack_ms, release_ms
            )

            result += band

        return result

    def _compress_signal(
        self,
        audio: np.ndarray,
        threshold_db: float,
        ratio: float,
        attack_ms: float,
        release_ms: float
    ) -> np.ndarray:
        """Apply compression to a signal"""
        threshold = 10 ** (threshold_db / 20)
        attack_coef = np.exp(-1 / (attack_ms * self.sr / 1000))
        release_coef = np.exp(-1 / (release_ms * self.sr / 1000))

        envelope = np.abs(audio)

        # Smooth envelope
        smoothed = np.zeros_like(envelope)
        for i in range(1, envelope.shape[-1]):
            for ch in range(envelope.shape[0]):
                if envelope[ch, i] > smoothed[ch, i-1]:
                    smoothed[ch, i] = attack_coef * smoothed[ch, i-1] + (1 - attack_coef) * envelope[ch, i]
                else:
                    smoothed[ch, i] = release_coef * smoothed[ch, i-1] + (1 - release_coef) * envelope[ch, i]

        # Calculate gain reduction
        gain = np.ones_like(smoothed)
        above_thresh = smoothed > threshold
        gain[above_thresh] = (threshold + (smoothed[above_thresh] - threshold) / ratio) / smoothed[above_thresh]

        return audio * gain

    def stereo_enhance(
        self,
        audio: np.ndarray,
        width: float = 1.15,
        bass_mono_freq: float = 120
    ) -> np.ndarray:
        """
        Enhance stereo width

        Args:
            audio: Stereo audio [2, samples]
            width: Stereo width multiplier (1.0 = no change, >1 = wider)
            bass_mono_freq: Frequency below which to mono the bass

        Returns:
            Enhanced stereo audio
        """
        if audio.shape[0] != 2:
            return audio  # Not stereo

        # Mid-side processing
        mid = (audio[0] + audio[1]) / 2
        side = (audio[0] - audio[1]) / 2

        # Enhance side (stereo width)
        side = side * width

        # Mono the bass for tighter low end
        if SCIPY_AVAILABLE and bass_mono_freq > 0:
            sos_low = signal.butter(2, bass_mono_freq, btype='low', fs=self.sr, output='sos')
            sos_high = signal.butter(2, bass_mono_freq, btype='high', fs=self.sr, output='sos')

            side_low = signal.sosfilt(sos_low, side)
            side_high = signal.sosfilt(sos_high, side)

            # Reduce low frequency stereo content
            side = side_high + side_low * 0.3

        # Recombine
        left = mid + side
        right = mid - side

        return np.stack([left, right])

    def harmonic_exciter(
        self,
        audio: np.ndarray,
        amount: float = 0.05,
        frequency: float = 3000
    ) -> np.ndarray:
        """
        Add harmonic enhancement (subtle saturation)

        Args:
            audio: Audio array
            amount: Amount of harmonics to add (0.0 - 0.2)
            frequency: Frequency above which to add harmonics
        """
        if not SCIPY_AVAILABLE:
            return audio

        # High-pass to get the content to enhance
        sos = signal.butter(2, frequency, btype='high', fs=self.sr, output='sos')
        high = signal.sosfilt(sos, audio)

        # Generate harmonics via soft saturation
        harmonics = np.tanh(high * 3) - high

        return audio + harmonics * amount

    def brickwall_limiter(
        self,
        audio: np.ndarray,
        ceiling_db: float = -0.3,
        lookahead_ms: float = 5,
        release_ms: float = 100
    ) -> np.ndarray:
        """
        Apply true-peak limiter

        Args:
            audio: Audio array
            ceiling_db: Maximum output level
            lookahead_ms: Lookahead time in ms
            release_ms: Release time in ms
        """
        ceiling = 10 ** (ceiling_db / 20)
        lookahead_samples = int(lookahead_ms * self.sr / 1000)
        release_coef = np.exp(-1 / (release_ms * self.sr / 1000))

        # Find peaks with lookahead
        peaks = np.abs(audio)
        for i in range(1, lookahead_samples):
            peaks = np.maximum(peaks, np.roll(np.abs(audio), -i, axis=-1))

        # Calculate gain reduction needed
        gain = np.where(peaks > ceiling, ceiling / peaks, 1.0)

        # Smooth gain changes (attack = instant, release = gradual)
        smoothed_gain = gain.copy()
        for i in range(1, gain.shape[-1]):
            for ch in range(gain.shape[0]):
                if gain[ch, i] >= smoothed_gain[ch, i-1]:
                    # Release
                    smoothed_gain[ch, i] = release_coef * smoothed_gain[ch, i-1] + (1 - release_coef) * gain[ch, i]
                else:
                    # Attack (instant)
                    smoothed_gain[ch, i] = gain[ch, i]

        return audio * smoothed_gain

    def normalize_loudness(
        self,
        audio: np.ndarray,
        target_lufs: float = -14.0
    ) -> np.ndarray:
        """
        Normalize to target loudness (simplified LUFS)

        Args:
            audio: Audio array
            target_lufs: Target loudness in LUFS

        Note: This is a simplified implementation. For accurate LUFS,
        use pyloudnorm library.
        """
        # Calculate current approximate loudness
        rms = np.sqrt(np.mean(audio ** 2))
        current_lufs = -0.691 + 10 * np.log10(rms ** 2 + 1e-10)

        # Calculate needed gain
        gain_db = target_lufs - current_lufs
        gain_linear = 10 ** (gain_db / 20)

        # Apply gain with headroom check
        result = audio * gain_linear
        peak = np.max(np.abs(result))

        if peak > 0.99:
            # Reduce to avoid clipping
            result = result * (0.99 / peak)

        return result

    def apply_mastering_preset(
        self,
        audio: np.ndarray,
        preset: str = "balanced"
    ) -> np.ndarray:
        """
        Apply complete mastering chain with preset

        Presets:
            - subtle: Minimal processing
            - balanced: Standard mastering
            - loud: Maximum loudness
            - broadcast: Streaming/broadcast ready (-14 LUFS)

        Args:
            audio: Audio array [channels, samples]
            preset: Mastering preset name

        Returns:
            Mastered audio
        """
        presets = {
            "subtle": {
                "eq": False,
                "multiband": False,
                "stereo_width": 1.05,
                "exciter": 0.02,
                "ceiling": -1.0,
                "target_lufs": None,
            },
            "balanced": {
                "eq": True,
                "multiband": True,
                "stereo_width": 1.1,
                "exciter": 0.05,
                "ceiling": -0.5,
                "target_lufs": None,
            },
            "loud": {
                "eq": True,
                "multiband": True,
                "stereo_width": 1.15,
                "exciter": 0.08,
                "ceiling": -0.1,
                "target_lufs": None,
            },
            "broadcast": {
                "eq": True,
                "multiband": True,
                "stereo_width": 1.05,
                "exciter": 0.03,
                "ceiling": -1.0,
                "target_lufs": -14.0,
            },
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")

        config = presets[preset]

        print(f"  Applying '{preset}' mastering preset...")

        # 1. Remove DC offset
        audio = self.remove_dc_offset(audio)

        # 2. High-pass filter
        audio = self.high_pass_filter(audio, cutoff_hz=30)

        # 3. EQ
        if config["eq"]:
            print("    - Parametric EQ")
            audio = self.parametric_eq(audio)

        # 4. Multi-band compression
        if config["multiband"]:
            print("    - Multi-band compression")
            audio = self.multiband_compress(audio)

        # 5. Harmonic enhancement
        if config["exciter"] > 0:
            print("    - Harmonic exciter")
            audio = self.harmonic_exciter(audio, amount=config["exciter"])

        # 6. Stereo enhancement
        if audio.shape[0] == 2 and config["stereo_width"] != 1.0:
            print("    - Stereo enhancement")
            audio = self.stereo_enhance(audio, width=config["stereo_width"])

        # 7. Loudness normalization (if target specified)
        if config["target_lufs"] is not None:
            print(f"    - Loudness normalization ({config['target_lufs']} LUFS)")
            audio = self.normalize_loudness(audio, target_lufs=config["target_lufs"])

        # 8. Final limiting
        print("    - Brickwall limiter")
        audio = self.brickwall_limiter(audio, ceiling_db=config["ceiling"])

        return audio


def master_audio_file(
    input_path: str,
    output_path: str = None,
    preset: str = "balanced",
    verbose: bool = True
) -> str:
    """
    Master an audio file

    Args:
        input_path: Path to input audio
        output_path: Path for output (auto-generated if None)
        preset: Mastering preset
        verbose: Print progress

    Returns:
        Path to mastered audio
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_mastered{input_path.suffix}"
    output_path = Path(output_path)

    if verbose:
        print(f"\nMastering: {input_path}")
        print(f"Output: {output_path}")
        print(f"Preset: {preset}")

    # Load audio
    waveform, sr = torchaudio.load(str(input_path))
    audio = waveform.numpy()

    # Create mastering chain
    chain = AudioMasteringChain(sample_rate=sr)

    # Analyze input
    if verbose:
        print("\nInput analysis:")
        analysis = chain.analyze_audio(audio)
        print(f"  Peak: {analysis['peak_db']:.1f} dB")
        print(f"  RMS: {analysis['rms_db']:.1f} dB")
        print(f"  Crest factor: {analysis['crest_factor_db']:.1f} dB")
        print(f"  Approx. loudness: {analysis['loudness_lufs_approx']:.1f} LUFS")

    # Apply mastering
    print("\nProcessing:")
    mastered = chain.apply_mastering_preset(audio, preset)

    # Analyze output
    if verbose:
        print("\nOutput analysis:")
        analysis = chain.analyze_audio(mastered)
        print(f"  Peak: {analysis['peak_db']:.1f} dB")
        print(f"  RMS: {analysis['rms_db']:.1f} dB")
        print(f"  Approx. loudness: {analysis['loudness_lufs_approx']:.1f} LUFS")

    # Save
    torchaudio.save(str(output_path), torch.from_numpy(mastered).float(), sr)

    if verbose:
        print(f"\n✓ Mastered audio saved to: {output_path}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Professional audio mastering for DiffRhythm outputs"
    )
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", nargs="?", help="Output audio file (optional)")
    parser.add_argument(
        "--preset", "-p",
        default="balanced",
        choices=["subtle", "balanced", "loud", "broadcast"],
        help="Mastering preset"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output"
    )

    args = parser.parse_args()

    master_audio_file(
        args.input,
        args.output,
        preset=args.preset,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
