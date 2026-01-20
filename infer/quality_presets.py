#!/usr/bin/env python3
"""
Quality Presets for DiffRhythm Inference

This module provides predefined quality presets that balance
generation quality against inference time.

Usage:
    python -m infer.infer --lrc-path song.lrc --ref-prompt "pop" --preset high

Or programmatically:
    from infer.quality_presets import get_preset, apply_preset
    config = get_preset("high")
"""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class QualityPreset:
    """Configuration for a quality preset"""
    name: str
    steps: int
    cfg_strength: float
    description: str
    estimated_time_cpu_min: float  # Estimated time in minutes on CPU
    estimated_time_gpu_min: float  # Estimated time in minutes on GPU
    recommended_for: str


# Quality presets from lowest to highest
QUALITY_PRESETS = {
    "preview": QualityPreset(
        name="preview",
        steps=4,
        cfg_strength=2.0,
        description="Ultra-fast preview, low quality",
        estimated_time_cpu_min=3,
        estimated_time_gpu_min=0.5,
        recommended_for="Quick testing, style verification"
    ),

    "draft": QualityPreset(
        name="draft",
        steps=8,
        cfg_strength=2.5,
        description="Fast draft, acceptable quality",
        estimated_time_cpu_min=6,
        estimated_time_gpu_min=1,
        recommended_for="Iterative development, lyric testing"
    ),

    "standard": QualityPreset(
        name="standard",
        steps=16,
        cfg_strength=3.0,
        description="Balanced quality and speed",
        estimated_time_cpu_min=12,
        estimated_time_gpu_min=1.5,
        recommended_for="General use, demos"
    ),

    "high": QualityPreset(
        name="high",
        steps=32,
        cfg_strength=4.0,
        description="High quality output",
        estimated_time_cpu_min=25,
        estimated_time_gpu_min=2.5,
        recommended_for="Final production, releases"
    ),

    "maximum": QualityPreset(
        name="maximum",
        steps=64,
        cfg_strength=5.0,
        description="Maximum quality, slow",
        estimated_time_cpu_min=50,
        estimated_time_gpu_min=5,
        recommended_for="Best possible quality, no time constraints"
    ),

    "ultra": QualityPreset(
        name="ultra",
        steps=100,
        cfg_strength=6.0,
        description="Ultra quality, very slow",
        estimated_time_cpu_min=80,
        estimated_time_gpu_min=8,
        recommended_for="Research, quality comparisons"
    ),
}


def get_preset(name: str) -> QualityPreset:
    """
    Get a quality preset by name

    Args:
        name: Preset name (preview, draft, standard, high, maximum, ultra)

    Returns:
        QualityPreset configuration

    Raises:
        ValueError: If preset name is not found
    """
    name = name.lower()
    if name not in QUALITY_PRESETS:
        available = ", ".join(QUALITY_PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return QUALITY_PRESETS[name]


def get_preset_for_device(device: str = None, quality_level: str = "standard") -> QualityPreset:
    """
    Get an appropriate preset based on device and desired quality

    Args:
        device: "cuda" or "cpu" (auto-detected if None)
        quality_level: Desired quality (low, medium, high)

    Returns:
        Appropriate QualityPreset
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    quality_mapping = {
        "cpu": {
            "low": "preview",
            "medium": "draft",
            "high": "standard",
        },
        "cuda": {
            "low": "draft",
            "medium": "standard",
            "high": "high",
        }
    }

    device_key = "cuda" if "cuda" in device.lower() else "cpu"
    preset_name = quality_mapping[device_key].get(quality_level, "standard")

    return get_preset(preset_name)


def apply_preset(preset_name: str, config: dict = None) -> dict:
    """
    Apply a preset to a configuration dictionary

    Args:
        preset_name: Name of the preset to apply
        config: Existing config to update (created if None)

    Returns:
        Updated configuration dictionary
    """
    if config is None:
        config = {}

    preset = get_preset(preset_name)

    config["steps"] = preset.steps
    config["cfg_strength"] = preset.cfg_strength

    return config


def print_presets(verbose: bool = False):
    """Print all available presets"""
    print("\n" + "="*70)
    print(" DiffRhythm Quality Presets")
    print("="*70)

    headers = ["Preset", "Steps", "CFG", "CPU Time", "GPU Time", "Description"]
    print(f"\n{'Preset':12} | {'Steps':5} | {'CFG':4} | {'CPU':8} | {'GPU':8} | Description")
    print("-"*70)

    for name, preset in QUALITY_PRESETS.items():
        cpu_time = f"{preset.estimated_time_cpu_min:.0f} min"
        gpu_time = f"{preset.estimated_time_gpu_min:.1f} min"

        print(f"{name:12} | {preset.steps:5} | {preset.cfg_strength:4.1f} | {cpu_time:8} | {gpu_time:8} | {preset.description}")

        if verbose:
            print(f"             └─ Recommended for: {preset.recommended_for}")

    print()


def recommend_preset(
    target_quality: str = "good",
    max_time_minutes: float = None,
    device: str = None
) -> QualityPreset:
    """
    Recommend a preset based on requirements

    Args:
        target_quality: "fast", "good", "best"
        max_time_minutes: Maximum acceptable time in minutes
        device: "cuda" or "cpu" (auto-detected if None)

    Returns:
        Recommended QualityPreset
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    is_gpu = "cuda" in device.lower()

    # Filter by time if specified
    valid_presets = []
    for name, preset in QUALITY_PRESETS.items():
        time_estimate = preset.estimated_time_gpu_min if is_gpu else preset.estimated_time_cpu_min
        if max_time_minutes is None or time_estimate <= max_time_minutes:
            valid_presets.append(preset)

    if not valid_presets:
        print(f"Warning: No preset fits within {max_time_minutes} minutes")
        return QUALITY_PRESETS["preview"]

    # Select based on quality preference
    if target_quality == "fast":
        return valid_presets[0]  # Fastest valid
    elif target_quality == "best":
        return valid_presets[-1]  # Highest quality valid
    else:  # "good" - middle ground
        mid_idx = len(valid_presets) // 2
        return valid_presets[mid_idx]


class PresetManager:
    """Manager for quality presets with device awareness"""

    def __init__(self, device: str = None):
        """
        Initialize preset manager

        Args:
            device: Device to use (auto-detected if None)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.is_gpu = "cuda" in device.lower()

    def get_preset(self, name: str) -> QualityPreset:
        """Get a preset by name"""
        return get_preset(name)

    def get_recommended(
        self,
        quality: str = "good",
        max_time: float = None
    ) -> QualityPreset:
        """Get recommended preset based on requirements"""
        return recommend_preset(quality, max_time, self.device)

    def get_estimated_time(self, preset_name: str) -> float:
        """Get estimated time in minutes for a preset on current device"""
        preset = get_preset(preset_name)
        if self.is_gpu:
            return preset.estimated_time_gpu_min
        return preset.estimated_time_cpu_min

    def list_presets(self) -> list:
        """Get list of available preset names"""
        return list(QUALITY_PRESETS.keys())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quality presets info")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed info")
    parser.add_argument("--recommend", action="store_true",
                       help="Show recommendations based on device")
    args = parser.parse_args()

    print_presets(verbose=args.verbose)

    if args.recommend:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device detected: {device}")
        print()

        for quality in ["fast", "good", "best"]:
            preset = recommend_preset(quality, device=device)
            print(f"  {quality:6} → {preset.name} ({preset.steps} steps)")
