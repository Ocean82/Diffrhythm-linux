#!/usr/bin/env python3
"""
High-Quality Song Generation Pipeline for DiffRhythm

This script provides an integrated pipeline that:
1. Builds optimized style prompts
2. Generates audio with quality presets
3. Applies professional mastering

Usage:
    python generate_high_quality.py --lyrics song.lrc --genre Pop --mood Upbeat

    # Or with preset
    python generate_high_quality.py --lyrics song.lrc --prompt "rock music" --preset high
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def build_style_prompt(genre=None, mood=None, instruments=None, tempo=None, vocals=None):
    """Build an optimized style prompt"""
    try:
        from infer.prompt_builder import StylePromptBuilder
        builder = StylePromptBuilder()

        if genre:
            # Use genre-specific template
            prompt = builder.genre_specific_prompt(genre)
            # Add additional parameters
            parts = [prompt]
            if mood and mood.lower() not in prompt.lower():
                parts.insert(1, mood.lower())
            if tempo:
                parts.append(f"{tempo} BPM" if tempo.isdigit() else tempo)
            if vocals:
                parts.append(f"{vocals} vocals")
            return ", ".join(parts)
        else:
            return builder.build_prompt(
                genre=genre,
                mood=mood,
                instruments=instruments,
                tempo=tempo,
                vocals=vocals
            )
    except ImportError:
        # Fallback to simple prompt
        parts = []
        if genre:
            parts.append(f"{genre} music")
        if mood:
            parts.append(mood.lower())
        if instruments:
            parts.append(f"featuring {', '.join(instruments)}")
        if tempo:
            parts.append(f"{tempo} BPM" if str(tempo).isdigit() else tempo)
        if vocals:
            parts.append(f"{vocals} vocals")
        parts.append("professional studio production")
        return ", ".join(parts) if parts else "pop music, professional production"


def run_generation(
    lyrics_path,
    style_prompt,
    output_dir="output",
    preset="high",
    auto_master=True,
    master_preset="balanced",
    audio_length=95
):
    """Run the full generation pipeline"""

    print("\n" + "="*70)
    print(" DiffRhythm High-Quality Generation Pipeline")
    print("="*70)

    print(f"\nConfiguration:")
    print(f"  Lyrics: {lyrics_path}")
    print(f"  Style: {style_prompt}")
    print(f"  Quality Preset: {preset}")
    print(f"  Audio Length: {audio_length}s")
    print(f"  Auto-Master: {auto_master} ({master_preset})")

    # Build command
    cmd = [
        sys.executable, "-m", "infer.infer",
        "--lrc-path", str(lyrics_path),
        "--ref-prompt", style_prompt,
        "--preset", preset,
        "--audio-length", str(audio_length),
        "--output-dir", str(output_dir),
    ]

    if auto_master:
        cmd.extend(["--auto-master", "--master-preset", master_preset])

    print(f"\nRunning generation...")
    print(f"Command: {' '.join(cmd[:6])}...")

    try:
        result = subprocess.run(cmd, check=True)
        print("\n✓ Generation complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Generation failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("\n✗ Python executable not found")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="High-Quality Song Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with genre and mood
  python generate_high_quality.py --lyrics song.lrc --genre Pop --mood Upbeat

  # Generate with custom prompt
  python generate_high_quality.py --lyrics song.lrc --prompt "epic orchestral music"

  # Generate with all options
  python generate_high_quality.py --lyrics song.lrc --genre Rock --mood Energetic \\
    --instruments "Electric Guitar,Drums" --tempo 140 --vocals male --preset maximum

Quality Presets:
  preview   - Fast, low quality (~3-5 min CPU)
  draft     - Reasonable quality (~6-10 min CPU)
  standard  - Balanced (~12-20 min CPU)
  high      - High quality (~25-40 min CPU, 2-3 min GPU) [DEFAULT]
  maximum   - Maximum quality (~50-80 min CPU, 4-6 min GPU)
  ultra     - Ultra quality (~80+ min CPU, 8+ min GPU)
        """
    )

    # Required
    parser.add_argument("--lyrics", "-l", required=True, help="Path to LRC lyrics file")

    # Style options (either --prompt or genre-based)
    style_group = parser.add_argument_group("Style Options")
    style_group.add_argument("--prompt", "-p", help="Direct style prompt (overrides genre options)")
    style_group.add_argument("--genre", "-g", help="Music genre (Pop, Rock, Jazz, etc.)")
    style_group.add_argument("--mood", "-m", help="Mood (Upbeat, Melancholic, Energetic, etc.)")
    style_group.add_argument("--instruments", "-i", help="Comma-separated instruments")
    style_group.add_argument("--tempo", "-t", help="Tempo (slow/medium/fast or BPM)")
    style_group.add_argument("--vocals", "-v", help="Vocal style (male, female, etc.)")

    # Quality options
    quality_group = parser.add_argument_group("Quality Options")
    quality_group.add_argument(
        "--preset",
        default="high",
        choices=["preview", "draft", "standard", "high", "maximum", "ultra"],
        help="Quality preset (default: high)"
    )
    quality_group.add_argument("--audio-length", type=int, default=95, help="Audio length in seconds")

    # Mastering options
    master_group = parser.add_argument_group("Mastering Options")
    master_group.add_argument("--no-master", action="store_true", help="Skip auto-mastering")
    master_group.add_argument(
        "--master-preset",
        default="balanced",
        choices=["subtle", "balanced", "loud", "broadcast"],
        help="Mastering preset (default: balanced)"
    )

    # Output options
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory")

    args = parser.parse_args()

    # Validate lyrics file
    if not os.path.exists(args.lyrics):
        print(f"✗ Lyrics file not found: {args.lyrics}")
        sys.exit(1)

    # Build style prompt
    if args.prompt:
        style_prompt = args.prompt
    else:
        instruments = args.instruments.split(",") if args.instruments else None
        style_prompt = build_style_prompt(
            genre=args.genre,
            mood=args.mood,
            instruments=instruments,
            tempo=args.tempo,
            vocals=args.vocals
        )

    print(f"\nGenerated style prompt: {style_prompt}")

    # Run generation
    success = run_generation(
        lyrics_path=args.lyrics,
        style_prompt=style_prompt,
        output_dir=args.output_dir,
        preset=args.preset,
        auto_master=not args.no_master,
        master_preset=args.master_preset,
        audio_length=args.audio_length
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
