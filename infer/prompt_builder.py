#!/usr/bin/env python3
"""
Intelligent Style Prompt Builder for DiffRhythm

This module helps create rich, consistent style prompts that improve
generation quality by providing detailed musical descriptions to MuQ-MuLan.

Usage:
    from infer.prompt_builder import StylePromptBuilder

    builder = StylePromptBuilder()
    prompt = builder.build_prompt(
        genre="Pop",
        mood="Upbeat",
        instruments=["Piano", "Synthesizer"],
        tempo="120",
        vocals="female",
        production="professional studio"
    )
"""

import json
import random
from pathlib import Path
from typing import List, Optional


class StylePromptBuilder:
    """Build rich, consistent style prompts for better generation quality"""

    # Production quality descriptors
    PRODUCTION_QUALITIES = [
        "professional studio production",
        "pristine audio quality",
        "polished mix",
        "radio-ready production",
        "high-fidelity recording",
        "crisp and clear sound",
        "well-balanced mix",
    ]

    # Tempo descriptors
    TEMPO_DESCRIPTORS = {
        "slow": ["slow", "ballad", "laid-back", "relaxed tempo"],
        "medium": ["medium tempo", "moderate pace", "steady groove"],
        "fast": ["fast", "upbeat", "energetic tempo", "driving rhythm"],
        "very_fast": ["very fast", "high energy", "intense tempo"],
    }

    # Vocal style descriptors
    VOCAL_STYLES = [
        "male vocals",
        "female vocals",
        "mixed vocals",
        "choir",
        "harmonized vocals",
        "soulful vocals",
        "powerful vocals",
        "soft vocals",
        "raspy vocals",
        "clear vocals",
    ]

    def __init__(self, keywords_path: str = None):
        """
        Initialize the prompt builder

        Args:
            keywords_path: Path to music_prompt_keywords.json
        """
        # Default path
        if keywords_path is None:
            base_dir = Path(__file__).parent
            keywords_path = base_dir / "example" / "music_prompt_keywords.json"

        self.keywords = self._load_keywords(keywords_path)

    def _load_keywords(self, path) -> dict:
        """Load keywords from JSON file"""
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Keywords file not found at {path}, using defaults")
            return {
                "genre": ["Pop", "Rock", "Electronic", "Jazz", "R&B", "Hip-Hop"],
                "mood": ["Upbeat", "Melancholic", "Energetic", "Relaxed", "Romantic"],
                "instrument": ["Piano", "Guitar", "Drums", "Synthesizer", "Bass", "Strings"]
            }

    def build_prompt(
        self,
        genre: str = None,
        mood: str = None,
        instruments: List[str] = None,
        tempo: str = None,
        vocals: str = None,
        production: str = None,
        additional: List[str] = None
    ) -> str:
        """
        Build a comprehensive style prompt

        Args:
            genre: Music genre (Pop, Rock, Electronic, etc.)
            mood: Emotional mood (Upbeat, Melancholic, etc.)
            instruments: List of instruments to feature
            tempo: Tempo description (slow/medium/fast) or BPM number
            vocals: Vocal style description
            production: Production quality description
            additional: Additional descriptors to include

        Returns:
            Rich prompt string optimized for MuQ-MuLan

        Example:
            >>> builder = StylePromptBuilder()
            >>> prompt = builder.build_prompt(
            ...     genre="Pop",
            ...     mood="Upbeat",
            ...     instruments=["Piano", "Synthesizer"],
            ...     tempo="120",
            ...     vocals="female"
            ... )
            >>> print(prompt)
            "Pop music, upbeat, featuring Piano and Synthesizer, 120 BPM, female vocals, professional studio production"
        """
        parts = []

        # Genre (required for best results)
        if genre:
            # Validate genre
            if genre not in self.keywords.get("genre", []):
                similar = self._find_similar(genre, self.keywords.get("genre", []))
                if similar:
                    print(f"Note: '{genre}' not found, using similar: '{similar}'")
                    genre = similar
            parts.append(f"{genre} music")
        else:
            # Default genre
            parts.append("contemporary music")

        # Mood
        if mood:
            parts.append(mood.lower())

        # Instruments
        if instruments:
            # Validate and limit instruments
            valid_instruments = []
            for inst in instruments[:4]:  # Limit to 4
                if inst in self.keywords.get("instrument", []):
                    valid_instruments.append(inst)
                else:
                    similar = self._find_similar(inst, self.keywords.get("instrument", []))
                    if similar:
                        valid_instruments.append(similar)

            if valid_instruments:
                if len(valid_instruments) == 1:
                    parts.append(f"featuring {valid_instruments[0]}")
                else:
                    inst_str = ", ".join(valid_instruments[:-1]) + f" and {valid_instruments[-1]}"
                    parts.append(f"featuring {inst_str}")

        # Tempo
        if tempo:
            if tempo.isdigit():
                parts.append(f"{tempo} BPM")
            elif tempo.lower() in self.TEMPO_DESCRIPTORS:
                parts.append(random.choice(self.TEMPO_DESCRIPTORS[tempo.lower()]))
            else:
                parts.append(f"{tempo} tempo")

        # Vocals
        if vocals:
            if "vocal" not in vocals.lower():
                vocals = f"{vocals} vocals"
            parts.append(vocals)

        # Production quality
        if production:
            parts.append(production)
        else:
            parts.append(random.choice(self.PRODUCTION_QUALITIES))

        # Additional descriptors
        if additional:
            parts.extend(additional)

        return ", ".join(parts)

    def _find_similar(self, query: str, options: List[str]) -> Optional[str]:
        """Find similar option using simple string matching"""
        query_lower = query.lower()
        for opt in options:
            if query_lower in opt.lower() or opt.lower() in query_lower:
                return opt
        return None

    def random_prompt(self, base_genre: str = None, ensure_quality: bool = True) -> str:
        """
        Generate a random high-quality prompt

        Args:
            base_genre: Optional genre to base the prompt on
            ensure_quality: Whether to always include production quality

        Returns:
            Random style prompt
        """
        genre = base_genre or random.choice(self.keywords.get("genre", ["Pop"]))
        mood = random.choice(self.keywords.get("mood", ["Upbeat"]))
        instruments = random.sample(
            self.keywords.get("instrument", ["Piano", "Guitar"]),
            k=min(2, len(self.keywords.get("instrument", [])))
        )

        return self.build_prompt(
            genre=genre,
            mood=mood,
            instruments=instruments,
            production="pristine" if ensure_quality else None
        )

    def genre_specific_prompt(self, genre: str) -> str:
        """
        Generate a genre-optimized prompt with typical characteristics

        Args:
            genre: Target genre

        Returns:
            Genre-appropriate prompt
        """
        genre_configs = {
            "Pop": {
                "mood": "Upbeat",
                "instruments": ["Synthesizer", "Drums", "Bass"],
                "tempo": "medium",
                "vocals": "clear",
            },
            "Rock": {
                "mood": "Energetic",
                "instruments": ["Electric Guitar", "Drums", "Bass Guitar"],
                "tempo": "fast",
                "vocals": "powerful",
            },
            "Jazz": {
                "mood": "Mellow",
                "instruments": ["Piano", "Saxophone", "Bass"],
                "tempo": "medium",
                "vocals": "soulful",
            },
            "Electronic": {
                "mood": "Energetic",
                "instruments": ["Synthesizer", "Drums"],
                "tempo": "fast",
                "production": "crisp electronic production",
            },
            "R&B": {
                "mood": "Romantic",
                "instruments": ["Piano", "Synthesizer", "Bass"],
                "tempo": "medium",
                "vocals": "soulful",
            },
            "Hip-Hop": {
                "mood": "Energetic",
                "instruments": ["Drums", "Synthesizer", "Bass"],
                "tempo": "medium",
                "vocals": "rhythmic",
            },
            "Classical": {
                "mood": "Emotional",
                "instruments": ["Piano", "Violin", "Cello", "Strings"],
                "tempo": "medium",
                "production": "orchestral recording",
            },
            "Folk": {
                "mood": "Heartfelt",
                "instruments": ["Acoustic Guitar", "Violin", "Harmonica"],
                "tempo": "medium",
                "vocals": "warm",
            },
            "Country": {
                "mood": "Nostalgic",
                "instruments": ["Acoustic Guitar", "Banjo", "Violin"],
                "tempo": "medium",
                "vocals": "heartfelt",
            },
            "Metal": {
                "mood": "Aggressive",
                "instruments": ["Electric Guitar", "Drums", "Bass Guitar"],
                "tempo": "very_fast",
                "vocals": "powerful",
            },
        }

        config = genre_configs.get(genre, {
            "mood": "Energetic",
            "instruments": ["Piano", "Guitar"],
            "tempo": "medium",
        })

        return self.build_prompt(genre=genre, **config)

    def get_available_options(self) -> dict:
        """Get all available options for building prompts"""
        return {
            "genres": self.keywords.get("genre", []),
            "moods": self.keywords.get("mood", []),
            "instruments": self.keywords.get("instrument", []),
            "tempos": list(self.TEMPO_DESCRIPTORS.keys()),
            "vocal_styles": self.VOCAL_STYLES,
            "production_qualities": self.PRODUCTION_QUALITIES,
        }


def interactive_prompt_builder():
    """Interactive command-line prompt builder"""
    builder = StylePromptBuilder()
    options = builder.get_available_options()

    print("\n" + "="*60)
    print("DiffRhythm Style Prompt Builder")
    print("="*60)

    print("\nAvailable Genres:", ", ".join(options["genres"][:10]), "...")
    genre = input("Enter genre (or press Enter for random): ").strip() or None

    print("\nAvailable Moods:", ", ".join(options["moods"][:10]), "...")
    mood = input("Enter mood (or press Enter for random): ").strip() or None

    print("\nAvailable Instruments:", ", ".join(options["instruments"][:10]), "...")
    instruments_input = input("Enter instruments (comma-separated, or Enter for random): ").strip()
    instruments = [i.strip() for i in instruments_input.split(",")] if instruments_input else None

    tempo = input("Enter tempo (slow/medium/fast or BPM, or Enter to skip): ").strip() or None
    vocals = input("Enter vocal style (or Enter to skip): ").strip() or None

    prompt = builder.build_prompt(
        genre=genre,
        mood=mood,
        instruments=instruments,
        tempo=tempo,
        vocals=vocals
    )

    print("\n" + "="*60)
    print("Generated Prompt:")
    print("="*60)
    print(f"\n{prompt}\n")

    return prompt


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_prompt_builder()
    else:
        # Demo
        builder = StylePromptBuilder()

        print("Style Prompt Builder Demo")
        print("="*60)

        # Custom prompt
        prompt1 = builder.build_prompt(
            genre="Pop",
            mood="Upbeat",
            instruments=["Piano", "Synthesizer", "Drums"],
            tempo="120",
            vocals="female"
        )
        print(f"\nCustom prompt:\n  {prompt1}")

        # Random prompt
        prompt2 = builder.random_prompt()
        print(f"\nRandom prompt:\n  {prompt2}")

        # Genre-specific prompt
        prompt3 = builder.genre_specific_prompt("Jazz")
        print(f"\nJazz-specific prompt:\n  {prompt3}")

        print("\nRun with --interactive for interactive mode")
