#!/usr/bin/env python3
"""
Create simple test lyrics for debugging
"""
import os

# Create simple test lyrics
test_lyrics = """[00:00.00]Hello world
[00:05.00]This is a test
[00:10.00]Can you hear me
[00:15.00]Testing one two three
[00:20.00]Simple song for debugging"""

# Save to output directory
os.makedirs("./output", exist_ok=True)
with open("./output/test.lrc", "w", encoding="utf-8") as f:
    f.write(test_lyrics)

print("✓ Created test lyrics: ./output/test.lrc")
print("Content:")
print(test_lyrics)
