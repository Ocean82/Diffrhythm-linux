#!/bin/bash

echo "=== Generating 2-Minute Song with Vocals ==="
echo

cd "$(dirname "$0")"
cd ../

export PYTHONPATH=$PYTHONPATH:$PWD

# For WSL, set espeak library path if needed
if [[ "$OSTYPE" =~ ^darwin ]]; then
    export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/Cellar/espeak-ng/1.52.0/lib/libespeak-ng.dylib
fi

echo "Song: 'Shine So Bright'"
echo "Duration: ~120 seconds (2 minutes)"
echo "Style: Pop ballad with emotional vocals"
echo "Estimated time: 20-25 minutes on CPU"
echo "Memory required: ~16GB RAM"
echo

# Check memory
available_mem=$(free -m | awk 'NR==2{print $7}')
echo "Available memory: ${available_mem}MB"
if [ "$available_mem" -lt 8000 ]; then
    echo "⚠️  WARNING: Low memory. Generation may fail."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo

python3 infer/infer.py \
    --lrc-path infer/example/two_minute_song.lrc \
    --ref-prompt "pop ballad, emotional female vocals, piano, strings, uplifting" \
    --audio-length 130 \
    --output-dir infer/example/output \
    --chunked \
    --batch-infer-num 1

if [ $? -eq 0 ]; then
    echo
    echo "✓ Song generated successfully!"
    echo "Output: infer/example/output/output.wav"
    echo
    echo "Play with: ffplay infer/example/output/output.wav"
else
    echo
    echo "✗ Generation failed"
    exit 1
fi
