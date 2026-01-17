#!/bin/bash

echo "=== Generating 95-Second Song with Vocals (Memory Optimized) ==="
echo

cd "$(dirname "$0")"
cd ../

export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=/mnt/d/_hugging-face

echo "Song: 'Finding My Rhythm'"
echo "Duration: 95 seconds"
echo "Style: Uplifting acoustic pop"
echo "Memory status:"
free -h

echo

.venv/bin/python3 -u infer/infer.py \
    --lrc-path infer/example/test_95s.lrc \
    --ref-prompt "uplifting acoustic pop, male vocals, acoustic guitar, gentle drums" \
    --audio-length 95 \
    --output-dir infer/example/output \
    --chunked \
    --batch-infer-num 1

if [ $? -eq 0 ]; then
    echo
    echo "✓ 95s Song generated successfully!"
    echo "Output: infer/example/output/output.wav"
    echo
else
    echo
    echo "✗ Generation failed"
    exit 1
fi
