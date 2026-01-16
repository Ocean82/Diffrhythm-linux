#!/bin/bash

echo "=== DiffRhythm WSL Inference Test ==="
echo

# Check if in WSL
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "[OK] Running in WSL"
else
    echo "[WARN] Not in WSL, running on Windows"
fi

# Check virtual environment
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "[OK] Virtual environment active: $VIRTUAL_ENV"
else
    echo "[ERROR] No virtual environment active"
    echo "Run: source .venv/bin/activate"
    exit 1
fi

# Check espeak-ng
if command -v espeak-ng >/dev/null 2>&1; then
    echo "[OK] espeak-ng installed"
else
    echo "[ERROR] espeak-ng not found"
    echo "Install: sudo apt-get install espeak-ng"
    exit 1
fi

# Check Python packages
echo
echo "Checking Python packages..."
python3 -c "import torch; print(f'[OK] torch {torch.__version__}')" || exit 1
python3 -c "import torchaudio; print(f'[OK] torchaudio')" || exit 1
python3 -c "import librosa; print(f'[OK] librosa')" || exit 1
python3 -c "import muq; print(f'[OK] muq')" || exit 1
python3 -c "from phonemizer import phonemize; print(f'[OK] phonemizer')" || exit 1

# Test model loading
echo
echo "Testing model loading..."
python3 test_model_loading.py || exit 1

# Run actual inference test
echo
echo "Running inference test (this may take several minutes)..."
echo "Using example: eg_cn_full.lrc with prompt reference"
echo

export PYTHONPATH=$PYTHONPATH:$PWD

python3 infer/infer.py \
    --lrc-path infer/example/eg_cn_full.lrc \
    --ref-prompt "folk, acoustic guitar, harmonica, touching." \
    --audio-length 95 \
    --output-dir infer/example/output \
    --chunked \
    --batch-infer-num 1

if [ $? -eq 0 ]; then
    echo
    echo "[SUCCESS] Inference completed!"
    echo "Output saved to: infer/example/output/output.wav"
    
    if [ -f "infer/example/output/output.wav" ]; then
        size=$(stat -f%z "infer/example/output/output.wav" 2>/dev/null || stat -c%s "infer/example/output/output.wav" 2>/dev/null)
        echo "File size: $size bytes"
    fi
else
    echo
    echo "[ERROR] Inference failed"
    exit 1
fi
