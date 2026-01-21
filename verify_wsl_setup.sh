#!/bin/bash

echo "=== DiffRhythm WSL Setup Verification ==="
echo

# Check if we're in WSL
if [[ -f /proc/version ]] && grep -qi microsoft /proc/version; then
    echo "✓ Running in WSL"
else
    echo "✗ Not running in WSL"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>/dev/null)
if [[ $? -eq 0 ]]; then
    echo "✓ Python3 available: $python_version"
else
    echo "✗ Python3 not found"
    exit 1
fi

# Check espeak-ng
if command -v espeak-ng >/dev/null 2>&1; then
    echo "✓ espeak-ng installed"
else
    echo "✗ espeak-ng not found. Install with: sudo apt-get install espeak-ng"
    exit 1
fi

# Check virtual environment
if [[ -d "venv" ]]; then
    echo "✓ Virtual environment exists"
else
    echo "! Virtual environment not found. Create with: python3 -m venv venv"
fi

# Check requirements.txt
if [[ -f "requirements.txt" ]]; then
    echo "✓ requirements.txt exists"
else
    echo "✗ requirements.txt missing"
    exit 1
fi

# Check key directories
dirs=("config" "infer" "model" "pretrained" "scripts" "g2p")
for dir in "${dirs[@]}"; do
    if [[ -d "$dir" ]]; then
        echo "✓ Directory exists: $dir"
    else
        echo "✗ Missing directory: $dir"
        exit 1
    fi
done

# Check inference scripts
scripts=("scripts/infer_prompt_ref.sh" "scripts/infer_wav_ref.sh")
for script in "${scripts[@]}"; do
    if [[ -f "$script" ]]; then
        echo "✓ Script exists: $script"
        # Make executable
        chmod +x "$script"
    else
        echo "✗ Missing script: $script"
        exit 1
    fi
done

# Check model files
echo
echo "Checking model files..."
model_files=(
    "pretrained/models--ASLP-lab--DiffRhythm-1_2/snapshots/*/cfm_model.pt"
    "pretrained/models--ASLP-lab--DiffRhythm-vae/snapshots/*/vae_model.pt"
)

for pattern in "${model_files[@]}"; do
    if ls $pattern 1> /dev/null 2>&1; then
        echo "✓ Model found: $(basename $pattern)"
    else
        echo "✗ Model missing: $pattern"
        echo "  Download with: huggingface-cli download ASLP-lab/DiffRhythm-1_2"
    fi
done

# Check example files
echo
echo "Checking example files..."
examples=("infer/example/eg_cn_full.lrc" "infer/example/eg_en_full.lrc" "infer/example/eg_en.mp3")
for example in "${examples[@]}"; do
    if [[ -f "$example" ]]; then
        echo "✓ Example exists: $example"
    else
        echo "✗ Missing example: $example"
    fi
done

echo
echo "=== Setup Summary ==="
echo "Your DiffRhythm setup is ready for WSL!"
echo
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Install requirements: pip install -r requirements.txt"
echo "3. Test inference: bash scripts/infer_prompt_ref.sh"
echo