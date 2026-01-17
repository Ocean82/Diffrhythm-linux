#!/bin/bash

echo "=== System Check ==="
echo

# Check memory
echo "Memory:"
free -h | grep -E "Mem|Swap"
echo

# Check if we have enough RAM (need ~16GB free)
available_mem=$(free -m | awk 'NR==2{print $7}')
if [ "$available_mem" -lt 8000 ]; then
    echo "⚠️  WARNING: Low memory ($available_mem MB available)"
    echo "   Recommended: 16GB+ free RAM"
    echo "   This may cause the process to be killed"
    echo
fi

# Check CPU
echo "CPU cores: $(nproc)"
echo

# Check disk space
echo "Disk space:"
df -h . | tail -1
echo

# Test a minimal generation first
echo "=== Running minimal test (10 seconds) ==="
echo

export PYTHONPATH=$PYTHONPATH:$PWD
# Set Hugging Face cache directory to D: drive for local storage
# On AWS, this can be set to an S3 mount point
export HF_HOME=/mnt/d/_hugging-face

# Create minimal test lyrics
cat > infer/example/test_minimal.lrc << 'EOF'
[00:00.00]Test song
[00:03.00]Just a test
[00:06.00]Very short
EOF

python3 infer/infer.py \
    --lrc-path infer/example/test_minimal.lrc \
    --ref-prompt "simple pop" \
    --audio-length 10 \
    --output-dir infer/example/output \
    --chunked \
    --batch-infer-num 1

if [ $? -eq 0 ]; then
    echo
    echo "✓ Test passed! System can handle generation."
    echo "  Now try the full 2-minute song."
else
    echo
    echo "✗ Test failed. Check errors above."
    exit 1
fi
