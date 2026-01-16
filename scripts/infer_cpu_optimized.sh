#!/bin/bash
# CPU-optimized inference settings

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

python3 infer/infer.py \
    --lrc-path "$1" \
    --ref-prompt "$2" \
    --audio-length 95 \
    --output-dir infer/example/output \
    --chunked \
    --batch-infer-num 1

# Expected time: 15-25 minutes per song on modern CPU
