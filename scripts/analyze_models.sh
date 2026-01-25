#!/bin/bash
# Analyze model files to determine which are required and which can be removed

set -e

echo "=========================================="
echo "Model Files Analysis"
echo "=========================================="
echo ""

PRETRAINED_DIR="/opt/diffrhythm/pretrained"

if [ ! -d "$PRETRAINED_DIR" ]; then
    echo "Error: $PRETRAINED_DIR does not exist"
    exit 1
fi

echo "1. Required Models for DiffRhythm:"
echo "   - CFM Model: ASLP-lab/DiffRhythm-1_2 (cfm_model.pt) - ~2GB"
echo "   - CFM Model (full): ASLP-lab/DiffRhythm-1_2-full (cfm_model.pt) - ~2GB"
echo "   - VAE Model: ASLP-lab/DiffRhythm-vae (vae_model.pt)"
echo "   - MuQ-MuLan: OpenMuQ/MuQ-MuLan-large (loaded via from_pretrained)"
echo ""

echo "2. Directory Structure:"
sudo du -sh "$PRETRAINED_DIR"/* 2>/dev/null | sort -hr | head -20
echo ""

echo "3. Model Files Found:"
find "$PRETRAINED_DIR" -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" -o -name "*.safetensors" \) -exec ls -lh {} \; 2>/dev/null | awk '{print $5, $9}'
echo ""

echo "4. HuggingFace Cache Directories:"
find "$PRETRAINED_DIR" -type d -name "models--*" 2>/dev/null | while read dir; do
    size=$(sudo du -sh "$dir" 2>/dev/null | cut -f1)
    echo "   $size - $dir"
done
echo ""

echo "5. Snapshots (may contain duplicates):"
find "$PRETRAINED_DIR" -type d -name "snapshots" 2>/dev/null | while read dir; do
    echo "   $dir:"
    find "$dir" -maxdepth 2 -type d 2>/dev/null | head -5 | sed 's/^/     /'
    echo ""
done
echo ""

echo "6. Large Files (>100MB):"
find "$PRETRAINED_DIR" -type f -size +100M -exec ls -lh {} \; 2>/dev/null | awk '{print $5, $9}' | sort -hr
echo ""

echo "=========================================="
echo "Analysis Complete"
echo "=========================================="
