#!/bin/bash
# Safely remove unused model files
# Only removes models that are confirmed not to be used by DiffRhythm

set -e

PRETRAINED_DIR="/opt/diffrhythm/pretrained"

echo "=========================================="
echo "Safe Model Removal"
echo "=========================================="
echo ""

# Check if pretrained directory exists
if [ ! -d "$PRETRAINED_DIR" ]; then
    echo "Error: $PRETRAINED_DIR does not exist"
    exit 1
fi

# Show current disk usage
echo "Current disk usage:"
df -h /
echo ""

# Calculate space before
SPACE_BEFORE=$(df / | tail -1 | awk '{print $3}')

echo "Models to remove (confirmed unused):"
echo ""

# 1. Remove MuQ-large-msd-iter (not used in current codebase)
if [ -d "$PRETRAINED_DIR/models--OpenMuQ--MuQ-large-msd-iter" ]; then
    SIZE=$(sudo du -sh "$PRETRAINED_DIR/models--OpenMuQ--MuQ-large-msd-iter" 2>/dev/null | cut -f1)
    echo "1. Removing OpenMuQ/MuQ-large-msd-iter ($SIZE) - NOT USED"
    sudo rm -rf "$PRETRAINED_DIR/models--OpenMuQ--MuQ-large-msd-iter"
    sudo rm -rf "$PRETRAINED_DIR/.locks/models--OpenMuQ--MuQ-large-msd-iter" 2>/dev/null || true
    echo "   ✓ Removed"
    echo ""
fi

# 2. Remove xlm-roberta-base (not used in current codebase)
if [ -d "$PRETRAINED_DIR/models--xlm-roberta-base" ]; then
    SIZE=$(sudo du -sh "$PRETRAINED_DIR/models--xlm-roberta-base" 2>/dev/null | cut -f1)
    echo "2. Removing xlm-roberta-base ($SIZE) - NOT USED"
    sudo rm -rf "$PRETRAINED_DIR/models--xlm-roberta-base"
    sudo rm -rf "$PRETRAINED_DIR/.locks/models--xlm-roberta-base" 2>/dev/null || true
    echo "   ✓ Removed"
    echo ""
fi

# 3. Optional: Remove DiffRhythm-1_2-full if only using 95s songs
# Uncomment if you only generate 95-second songs
# if [ -d "$PRETRAINED_DIR/models--ASLP-lab--DiffRhythm-1_2-full" ]; then
#     SIZE=$(sudo du -sh "$PRETRAINED_DIR/models--ASLP-lab--DiffRhythm-1_2-full" 2>/dev/null | cut -f1)
#     echo "3. Removing DiffRhythm-1_2-full ($SIZE) - Only needed for songs >95s"
#     read -p "   Remove DiffRhythm-1_2-full? (y/N): " -n 1 -r
#     echo
#     if [[ $REPLY =~ ^[Yy]$ ]]; then
#         sudo rm -rf "$PRETRAINED_DIR/models--ASLP-lab--DiffRhythm-1_2-full"
#         sudo rm -rf "$PRETRAINED_DIR/.locks/models--ASLP-lab--DiffRhythm-1_2-full" 2>/dev/null || true
#         echo "   ✓ Removed"
#     else
#         echo "   ✗ Skipped"
#     fi
#     echo ""
# fi

# Calculate space after
SPACE_AFTER=$(df / | tail -1 | awk '{print $3}')
SPACE_FREED=$((SPACE_BEFORE - SPACE_AFTER))

echo "=========================================="
echo "Cleanup Complete"
echo "=========================================="
echo ""
echo "Final disk usage:"
df -h /
echo ""
echo "Space freed: ~${SPACE_FREED}KB"
echo ""
echo "Required models kept:"
echo "  ✓ ASLP-lab/DiffRhythm-1_2 (95s songs)"
echo "  ✓ ASLP-lab/DiffRhythm-vae"
echo "  ✓ OpenMuQ/MuQ-MuLan-large"
echo ""
if [ -d "$PRETRAINED_DIR/models--ASLP-lab--DiffRhythm-1_2-full" ]; then
    echo "  ✓ ASLP-lab/DiffRhythm-1_2-full (longer songs)"
fi
