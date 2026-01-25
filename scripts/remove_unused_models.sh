#!/bin/bash
# Remove unused model directories to free disk space
# Safely removes directories that are not in use

set -e

echo "=========================================="
echo "Unused Models Cleanup"
echo "=========================================="
echo ""

# Show current disk usage
echo "Current disk usage:"
df -h /
echo ""

# Calculate space before
SPACE_BEFORE=$(df / | tail -1 | awk '{print $3}')

MODELS_DIR="$HOME/app/models"
TOTAL_FREED=0

echo "Analyzing model directories in $MODELS_DIR..."
echo ""

# 1. Remove DiffRhythm-main (confirmed old/unused)
if [ -d "$MODELS_DIR/DiffRhythm-main" ]; then
    SIZE=$(sudo du -sh "$MODELS_DIR/DiffRhythm-main" 2>/dev/null | cut -f1)
    SIZE_KB=$(sudo du -sk "$MODELS_DIR/DiffRhythm-main" 2>/dev/null | cut -f1)
    echo "1. Removing DiffRhythm-main ($SIZE) - Old version, not used by Docker"
    sudo rm -rf "$MODELS_DIR/DiffRhythm-main"
    TOTAL_FREED=$((TOTAL_FREED + SIZE_KB))
    echo "   ✓ Removed"
    echo ""
fi

# 2. Check FIREGIRL SINGER-12B-GGUF
if [ -d "$MODELS_DIR/FIREGIRL SINGER-12B-GGUF" ]; then
    SIZE=$(sudo du -sh "$MODELS_DIR/FIREGIRL SINGER-12B-GGUF" 2>/dev/null | cut -f1)
    SIZE_KB=$(sudo du -sk "$MODELS_DIR/FIREGIRL SINGER-12B-GGUF" 2>/dev/null | cut -f1)
    
    # Check if it's referenced anywhere
    REFERENCED=$(grep -r "FIREGIRL\|SINGER-12B" /opt/diffrhythm 2>/dev/null | wc -l || echo "0")
    DOCKER_MOUNT=$(sudo docker ps -a --format "{{.Mounts}}" 2>/dev/null | grep -i "FIREGIRL" || true)
    
    if [ "$REFERENCED" -eq 0 ] && [ -z "$DOCKER_MOUNT" ]; then
        echo "2. Removing FIREGIRL SINGER-12B-GGUF ($SIZE) - Not referenced"
        sudo rm -rf "$MODELS_DIR/FIREGIRL SINGER-12B-GGUF"
        TOTAL_FREED=$((TOTAL_FREED + SIZE_KB))
        echo "   ✓ Removed"
    else
        echo "2. Keeping FIREGIRL SINGER-12B-GGUF ($SIZE) - May be in use"
        if [ "$REFERENCED" -gt 0 ]; then
            echo "   ⚠ Found $REFERENCED references"
        fi
        if [ -n "$DOCKER_MOUNT" ]; then
            echo "   ⚠ Found in Docker mounts"
        fi
    fi
    echo ""
fi

# 3. Check SongGeneration
if [ -d "$MODELS_DIR/SongGeneration" ]; then
    SIZE=$(sudo du -sh "$MODELS_DIR/SongGeneration" 2>/dev/null | cut -f1)
    SIZE_KB=$(sudo du -sk "$MODELS_DIR/SongGeneration" 2>/dev/null | cut -f1)
    
    # Check if it's referenced anywhere
    REFERENCED=$(grep -r "SongGeneration" /opt/diffrhythm 2>/dev/null | wc -l || echo "0")
    DOCKER_MOUNT=$(sudo docker ps -a --format "{{.Mounts}}" 2>/dev/null | grep -i "SongGeneration" || true)
    
    if [ "$REFERENCED" -eq 0 ] && [ -z "$DOCKER_MOUNT" ]; then
        echo "3. Removing SongGeneration ($SIZE) - Not referenced"
        sudo rm -rf "$MODELS_DIR/SongGeneration"
        TOTAL_FREED=$((TOTAL_FREED + SIZE_KB))
        echo "   ✓ Removed"
    else
        echo "3. Keeping SongGeneration ($SIZE) - May be in use"
        if [ "$REFERENCED" -gt 0 ]; then
            echo "   ⚠ Found $REFERENCED references"
        fi
        if [ -n "$DOCKER_MOUNT" ]; then
            echo "   ⚠ Found in Docker mounts"
        fi
    fi
    echo ""
fi

# 4. Check ai directory
if [ -d "$MODELS_DIR/ai" ]; then
    SIZE=$(sudo du -sh "$MODELS_DIR/ai" 2>/dev/null | cut -f1)
    SIZE_KB=$(sudo du -sk "$MODELS_DIR/ai" 2>/dev/null | cut -f1)
    
    # Check if it's referenced anywhere
    REFERENCED=$(grep -r "\"$MODELS_DIR/ai\"\|~/app/models/ai" /opt/diffrhythm 2>/dev/null | wc -l || echo "0")
    DOCKER_MOUNT=$(sudo docker ps -a --format "{{.Mounts}}" 2>/dev/null | grep -i "/app/models/ai" || true)
    
    if [ "$REFERENCED" -eq 0 ] && [ -z "$DOCKER_MOUNT" ]; then
        echo "4. Removing ai directory ($SIZE) - Not referenced"
        sudo rm -rf "$MODELS_DIR/ai"
        TOTAL_FREED=$((TOTAL_FREED + SIZE_KB))
        echo "   ✓ Removed"
    else
        echo "4. Keeping ai directory ($SIZE) - May be in use"
        if [ "$REFERENCED" -gt 0 ]; then
            echo "   ⚠ Found $REFERENCED references"
        fi
        if [ -n "$DOCKER_MOUNT" ]; then
            echo "   ⚠ Found in Docker mounts"
        fi
    fi
    echo ""
fi

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
echo "Total space freed: ~${SPACE_FREED}KB (~$((SPACE_FREED / 1024 / 1024))GB)"
echo ""
echo "Removed directories:"
echo "  ✓ DiffRhythm-main (old version)"
if [ -d "$MODELS_DIR/FIREGIRL SINGER-12B-GGUF" ]; then
    echo "  ⚠ FIREGIRL SINGER-12B-GGUF (kept - may be in use)"
else
    echo "  ✓ FIREGIRL SINGER-12B-GGUF (removed)"
fi
if [ -d "$MODELS_DIR/SongGeneration" ]; then
    echo "  ⚠ SongGeneration (kept - may be in use)"
else
    echo "  ✓ SongGeneration (removed)"
fi
if [ -d "$MODELS_DIR/ai" ]; then
    echo "  ⚠ ai (kept - may be in use)"
else
    echo "  ✓ ai (removed)"
fi
echo ""
