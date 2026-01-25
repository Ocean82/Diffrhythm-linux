#!/bin/bash
# Safely remove DiffRhythm-main directory if it's confirmed to be old/unused

set -e

DIFFRHYTHM_MAIN="$HOME/app/models/DiffRhythm-main"
CURRENT_PROJECT="/opt/diffrhythm"

echo "=========================================="
echo "Safe Removal of DiffRhythm-Main"
echo "=========================================="
echo ""

# Check if directory exists
if [ ! -d "$DIFFRHYTHM_MAIN" ]; then
    echo "Error: $DIFFRHYTHM_MAIN does not exist"
    exit 1
fi

# Show current disk usage
echo "Current disk usage:"
df -h /
echo ""

# Calculate space before
SPACE_BEFORE=$(df / | tail -1 | awk '{print $3}')

# Show directory info
echo "Directory to remove: $DIFFRHYTHM_MAIN"
SIZE=$(sudo du -sh "$DIFFRHYTHM_MAIN" 2>/dev/null | cut -f1)
echo "Size: $SIZE"
echo ""

# Verification checks
echo "Verification Checks:"
echo ""

# Check 1: Has backend directory?
if [ -d "$DIFFRHYTHM_MAIN/backend" ]; then
    echo "  ⚠ Has 'backend' directory (newer structure)"
    HAS_BACKEND=true
else
    echo "  ✓ No 'backend' directory (old structure - safe to remove)"
    HAS_BACKEND=false
fi

# Check 2: Is it the current project?
if [ "$(readlink -f "$DIFFRHYTHM_MAIN")" = "$(readlink -f "$CURRENT_PROJECT")" ]; then
    echo "  ⚠ WARNING: This is the same as current project!"
    exit 1
else
    echo "  ✓ Different from current project at $CURRENT_PROJECT"
fi

# Check 3: Docker mounts
if command -v docker &> /dev/null; then
    MOUNTED=$(sudo docker ps -a --format "{{.Mounts}}" 2>/dev/null | grep -i "DiffRhythm-main" || true)
    if [ -n "$MOUNTED" ]; then
        echo "  ⚠ WARNING: Found in Docker mounts!"
        echo "  $MOUNTED"
        exit 1
    else
        echo "  ✓ Not mounted in Docker containers"
    fi
fi

# Check 4: Last modified date
LAST_MOD=$(stat "$DIFFRHYTHM_MAIN" 2>/dev/null | grep Modify | awk '{print $2, $3}')
echo "  Last modified: $LAST_MOD"
echo ""

# Final confirmation
echo "=========================================="
echo "Ready to Remove"
echo "=========================================="
echo ""
echo "This will remove:"
echo "  - Directory: $DIFFRHYTHM_MAIN"
echo "  - Size: $SIZE"
echo "  - Structure: $(if [ "$HAS_BACKEND" = true ]; then echo "Newer (has backend)"; else echo "Old (no backend)"; fi)"
echo ""
echo "This directory is:"
echo "  - NOT the current project (/opt/diffrhythm)"
echo "  - NOT mounted in Docker"
echo "  - Located in ~/app/models (not used by deployment)"
echo ""

read -p "Remove this directory? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Removing $DIFFRHYTHM_MAIN..."
sudo rm -rf "$DIFFRHYTHM_MAIN"
echo "✓ Removed"
echo ""

# Calculate space after
SPACE_AFTER=$(df / | tail -1 | awk '{print $3}')
SPACE_FREED=$((SPACE_BEFORE - SPACE_AFTER))

echo "=========================================="
echo "Removal Complete"
echo "=========================================="
echo ""
echo "Final disk usage:"
df -h /
echo ""
echo "Space freed: ~${SPACE_FREED}KB"
echo ""
echo "✓ DiffRhythm-main directory removed successfully!"
