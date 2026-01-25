#!/bin/bash
# Analyze DiffRhythm-main directory to determine if it's safe to remove

set -e

DIFFRHYTHM_MAIN="$HOME/app/models/DiffRhythm-main"

echo "=========================================="
echo "DiffRhythm-Main Directory Analysis"
echo "=========================================="
echo ""

if [ ! -d "$DIFFRHYTHM_MAIN" ]; then
    echo "Error: $DIFFRHYTHM_MAIN does not exist"
    exit 1
fi

echo "Location: $DIFFRHYTHM_MAIN"
echo ""

# Size
echo "1. Directory Size:"
sudo du -sh "$DIFFRHYTHM_MAIN"
echo ""

# Contents
echo "2. Top-level Contents:"
ls -lah "$DIFFRHYTHM_MAIN" | head -20
echo ""

# Large subdirectories
echo "3. Large Subdirectories:"
sudo du -sh "$DIFFRHYTHM_MAIN"/* 2>/dev/null | sort -hr | head -15
echo ""

# Check if it has backend directory (current project structure)
echo "4. Project Structure Check:"
if [ -d "$DIFFRHYTHM_MAIN/backend" ]; then
    echo "   ✓ Has 'backend' directory (new structure)"
    echo "   Size: $(sudo du -sh "$DIFFRHYTHM_MAIN/backend" 2>/dev/null | cut -f1)"
else
    echo "   ✗ No 'backend' directory (old structure)"
fi
echo ""

# Check for Docker files
echo "5. Docker Files:"
if [ -f "$DIFFRHYTHM_MAIN/Dockerfile.prod" ]; then
    echo "   ✓ Has Dockerfile.prod"
elif [ -f "$DIFFRHYTHM_MAIN/Dockerfile" ]; then
    echo "   ⚠ Has old Dockerfile (not Dockerfile.prod)"
else
    echo "   ✗ No Dockerfile found"
fi
echo ""

# Check modification date
echo "6. Last Modified:"
stat "$DIFFRHYTHM_MAIN" 2>/dev/null | grep Modify || ls -ld "$DIFFRHYTHM_MAIN"
echo ""

# Compare with current project
echo "7. Comparison with Current Project:"
CURRENT_PROJECT="/opt/diffrhythm"
if [ -d "$CURRENT_PROJECT" ]; then
    echo "   Current project: $CURRENT_PROJECT"
    echo "   Size: $(sudo du -sh "$CURRENT_PROJECT" 2>/dev/null | cut -f1)"
    echo "   DiffRhythm-main: $DIFFRHYTHM_MAIN"
    echo "   Size: $(sudo du -sh "$DIFFRHYTHM_MAIN" 2>/dev/null | cut -f1)"
    echo ""
    echo "   Are they the same?"
    if [ -d "$CURRENT_PROJECT/backend" ] && [ -d "$DIFFRHYTHM_MAIN/backend" ]; then
        echo "   ⚠ Both have 'backend' directory - may be duplicates"
    else
        echo "   ✓ Different structures - likely different versions"
    fi
else
    echo "   Current project not found at $CURRENT_PROJECT"
fi
echo ""

# Check if mounted in Docker
echo "8. Docker Mount Check:"
if command -v docker &> /dev/null; then
    MOUNTED=$(sudo docker ps -a --format "{{.Mounts}}" 2>/dev/null | grep -i "DiffRhythm-main" || true)
    if [ -n "$MOUNTED" ]; then
        echo "   ⚠ WARNING: Found in Docker mounts!"
        echo "   $MOUNTED"
    else
        echo "   ✓ Not mounted in any Docker containers"
    fi
else
    echo "   Docker not available"
fi
echo ""

echo "=========================================="
echo "Conclusion"
echo "=========================================="
echo ""
echo "If this directory:"
echo "  - Has no 'backend' directory → Old version, safe to remove"
echo "  - Has 'backend' directory but different from /opt/diffrhythm → Old copy, safe to remove"
echo "  - Is NOT mounted in Docker → Safe to remove"
echo "  - Is in ~/app/models (not /opt/diffrhythm) → Not used by current deployment"
echo ""
echo "Recommendation: Safe to remove if not actively used"
