#!/bin/bash
# Check if diffrhythm-main directory exists and if it's safe to remove

set -e

echo "=========================================="
echo "DiffRhythm-Main Directory Investigation"
echo "=========================================="
echo ""

# Check various possible locations
LOCATIONS=(
    "/opt/diffrhythm-main"
    "/opt/diffrhythm/diffrhythm-main"
    "/home/ubuntu/diffrhythm-main"
    "/root/diffrhythm-main"
    "/tmp/diffrhythm-main"
)

FOUND=0

for loc in "${LOCATIONS[@]}"; do
    if [ -d "$loc" ]; then
        echo "✓ Found: $loc"
        SIZE=$(sudo du -sh "$loc" 2>/dev/null | cut -f1)
        echo "  Size: $SIZE"
        echo "  Contents:"
        ls -lah "$loc" | head -10 | sed 's/^/    /'
        echo ""
        FOUND=1
    fi
done

if [ $FOUND -eq 0 ]; then
    echo "✗ No 'diffrhythm-main' directory found in common locations"
    echo ""
fi

# Check current project directory
echo "Current project directory: /opt/diffrhythm"
echo "Size: $(sudo du -sh /opt/diffrhythm 2>/dev/null | cut -f1)"
echo ""

# Check Docker build context
echo "Docker Build Context:"
echo "  - Dockerfile.prod uses: COPY . ."
echo "  - This copies current directory (where docker build is run)"
echo "  - Project directory: /opt/diffrhythm"
echo ""

# Check if diffrhythm-main is referenced
echo "Checking for references to 'diffrhythm-main':"
if grep -r "diffrhythm-main" /opt/diffrhythm 2>/dev/null | head -5; then
    echo "  ⚠ Found references"
else
    echo "  ✓ No references found in /opt/diffrhythm"
fi
echo ""

echo "=========================================="
echo "Conclusion"
echo "=========================================="
if [ $FOUND -eq 0 ]; then
    echo "No 'diffrhythm-main' directory exists."
    echo "Current project is at: /opt/diffrhythm"
    echo ""
    echo "If you find a 'diffrhythm-main' directory elsewhere:"
    echo "  - It's likely an old/unused directory"
    echo "  - Safe to remove if not referenced in Docker builds"
    echo "  - Docker builds from /opt/diffrhythm, not diffrhythm-main"
else
    echo "Found diffrhythm-main directory(s) listed above."
    echo "These are likely old/unused and can be safely removed."
fi
