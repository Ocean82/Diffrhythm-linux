#!/bin/bash
# Direct cleanup script to run on server
# Removes old/broken code and unnecessary files

cd /opt/diffrhythm || exit 1

echo "=== Cleaning up server ==="

# Remove training directories
echo "Removing training directories..."
rm -rf ckpts dataset train 2>/dev/null
echo "✓ Training directories removed"

# Remove investigation markdown files (keep essential)
echo "Removing investigation documentation..."
find . -maxdepth 1 -type f \( \
    -name '*_INVESTIGATION*.md' -o \
    -name '*_ANALYSIS*.md' -o \
    -name '*_REPORT*.md' -o \
    -name '*_SUMMARY*.md' -o \
    -name '*_GUIDE*.md' -o \
    -name '*_STATUS*.md' -o \
    -name '*_TROUBLESHOOTING*.md' -o \
    -name '*_FIXES*.md' -o \
    -name '*_CHANGES*.md' -o \
    -name '*_REFERENCE*.md' -o \
    -name '*_INDEX*.md' -o \
    -name '*_BREAKDOWN*.md' -o \
    -name '*_INTEGRATION*.md' -o \
    -name '*_IMPROVEMENTS*.md' -o \
    -name 'DEPLOYMENT_READY.md' -o \
    -name 'FIXES_VISUAL_SUMMARY.txt' \
\) ! -name 'DEPLOYMENT.md' ! -name 'README.md' ! -name 'Readme.md' -delete 2>/dev/null
echo "✓ Investigation docs removed"

# Remove Python cache
echo "Removing Python cache..."
find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find . -type f -name '*.pyc' -delete 2>/dev/null || true
find . -type f -name '*.pyo' -delete 2>/dev/null || true
echo "✓ Python cache removed"

# Show final status
echo ""
echo "=== Final Status ==="
du -sh .
df -h / | tail -1
echo ""
echo "=== Remaining markdown files ==="
ls -1 *.md 2>/dev/null | head -10 || echo "None found"
echo ""
echo "=== Cleanup complete ==="
