#!/bin/bash
# Find redundant files on server
# Run this directly on the server

cd /opt/diffrhythm || exit 1

echo "=========================================="
echo "Searching for Redundant Files"
echo "=========================================="
echo ""

echo "1. Checking for virtual environments..."
echo "----------------------------------------"
VENV_DIRS=$(find . -type d \( -name ".venv" -o -name "venv" -o -name "env" -o -name "ENV" \) 2>/dev/null)
if [ -n "$VENV_DIRS" ]; then
    echo "Found virtual environments:"
    echo "$VENV_DIRS"
    echo ""
    for dir in $VENV_DIRS; do
        if [ -d "$dir" ]; then
            size=$(du -sh "$dir" 2>/dev/null | awk '{print $1}')
            echo "  $dir: $size"
        fi
    done
else
    echo "✓ No virtual environments found"
fi
echo ""

echo "2. Checking for Python cache files..."
echo "----------------------------------------"
PYCACHE_COUNT=$(find . -type d -name "__pycache__" 2>/dev/null | wc -l)
PYC_COUNT=$(find . -type f -name "*.pyc" 2>/dev/null | wc -l)
PYO_COUNT=$(find . -type f -name "*.pyo" 2>/dev/null | wc -l)
echo "  __pycache__ directories: $PYCACHE_COUNT"
echo "  .pyc files: $PYC_COUNT"
echo "  .pyo files: $PYO_COUNT"
if [ "$PYCACHE_COUNT" -gt 0 ] || [ "$PYC_COUNT" -gt 0 ] || [ "$PYO_COUNT" -gt 0 ]; then
    echo "  ⚠ Python cache files found"
else
    echo "  ✓ No Python cache files"
fi
echo ""

echo "3. Checking for temporary/backup files..."
echo "----------------------------------------"
TEMP_FILES=$(find . -type f \( -name "*.bak" -o -name "*.tmp" -o -name "*.swp" -o -name "*.swo" -o -name "*~" -o -name ".DS_Store" \) 2>/dev/null)
if [ -n "$TEMP_FILES" ]; then
    echo "Found temporary/backup files:"
    echo "$TEMP_FILES" | head -20
    if [ $(echo "$TEMP_FILES" | wc -l) -gt 20 ]; then
        echo "  ... and more"
    fi
else
    echo "✓ No temporary/backup files found"
fi
echo ""

echo "4. Checking for log files..."
echo "----------------------------------------"
LOG_FILES=$(find . -type f -name "*.log" 2>/dev/null)
if [ -n "$LOG_FILES" ]; then
    echo "Found log files:"
    echo "$LOG_FILES" | head -20
    LOG_SIZE=$(find . -type f -name "*.log" -exec du -ch {} + 2>/dev/null | tail -1 | awk '{print $1}')
    echo "  Total log size: $LOG_SIZE"
else
    echo "✓ No log files found"
fi
echo ""

echo "5. Checking for build artifacts..."
echo "----------------------------------------"
BUILD_ARTIFACTS=$(find . -type d \( -name "dist" -o -name "build" -o -name "*.egg-info" \) 2>/dev/null)
if [ -n "$BUILD_ARTIFACTS" ]; then
    echo "Found build artifacts:"
    echo "$BUILD_ARTIFACTS"
    for artifact in $BUILD_ARTIFACTS; do
        if [ -e "$artifact" ]; then
            size=$(du -sh "$artifact" 2>/dev/null | awk '{print $1}')
            echo "  $artifact: $size"
        fi
    done
else
    echo "✓ No build artifacts found"
fi
echo ""

echo "6. Checking for duplicate files..."
echo "----------------------------------------"
DUPLICATES=$(find . -type f -name "*.py" 2>/dev/null | sort | uniq -d)
if [ -n "$DUPLICATES" ]; then
    echo "Found potential duplicate Python files:"
    echo "$DUPLICATES"
else
    echo "✓ No obvious duplicate files found"
fi
echo ""

echo "7. Checking for large files (>10MB)..."
echo "----------------------------------------"
LARGE_FILES=$(find . -type f -size +10M 2>/dev/null)
if [ -n "$LARGE_FILES" ]; then
    echo "Found large files:"
    for file in $LARGE_FILES; do
        size=$(du -h "$file" 2>/dev/null | awk '{print $1}')
        echo "  $file: $size"
    done
else
    echo "✓ No unusually large files found"
fi
echo ""

echo "8. Checking output/temp directories..."
echo "----------------------------------------"
OUTPUT_SIZE=$(du -sh output 2>/dev/null | awk '{print $1}')
TEMP_SIZE=$(du -sh temp 2>/dev/null | awk '{print $1}')
echo "  output/: $OUTPUT_SIZE"
echo "  temp/: $TEMP_SIZE"
OUTPUT_FILES=$(find output -type f 2>/dev/null | wc -l)
TEMP_FILES=$(find temp -type f 2>/dev/null | wc -l)
echo "  Files in output/: $OUTPUT_FILES"
echo "  Files in temp/: $TEMP_FILES"
echo ""

echo "9. Checking Docker build cache..."
echo "----------------------------------------"
DOCKER_CACHE=$(sudo docker system df 2>/dev/null | grep -i cache || echo "N/A")
echo "Docker cache:"
echo "$DOCKER_CACHE"
echo ""

echo "10. Overall disk usage..."
echo "----------------------------------------"
echo "Project directory:"
du -sh .
echo ""
echo "System disk:"
df -h / | tail -1
echo ""

echo "=========================================="
echo "Summary Complete"
echo "=========================================="
