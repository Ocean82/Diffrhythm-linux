#!/bin/bash
# Cleanup script to remove old/broken code and unnecessary files from server
# Usage: bash scripts/cleanup_server.sh [--dry-run]

set -e

EC2_HOST="52.0.207.242"
EC2_USER="ubuntu"
SSH_KEY="$HOME/server_saver_key"
PROJECT_DIR="/opt/diffrhythm"

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - No files will be deleted ==="
fi

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=========================================="
echo "Server Cleanup Script"
echo "=========================================="
echo "Server: $EC2_USER@$EC2_HOST"
echo "Project: $PROJECT_DIR"
echo ""

# Function to calculate space that will be freed
calculate_space() {
    ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "cd $PROJECT_DIR && du -sh $1 2>/dev/null | awk '{print \$1}' || echo '0'"
}

# Function to remove files/directories
remove_if_exists() {
    local path="$1"
    local description="$2"
    
    echo -e "${YELLOW}Checking: $description${NC}"
    local size=$(ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "cd $PROJECT_DIR && [ -e '$path' ] && du -sh '$path' 2>/dev/null | awk '{print \$1}' || echo 'NOT_FOUND'")
    
    if [[ "$size" != "NOT_FOUND" ]]; then
        echo -e "  Found: $path (${size})"
        if [[ "$DRY_RUN" == "false" ]]; then
            ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "cd $PROJECT_DIR && rm -rf '$path'"
            echo -e "  ${GREEN}✓ Removed${NC}"
        else
            echo -e "  ${YELLOW}[DRY RUN] Would remove${NC}"
        fi
    else
        echo -e "  ${GREEN}✓ Not found (already clean)${NC}"
    fi
    echo ""
}

echo "Step 1: Removing Python cache files..."
if [[ "$DRY_RUN" == "false" ]]; then
    ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "cd $PROJECT_DIR && find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true && find . -type f -name '*.pyc' -delete && find . -type f -name '*.pyo' -delete"
    echo -e "${GREEN}✓ Python cache cleaned${NC}"
else
    echo -e "${YELLOW}[DRY RUN] Would remove Python cache files${NC}"
fi
echo ""

echo "Step 2: Removing development/investigation documentation..."
# Remove investigation/analysis markdown files (keep only essential docs)
# Use find to match patterns properly
if [[ "$DRY_RUN" == "false" ]]; then
    ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "cd $PROJECT_DIR && find . -maxdepth 1 -type f \( \
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
        -name 'FIXES_VISUAL_SUMMARY.txt' \
    \) ! -name 'DEPLOYMENT.md' ! -name 'README.md' ! -name 'Readme.md' -delete"
    echo -e "${GREEN}✓ Investigation documentation cleaned${NC}"
else
    echo -e "${YELLOW}[DRY RUN] Would remove investigation markdown files${NC}"
    ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "cd $PROJECT_DIR && find . -maxdepth 1 -type f \( \
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
        -name 'FIXES_VISUAL_SUMMARY.txt' \
    \) ! -name 'DEPLOYMENT.md' ! -name 'README.md' ! -name 'Readme.md' | head -20"
fi

# Keep essential docs
KEEP_DOCS=("DEPLOYMENT.md" "README.md" "Readme.md" "LICENSE" "LICENSE.md")
echo -e "${GREEN}Keeping essential docs: ${KEEP_DOCS[*]}${NC}"
echo ""

echo "Step 3: Removing development/debug scripts..."
DEV_SCRIPTS=(
    "fix_*.py"
    "check_*.py"
    "verify_*.py"
    "debug_*.py"
    "test_*.py"
    "diagnose_*.py"
    "analyze_*.py"
    "quick_*.py"
)

for pattern in "${DEV_SCRIPTS[@]}"; do
    remove_if_exists "$pattern" "Development scripts: $pattern"
done

# Keep essential scripts
KEEP_SCRIPTS=("generate_verification_95s_song.py")
echo -e "${GREEN}Keeping essential scripts: ${KEEP_SCRIPTS[*]}${NC}"
echo ""

echo "Step 4: Removing training/development directories..."
remove_if_exists "ckpts" "Model checkpoints (not needed for production)"
remove_if_exists "dataset" "Training dataset (not needed for production)"
remove_if_exists "train" "Training code (not needed for production)"
remove_if_exists "wandb" "Weights & Biases logs"
echo ""

echo "Step 5: Removing IDE/development directories..."
remove_if_exists ".continue" "Continue IDE directory"
remove_if_exists ".cursor" "Cursor IDE directory"
remove_if_exists ".refact" "Refact IDE directory"
remove_if_exists ".github" "GitHub workflows (not needed on server)"
remove_if_exists ".vscode" "VS Code settings"
remove_if_exists ".idea" "IntelliJ IDEA settings"
echo ""

echo "Step 6: Removing old Docker files..."
remove_if_exists "Dockerfile" "Old Dockerfile (using Dockerfile.prod)"
remove_if_exists "docker" "Old docker directory"
remove_if_exists "docker-compose.yml" "Old docker-compose.yml (using docker-compose.prod.yml)"
echo ""

echo "Step 7: Removing thirdparty (will be installed via pip)..."
remove_if_exists "thirdparty" "Third-party code (installed via pip)"
echo ""

echo "Step 8: Cleaning up logs..."
if [[ "$DRY_RUN" == "false" ]]; then
    ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "cd $PROJECT_DIR && find . -type f -name '*.log' -delete 2>/dev/null || true"
    echo -e "${GREEN}✓ Log files cleaned${NC}"
else
    echo -e "${YELLOW}[DRY RUN] Would remove log files${NC}"
fi
echo ""

echo "Step 9: Checking for large files..."
echo "Large files (>10MB):"
ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "cd $PROJECT_DIR && find . -type f -size +10M 2>/dev/null | head -10"
echo ""

echo "Step 10: Final disk usage..."
ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" "cd $PROJECT_DIR && du -sh . && df -h /"

echo ""
echo "=========================================="
if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}DRY RUN COMPLETE${NC}"
    echo "Run without --dry-run to actually remove files"
else
    echo -e "${GREEN}CLEANUP COMPLETE${NC}"
fi
echo "=========================================="
