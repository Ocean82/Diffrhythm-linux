#!/bin/bash
# Verify production setup is complete

echo "=========================================="
echo "DiffRhythm Production Setup Verification"
echo "=========================================="

ERRORS=0

# Check backend files
echo "Checking backend files..."
BACKEND_FILES=(
    "backend/__init__.py"
    "backend/api.py"
    "backend/config.py"
    "backend/exceptions.py"
    "backend/logging_config.py"
    "backend/metrics.py"
    "backend/security.py"
)

for file in "${BACKEND_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (MISSING)"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check Docker files
echo ""
echo "Checking Docker files..."
DOCKER_FILES=(
    "Dockerfile.prod"
    "docker-compose.prod.yml"
    ".dockerignore"
)

for file in "${DOCKER_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (MISSING)"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check deployment files
echo ""
echo "Checking deployment files..."
DEPLOY_FILES=(
    "scripts/deploy.sh"
    "scripts/ec2-setup.sh"
    "scripts/health-check.sh"
    "config/ec2-config.env"
    "config/nginx.conf"
    "config/systemd/diffrhythm.service"
    "DEPLOYMENT.md"
)

for file in "${DEPLOY_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (MISSING)"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check test files
echo ""
echo "Checking test files..."
if [ -f "tests/test_api.py" ]; then
    echo "  ✓ tests/test_api.py"
else
    echo "  ✗ tests/test_api.py (MISSING)"
    ERRORS=$((ERRORS + 1))
fi

# Check Python syntax
echo ""
echo "Checking Python syntax..."
python3 -m py_compile backend/api.py 2>/dev/null && echo "  ✓ backend/api.py" || { echo "  ✗ backend/api.py (SYNTAX ERROR)"; ERRORS=$((ERRORS + 1)); }
python3 -m py_compile backend/config.py 2>/dev/null && echo "  ✓ backend/config.py" || { echo "  ✗ backend/config.py (SYNTAX ERROR)"; ERRORS=$((ERRORS + 1)); }

# Summary
echo ""
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo "✓ All checks passed! Production setup is complete."
    exit 0
else
    echo "✗ Found $ERRORS error(s). Please fix before deployment."
    exit 1
fi
