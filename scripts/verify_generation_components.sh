#!/bin/bash
# Comprehensive verification of all song generation components on server

set -e

echo "=========================================="
echo "Song Generation Components Verification"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((ERRORS++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

# 1. Check Python environment
echo "1. Python Environment"
echo "-------------------"
if python3 --version &>/dev/null; then
    PYTHON_VERSION=$(python3 --version)
    check_pass "Python installed: $PYTHON_VERSION"
else
    check_fail "Python3 not found"
fi

# 2. Check required Python packages
echo ""
echo "2. Required Python Packages"
echo "----------------------------"
REQUIRED_PACKAGES=(
    "torch"
    "torchaudio"
    "fastapi"
    "uvicorn"
    "pydantic"
    "numpy"
    "scipy"
    "librosa"
    "transformers"
    "einops"
)

for package in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        VERSION=$(python3 -c "import $package; print(getattr($package, '__version__', 'unknown'))" 2>/dev/null || echo "installed")
        check_pass "$package: $VERSION"
    else
        check_fail "$package: NOT INSTALLED"
    fi
done

# 3. Check project structure
echo ""
echo "3. Project Structure"
echo "-------------------"
REQUIRED_DIRS=(
    "backend"
    "infer"
    "model"
    "post_processing"
    "output"
    "pretrained"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        check_pass "Directory exists: $dir"
    else
        check_fail "Directory missing: $dir"
    fi
done

# 4. Check critical files
echo ""
echo "4. Critical Files"
echo "---------------"
REQUIRED_FILES=(
    "backend/api.py"
    "backend/config.py"
    "backend/exceptions.py"
    "backend/logging_config.py"
    "backend/metrics.py"
    "backend/security.py"
    "infer/infer.py"
    "infer/infer_utils.py"
    "model/cfm.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        SIZE=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
        check_pass "File exists: $file ($SIZE bytes)"
    else
        check_fail "File missing: $file"
    fi
done

# 5. Check imports
echo ""
echo "5. Module Imports"
echo "---------------"
python3 << 'PYEOF'
import sys
import os
sys.path.insert(0, os.getcwd())

errors = []
warnings = []

# Test backend imports
try:
    from backend.config import Config
    print("✓ backend.config imported")
except Exception as e:
    print(f"✗ backend.config import failed: {e}")
    errors.append("backend.config")

try:
    from backend.api import app
    print("✓ backend.api imported")
except Exception as e:
    print(f"✗ backend.api import failed: {e}")
    errors.append("backend.api")

try:
    from infer.infer_utils import prepare_model, get_lrc_token, get_style_prompt
    print("✓ infer.infer_utils imported")
except Exception as e:
    print(f"✗ infer.infer_utils import failed: {e}")
    errors.append("infer.infer_utils")

try:
    from infer.infer import inference, save_audio_robust
    print("✓ infer.infer imported")
except Exception as e:
    print(f"✗ infer.infer import failed: {e}")
    errors.append("infer.infer")

try:
    from model.cfm import CFM
    print("✓ model.cfm imported")
except Exception as e:
    print(f"✗ model.cfm import failed: {e}")
    errors.append("model.cfm")

# Test optional imports
try:
    from post_processing.mastering import master_audio_file
    print("✓ post_processing.mastering imported (optional)")
except Exception as e:
    print(f"⚠ post_processing.mastering not available: {e}")
    warnings.append("post_processing.mastering")

if errors:
    sys.exit(1)
PYEOF

IMPORT_STATUS=$?
if [ $IMPORT_STATUS -eq 0 ]; then
    check_pass "All critical imports successful"
else
    check_fail "Some imports failed"
fi

# 6. Check configuration
echo ""
echo "6. Configuration"
echo "---------------"
if [ -f "config/ec2-config.env" ]; then
    check_pass "Config file exists: config/ec2-config.env"
    
    # Check key settings
    if grep -q "CPU_STEPS=32" config/ec2-config.env 2>/dev/null; then
        check_pass "CPU_STEPS set to 32 (high quality)"
    else
        CPU_STEPS=$(grep "CPU_STEPS=" config/ec2-config.env 2>/dev/null | cut -d= -f2 || echo "not found")
        check_warn "CPU_STEPS is $CPU_STEPS (expected 32 for high quality)"
    fi
    
    if grep -q "CPU_CFG_STRENGTH=4.0" config/ec2-config.env 2>/dev/null; then
        check_pass "CPU_CFG_STRENGTH set to 4.0 (high quality)"
    else
        CFG=$(grep "CPU_CFG_STRENGTH=" config/ec2-config.env 2>/dev/null | cut -d= -f2 || echo "not found")
        check_warn "CPU_CFG_STRENGTH is $CFG (expected 4.0 for high quality)"
    fi
else
    check_fail "Config file missing: config/ec2-config.env"
fi

# 7. Check output directory
echo ""
echo "7. Output Directory"
echo "-----------------"
if [ -d "output" ]; then
    check_pass "Output directory exists"
    
    if [ -w "output" ]; then
        check_pass "Output directory is writable"
    else
        check_fail "Output directory is not writable"
    fi
    
    # Check disk space
    AVAILABLE=$(df -BG output 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//' || echo "0")
    if [ "$AVAILABLE" -gt 5 ]; then
        check_pass "Sufficient disk space: ${AVAILABLE}GB available"
    else
        check_warn "Low disk space: ${AVAILABLE}GB available"
    fi
else
    check_fail "Output directory missing"
fi

# 8. Check pretrained models directory
echo ""
echo "8. Pretrained Models"
echo "------------------"
if [ -d "pretrained" ]; then
    check_pass "Pretrained directory exists"
    
    MODEL_COUNT=$(find pretrained -type f -name "*.bin" -o -name "*.safetensors" -o -name "*.pt" -o -name "*.pth" 2>/dev/null | wc -l)
    if [ "$MODEL_COUNT" -gt 0 ]; then
        check_pass "Found $MODEL_COUNT model files"
    else
        check_warn "No model files found (models will be downloaded on first use)"
    fi
else
    check_warn "Pretrained directory missing (will be created on first use)"
fi

# 9. Check API configuration
echo ""
echo "9. API Configuration"
echo "-------------------"
python3 << 'PYEOF'
import sys
import os
sys.path.insert(0, os.getcwd())

try:
    from backend.config import Config
    
    print(f"✓ HOST: {Config.HOST}")
    print(f"✓ PORT: {Config.PORT}")
    print(f"✓ API_PREFIX: {Config.API_PREFIX}")
    print(f"✓ DEVICE: {Config.DEVICE}")
    print(f"✓ CPU_STEPS: {Config.CPU_STEPS}")
    print(f"✓ CPU_CFG_STRENGTH: {Config.CPU_CFG_STRENGTH}")
    print(f"✓ OUTPUT_DIR: {Config.OUTPUT_DIR}")
    
    # Validate config
    errors = Config.validate()
    if errors:
        print(f"✗ Configuration errors: {errors}")
        sys.exit(1)
    else:
        print("✓ Configuration validation passed")
        
except Exception as e:
    print(f"✗ Configuration check failed: {e}")
    sys.exit(1)
PYEOF

CONFIG_STATUS=$?
if [ $CONFIG_STATUS -eq 0 ]; then
    check_pass "API configuration valid"
else
    check_fail "API configuration has errors"
fi

# 10. Check audio processing dependencies
echo ""
echo "10. Audio Processing"
echo "-------------------"
AUDIO_LIBS=("torchaudio" "soundfile" "scipy.io.wavfile" "librosa")

for lib in "${AUDIO_LIBS[@]}"; do
    if python3 -c "import $lib" 2>/dev/null; then
        check_pass "$lib available"
    else
        check_warn "$lib not available (may use fallback)"
    fi
done

# 11. Check quality presets
echo ""
echo "11. Quality Presets"
echo "------------------"
if [ -f "infer/quality_presets.py" ]; then
    check_pass "Quality presets file exists"
    
    python3 << 'PYEOF'
import sys
import os
sys.path.insert(0, os.getcwd())

try:
    from infer.quality_presets import get_preset, QUALITY_PRESETS
    
    print(f"✓ Available presets: {', '.join(QUALITY_PRESETS.keys())}")
    
    # Test each preset
    for preset_name in QUALITY_PRESETS.keys():
        preset = get_preset(preset_name)
        print(f"  - {preset_name}: {preset.steps} steps, {preset.cfg_strength} CFG")
    
except Exception as e:
    print(f"✗ Quality presets check failed: {e}")
    sys.exit(1)
PYEOF
    
    PRESET_STATUS=$?
    if [ $PRESET_STATUS -eq 0 ]; then
        check_pass "Quality presets functional"
    else
        check_fail "Quality presets have errors"
    fi
else
    check_warn "Quality presets file not found"
fi

# 12. Check Docker/Service status
echo ""
echo "12. Service Status"
echo "-----------------"
if command -v docker &>/dev/null; then
    check_pass "Docker installed"
    
    if docker ps | grep -q diffrhythm; then
        check_pass "DiffRhythm container is running"
    else
        check_warn "DiffRhythm container not running"
    fi
else
    check_warn "Docker not found (service may run directly)"
fi

# Summary
echo ""
echo "=========================================="
echo "Verification Summary"
echo "=========================================="
echo -e "Errors: ${RED}$ERRORS${NC}"
echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All critical components verified${NC}"
    exit 0
else
    echo -e "${RED}✗ Some components have errors${NC}"
    exit 1
fi
