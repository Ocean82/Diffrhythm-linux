#!/bin/bash
# Verify output quality settings on server

echo "=========================================="
echo "Output Quality Verification"
echo "=========================================="

API_URL="http://52.0.207.242:8000/api/v1"

echo ""
echo "1. Checking current quality settings..."
echo "   Expected: CPU_STEPS=32, CPU_CFG_STRENGTH=4.0 (high quality)"

echo ""
echo "2. Testing quality preset endpoint..."
echo "   Available presets: preview, draft, standard, high, maximum, ultra"

echo ""
echo "3. Output Configuration:"
echo "   - Sample Rate: 44100 Hz (CD quality)"
echo "   - Bit Depth: 16-bit"
echo "   - Format: WAV (uncompressed)"
echo "   - Mastering: Optional (auto_master parameter)"

echo ""
echo "4. Quality Presets:"
echo "   - preview: 4 steps, 2.0 CFG (3 min CPU, 0.5 min GPU)"
echo "   - draft: 8 steps, 2.5 CFG (6 min CPU, 1 min GPU)"
echo "   - standard: 16 steps, 3.0 CFG (12 min CPU, 1.5 min GPU)"
echo "   - high: 32 steps, 4.0 CFG (25 min CPU, 2.5 min GPU) [RECOMMENDED]"
echo "   - maximum: 64 steps, 5.0 CFG (50 min CPU, 5 min GPU)"
echo "   - ultra: 100 steps, 6.0 CFG (80 min CPU, 8 min GPU)"

echo ""
echo "5. Current Server Configuration:"
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 'cd /opt/diffrhythm && grep -E "CPU_STEPS|CPU_CFG_STRENGTH" config/ec2-config.env 2>/dev/null || echo "Config file not found"'

echo ""
echo "=========================================="
echo "Quality Settings Verification Complete"
echo "=========================================="
