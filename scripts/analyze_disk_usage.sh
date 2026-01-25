#!/bin/bash
# Analyze disk usage and identify safe-to-remove items
# Run on the server to identify space-consuming items

set -e

echo "=========================================="
echo "Disk Usage Analysis"
echo "=========================================="
echo ""

# Overall disk usage
echo "1. Overall Disk Usage:"
df -h /
echo ""

# Docker usage
echo "2. Docker Disk Usage:"
sudo docker system df
echo ""

# Journal logs
echo "3. System Journal Logs:"
journalctl --disk-usage
echo ""

# Large files in /var
echo "4. Large Directories in /var:"
sudo du -sh /var/* 2>/dev/null | sort -hr | head -10
echo ""

# Docker detailed usage
echo "5. Docker Detailed Usage:"
sudo du -sh /var/lib/docker/* 2>/dev/null | sort -hr | head -10
echo ""

# Large log files
echo "6. Large Log Files (>10MB):"
sudo find /var/log -type f -size +10M -exec ls -lh {} \; 2>/dev/null | awk '{print $5, $9}' | head -10
echo ""

# Cache directories
echo "7. Cache Directories:"
sudo du -sh /var/cache/* 2>/dev/null | sort -hr | head -10
echo ""

# Temporary files
echo "8. Large Temporary Files (>50MB):"
find /tmp -type f -size +50M 2>/dev/null | head -10
echo ""

# Project directory large files
echo "9. Large Files in /opt/diffrhythm (>10MB):"
sudo find /opt/diffrhythm -type f -size +10M 2>/dev/null | head -20
echo ""

# APT cache
echo "10. APT Cache Size:"
sudo du -sh /var/cache/apt 2>/dev/null
echo ""

# Python cache
echo "11. Python Cache (__pycache__):"
find /opt/diffrhythm -type d -name __pycache__ -exec du -sh {} \; 2>/dev/null | head -10
echo ""

# Old kernels
echo "12. Old Kernel Packages:"
dpkg -l | grep -E 'linux-image|linux-headers' | grep -v $(uname -r) | wc -l
echo ""

echo "=========================================="
echo "Analysis Complete"
echo "=========================================="
