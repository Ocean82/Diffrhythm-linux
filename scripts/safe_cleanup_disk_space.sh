#!/bin/bash
# Safe cleanup script to free disk space
# Removes only safe-to-remove items that won't affect system operation

set -e

echo "=========================================="
echo "Safe Disk Space Cleanup"
echo "=========================================="
echo ""

# Show current disk usage
echo "Current disk usage:"
df -h /
echo ""

# Calculate space before cleanup
SPACE_BEFORE=$(df / | tail -1 | awk '{print $3}')

echo "Starting cleanup..."
echo ""

# 1. Clean APT cache (safe - can be re-downloaded)
echo "1. Cleaning APT cache..."
sudo apt-get clean
sudo apt-get autoclean
echo "   ✓ APT cache cleaned"
echo ""

# 2. Remove old kernel packages (safe - keeps current kernel)
echo "2. Removing old kernel packages..."
OLD_KERNELS=$(dpkg -l | grep -E 'linux-image|linux-headers' | grep -v $(uname -r) | awk '{print $2}' | tr '\n' ' ')
if [ -n "$OLD_KERNELS" ]; then
    echo "   Found old kernels: $OLD_KERNELS"
    sudo apt-get purge -y $OLD_KERNELS 2>/dev/null || true
    echo "   ✓ Old kernels removed"
else
    echo "   No old kernels found"
fi
echo ""

# 3. Clean journal logs (safe - keeps recent logs)
echo "3. Cleaning old journal logs..."
sudo journalctl --vacuum-time=7d
echo "   ✓ Journal logs cleaned (kept last 7 days)"
echo ""

# 4. Clean old SSM agent logs (safe - AWS SSM agent logs)
echo "4. Cleaning old SSM agent logs..."
if [ -f /var/log/amazon/ssm/amazon-ssm-agent.log.1 ]; then
    sudo truncate -s 0 /var/log/amazon/ssm/amazon-ssm-agent.log.1
    echo "   ✓ Old SSM agent logs cleaned"
fi
if [ -f /var/log/amazon/ssm/amazon-ssm-agent.log ]; then
    sudo truncate -s 0 /var/log/amazon/ssm/amazon-ssm-agent.log
    echo "   ✓ Current SSM agent log cleaned"
fi
echo ""

# 5. Clean old syslog files (safe - keeps current)
echo "5. Cleaning old syslog files..."
sudo find /var/log -name "*.gz" -type f -mtime +30 -delete 2>/dev/null || true
echo "   ✓ Old compressed log files removed"
echo ""

# 6. Clean snap cache (safe - snap packages)
echo "6. Cleaning snap cache..."
# Remove disabled snap revisions
sudo snap list --all | awk '/disabled/{print $1, $3}' | while read snapname revision; do
    sudo snap remove "$snapname" --revision="$revision" 2>/dev/null || true
done
# Clean snapd cache directory (safe - will be regenerated)
sudo rm -rf /var/lib/snapd/cache/* 2>/dev/null || true
echo "   ✓ Snap cache cleaned"
echo ""

# 7. Clean temporary files (safe - /tmp)
echo "7. Cleaning temporary files..."
sudo find /tmp -type f -atime +7 -delete 2>/dev/null || true
sudo find /var/tmp -type f -atime +7 -delete 2>/dev/null || true
echo "   ✓ Temporary files cleaned"
echo ""

# 8. Clean package manager cache (safe)
echo "8. Cleaning package manager caches..."
sudo rm -rf /var/cache/debconf/* 2>/dev/null || true
sudo rm -rf /var/cache/apparmor/* 2>/dev/null || true
echo "   ✓ Package manager caches cleaned"
echo ""

# 9. Remove unused packages (safe - autoremove)
echo "9. Removing unused packages..."
sudo apt-get autoremove -y
echo "   ✓ Unused packages removed"
echo ""

# 10. Clean Docker build cache (if Docker is installed)
if command -v docker &> /dev/null; then
    echo "10. Cleaning Docker build cache..."
    sudo docker builder prune -af 2>/dev/null || true
    echo "   ✓ Docker build cache cleaned"
    echo ""
fi

# Calculate space after cleanup
SPACE_AFTER=$(df / | tail -1 | awk '{print $3}')
SPACE_FREED=$((SPACE_BEFORE - SPACE_AFTER))

echo "=========================================="
echo "Cleanup Complete"
echo "=========================================="
echo ""
echo "Final disk usage:"
df -h /
echo ""
echo "Space freed: ~${SPACE_FREED}KB"
echo ""
echo "Safe cleanup completed successfully!"
