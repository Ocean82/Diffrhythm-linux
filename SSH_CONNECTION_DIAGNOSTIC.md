# SSH Connection Diagnostic Report

**Date:** January 24, 2026  
**Server:** `ubuntu@52.0.207.242`  
**Status:** ⚠️ **CONNECTION TIMEOUT DURING SSH HANDSHAKE**

## Diagnostic Results

### ✅ What's Working

1. **Network Connectivity**
   - ✅ Port 22 is open and accepting connections
   - ✅ TCP connection to `52.0.207.242:22` succeeds
   - ✅ Server resolves to `ec2-52-0-207-242.compute-1.amazonaws.com`

2. **SSH Key**
   - ✅ SSH key exists at `C:\Users\sammy\.ssh\server_saver_key`
   - ✅ Key is accessible via WSL path `/mnt/c/Users/sammy/.ssh/server_saver_key`
   - ✅ Deploy script path conversion works (lowercase conversion is fine - WSL is case-insensitive)

3. **Deploy Script**
   - ✅ Path conversion logic is correct
   - ✅ Key copying and permission setting works

### ❌ The Problem

**Connection Timeout During Banner Exchange**

```
Connection established.
Connection timed out during banner exchange
Connection to 52.0.207.242 port 22 timed out
```

This means:
- ✅ TCP connection succeeds (port 22 is open)
- ❌ SSH handshake fails/times out before completing
- The server accepts the connection but doesn't complete the SSH protocol negotiation

## Possible Causes

### 1. **EC2 Security Group / Firewall**
   - Security group may allow port 22 but have restrictive rules
   - Network ACL might be blocking SSH traffic
   - **Check:** AWS Console → EC2 → Security Groups → Inbound rules for port 22

### 2. **SSH Daemon Not Running**
   - SSH service (`sshd`) may be stopped or crashed
   - **Check:** Need AWS Systems Manager or console access to verify

### 3. **Server Overloaded/Unresponsive**
   - EC2 instance may be overloaded or frozen
   - CPU/memory exhaustion causing SSH to hang
   - **Check:** AWS Console → EC2 → Instance status and CloudWatch metrics

### 4. **SSH Configuration Issues**
   - `sshd_config` may have restrictive settings
   - MaxStartups or other limits may be blocking connections
   - **Check:** Need server access to review `/etc/ssh/sshd_config`

### 5. **Network Issues**
   - Intermittent network problems between client and server
   - AWS region/availability zone issues
   - **Check:** Try from different network or use AWS Systems Manager Session Manager

## Recommended Actions

### Immediate Steps

1. **Check EC2 Instance Status**
   ```bash
   # Via AWS CLI (if configured)
   aws ec2 describe-instance-status --instance-ids <instance-id>
   ```

2. **Verify Security Group Rules**
   - AWS Console → EC2 → Security Groups
   - Ensure inbound rule for port 22 allows your IP (or 0.0.0.0/0 for testing)
   - Check outbound rules

3. **Use AWS Systems Manager Session Manager**
   - If enabled, you can access the instance without SSH
   - AWS Console → EC2 → Connect → Session Manager

4. **Check CloudWatch Logs**
   - Look for system logs or SSH-related errors
   - Check instance metrics (CPU, memory, network)

### Alternative Access Methods

1. **AWS Systems Manager Session Manager**
   ```bash
   aws ssm start-session --target <instance-id>
   ```

2. **EC2 Instance Connect**
   - AWS Console → EC2 → Connect → EC2 Instance Connect
   - Browser-based terminal access

3. **AWS Console Serial Console** (if enabled)
   - Direct console access without network

### Fixing the Deploy Script

The deploy script is actually working correctly. The issue is server-side. However, we can improve error handling:

**Current behavior:** Script fails silently with "Cannot connect to server"

**Suggested improvement:** Add more detailed error messages and timeout handling.

## Test Results Summary

```
✓ Port 22: Open (TCP connection succeeds)
✓ SSH Key: Found and accessible
✓ Path Conversion: Working correctly
✗ SSH Handshake: Times out during banner exchange
✗ Ping: Fails (ICMP may be blocked, but this is normal for EC2)
```

## Next Steps

1. **Verify EC2 instance is running** (not stopped/terminated)
2. **Check security group** allows SSH from your IP
3. **Review CloudWatch metrics** for instance health
4. **Try AWS Systems Manager** if SSH continues to fail
5. **Consider restarting the instance** if it appears frozen

## Files Created

- `scripts/test-ssh-connection.sh` - Diagnostic script for future troubleshooting

---

## Update: SSH Key Issue Identified

**Root Cause Found:** The deploy script was using the wrong SSH key.

- ❌ Script was looking for: `server_saver_key`
- ✅ Instance requires: `Burnt-Beats-KEY.pem`
- ❌ `Burnt-Beats-KEY.pem` not found in local directories

**Solution:** Download `Burnt-Beats-KEY.pem` from AWS Console and place it in:
- `C:\Users\sammy\OneDrive\Desktop\AWS ITEMS\Burnt-Beats-KEY.pem` (preferred)
- Or set `SSH_KEY` environment variable

See **SSH_KEY_SETUP_INSTRUCTIONS.md** for detailed steps.

---

**Conclusion:** The deployment script has been updated to use the correct key. Once `Burnt-Beats-KEY.pem` is downloaded and placed in the correct location, deployment should work.
