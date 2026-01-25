# SSH Connection Investigation Report

**Date:** January 24, 2026  
**Instance:** i-0aedf69f3127e24f8 (52.0.207.242)

## Investigation Summary

### ✅ What's Confirmed Working

1. **Instance Status:** ✅ Running
2. **SSH Service:** ✅ Started and running (confirmed via console output)
3. **Security Group:** ✅ Allows SSH from `68.251.50.12/32` (your current public IP)
4. **SSH Key File:** ✅ Exists at `C:\Users\sammy\.ssh\server_saver_key`
5. **SSM Agent:** ✅ Running (Amazon SSM Agent v3.3.3050.0)

### ❌ What's Failing

1. **TCP Connection to Port 22:** ❌ Connection timeout
2. **Ping:** ❌ Timeout
3. **HTTP/HTTPS:** ⚠️ Need to test (ports 80/443 should work if SSH is only issue)

### 🔍 Key Findings

#### Network Connectivity Test Results
```
Source Address: 192.168.1.105 (local network)
Public IP: 68.251.50.12 (matches security group rule)
Target: 52.0.207.242:22
Result: Connection timeout
Ping: Failed
```

#### Instance Console Output
- SSH service started successfully: `[OK] Started OpenBSD Secure Shell server`
- SSM Agent running: `Amazon SSM Agent v3.3.3050.0 is running`
- ⚠️ **OOM (Out of Memory) kills detected** - Python processes being killed due to memory pressure

#### Security Group Rules (Port 22)
- `172.29.128.1/32` - WSL SSH (rule: sgr-0b51735f0e0ce07d7)
- `68.251.50.12/32` - SSH (rule: sgr-08b81c351f8a1b677) ✅ **YOUR IP IS ALLOWED**

#### Instance Configuration
- **Key Name:** `Burnt-Beats-KEY` (you're using `server_saver_key` - verify these match)
- **Instance Type:** t3.large
- **VPC:** vpc-0168169d9b22fb66f
- **Subnet:** subnet-0d2222243800861ee

## Root Cause Analysis

### Primary Issue: Network-Level Blocking

The connection timeout occurs **before** reaching AWS infrastructure, indicating:

1. **Local Network/Router Blocking:**
   - Your router (192.168.1.254) may be blocking outbound port 22
   - ISP may be blocking outbound SSH connections
   - Corporate firewall (if applicable)

2. **Windows Firewall:**
   - Outbound rules may be blocking port 22
   - Need to check Windows Firewall rules

3. **ISP-Level Blocking:**
   - Some ISPs block outbound SSH (port 22) for security
   - Common in residential networks

### Secondary Issue: Key Name Mismatch

- **Instance expects:** `Burnt-Beats-KEY`
- **You're using:** `server_saver_key`

**Action:** Verify these are the same key file (just different names), or use the correct key.

## Solutions

### Solution 1: Use AWS Systems Manager (SSM) ⭐ RECOMMENDED

**SSM Agent is already running!** You can connect without SSH:

```powershell
# Check if instance has IAM role for SSM
aws ec2 describe-instances --instance-ids i-0aedf69f3127e24f8 --query "Reservations[0].Instances[0].IamInstanceProfile"

# If no IAM role, create one (one-time setup):
# 1. Create IAM role with SSM permissions
# 2. Attach to instance
# 3. Wait 5-10 minutes for SSM to connect

# Then connect:
aws ssm start-session --target i-0aedf69f3127e24f8
```

**Advantages:**
- No SSH needed
- Works through firewalls
- More secure
- Already have SSM Agent running

### Solution 2: Check Windows Firewall

```powershell
# Check outbound rules blocking port 22
Get-NetFirewallRule | Where-Object { 
    $_.Direction -eq "Outbound" -and 
    ($_.DisplayName -like "*SSH*" -or $_.DisplayName -like "*22*")
} | Format-Table DisplayName,Action,Enabled

# If blocking, create allow rule:
New-NetFirewallRule -DisplayName "Allow SSH Outbound" -Direction Outbound -Protocol TCP -LocalPort 22 -Action Allow
```

### Solution 3: Router/Network Configuration

1. **Check Router Settings:**
   - Log into router (192.168.1.254)
   - Check firewall/security settings
   - Look for port blocking or outbound restrictions

2. **Contact ISP:**
   - Ask if they block outbound port 22
   - Request port 22 to be unblocked (may require business account)

### Solution 4: Use Different Port (Requires Server Access)

If you can access via SSM or another method:
1. Change SSH port on server to 2222 or 443
2. Update security group to allow new port
3. Connect via new port

### Solution 5: Mobile Hotspot / Different Network

1. Connect to mobile hotspot
2. Get new public IP
3. Add new IP to security group
4. Test SSH connection

## Immediate Action Plan

### Step 1: Try SSM (Easiest)

```powershell
# Check if instance has IAM role
aws ec2 describe-instances --instance-ids i-0aedf69f3127e24f8 --query "Reservations[0].Instances[0].IamInstanceProfile"

# If it returns null, we need to set up IAM role
# If it returns a profile, try:
aws ssm start-session --target i-0aedf69f3127e24f8
```

### Step 2: Check Windows Firewall

```powershell
# List outbound rules
Get-NetFirewallRule -Direction Outbound | Where-Object { $_.DisplayName -like "*Block*" } | Format-Table
```

### Step 3: Test HTTP/HTTPS Connectivity

```powershell
# If HTTP/HTTPS work but SSH doesn't, it confirms port 22 blocking
Test-NetConnection -ComputerName 52.0.207.242 -Port 80
Test-NetConnection -ComputerName 52.0.207.242 -Port 443
```

### Step 4: Verify SSH Key

```powershell
# Check if server_saver_key matches Burnt-Beats-KEY
# Compare key fingerprints or try both keys
ssh-keygen -lf "C:\Users\sammy\.ssh\server_saver_key"
```

## Additional Notes

### OOM (Out of Memory) Warnings

Console output shows Python processes being killed due to memory pressure:
```
Out of memory: Killed process 1058 (python3) total-vm:14644508kB
```

**This is separate from SSH issue but should be addressed:**
- Instance type: t3.large (8GB RAM)
- May need to optimize memory usage or upgrade instance

### Network ACL Check

Need to verify Network ACLs aren't blocking (though security groups should override):
- Default VPC typically has permissive ACLs
- Custom VPC may have restrictive ACLs

## Conclusion

**Primary Issue:** Network-level blocking of outbound port 22 (router/ISP/Windows Firewall)

**Best Solution:** Use AWS Systems Manager (SSM) since agent is already running

**Next Steps:**
1. Check if instance has IAM role for SSM
2. If yes, try `aws ssm start-session`
3. If no, set up IAM role (10-15 minutes)
4. Alternatively, check Windows Firewall and router settings

---

**Status:** SSH blocked at network level, but SSM available as alternative  
**Recommendation:** Use SSM for immediate access, then investigate network blocking
