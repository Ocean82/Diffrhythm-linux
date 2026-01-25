# SSH Troubleshooting Report

**Date:** January 24, 2026  
**Instance:** i-0aedf69f3127e24f8 (52.0.207.242)

## Findings

### ✅ Instance Status
- **State:** Running
- **Public IP:** 52.0.207.242
- **Key Name:** Burnt-Beats-KEY
- **Your IP:** 68.251.50.12 ✅ (Already allowed in security group)

### ✅ Security Group
- Port 22 (SSH) is allowed from your IP: `68.251.50.12/32`
- Security Group: `sg-0381e5cf859d3feb4`

### ⚠️ Issue
SSH connection still timing out despite security group allowing your IP.

## Possible Causes

### 1. Key Name Mismatch
- **Instance expects:** `Burnt-Beats-KEY`
- **You're using:** `server_saver_key`

**Solution:** Verify the key file matches the instance key pair, or use the correct key:
```powershell
# Check if you have the Burnt-Beats-KEY
Test-Path "C:\Users\sammy\.ssh\Burnt-Beats-KEY"
```

### 2. Windows Firewall / Network Restrictions
Your local network or Windows Firewall might be blocking outbound SSH (port 22).

**Test:**
```powershell
# Test if port 22 is reachable
Test-NetConnection -ComputerName 52.0.207.242 -Port 22
```

### 3. Instance-Level Firewall (ufw)
The instance might have ufw (Uncomplicated Firewall) blocking SSH.

**Check on server (if you can access via AWS Systems Manager):**
```bash
sudo ufw status
```

### 4. Network ACL Restrictions
VPC-level Network ACLs might be blocking traffic.

### 5. ISP/Router Blocking
Some ISPs block outbound SSH connections.

## Solutions

### Solution 1: Use AWS Systems Manager Session Manager (No SSH Needed)

If SSH is blocked, use AWS Systems Manager:

```powershell
# Start a session
aws ssm start-session --target i-0aedf69f3127e24f8
```

**Prerequisites:**
- Instance must have SSM Agent installed
- IAM role with SSM permissions

### Solution 2: Verify Key File

```powershell
# Check key file permissions (should be readable only by owner)
icacls "C:\Users\sammy\.ssh\server_saver_key"

# Try with explicit key
ssh -i "C:\Users\sammy\.ssh\server_saver_key" -v ubuntu@52.0.207.242
```

### Solution 3: Test from Different Network

Try SSH from:
- Different network (mobile hotspot)
- WSL (if available): `172.29.128.1` is already allowed
- AWS CloudShell

### Solution 4: Check Instance Logs

```powershell
# Get instance console output (may show SSH service status)
aws ec2 get-console-output --instance-id i-0aedf69f3127e24f8 --output text | Select-Object -Last 50
```

### Solution 5: Use Port Forwarding via AWS Systems Manager

```powershell
# Set up port forwarding
aws ssm start-session --target i-0aedf69f3127e24f8 --document-name AWS-StartPortForwardingSession --parameters '{"portNumber":["22"],"localPortNumber":["2222"]}'

# Then SSH to localhost
ssh -i "C:\Users\sammy\.ssh\server_saver_key" -p 2222 ubuntu@localhost
```

## Quick Diagnostic Commands

```powershell
# 1. Test network connectivity
Test-NetConnection -ComputerName 52.0.207.242 -Port 22

# 2. Check security group rules
aws ec2 describe-security-groups --group-ids sg-0381e5cf859d3feb4 --query "SecurityGroups[0].IpPermissions[?FromPort==\`22\`]" --output json

# 3. Verify instance is running
aws ec2 describe-instances --instance-ids i-0aedf69f3127e24f8 --query "Reservations[0].Instances[0].State.Name"

# 4. Check if SSM is available
aws ssm describe-instance-information --filters "Key=InstanceIds,Values=i-0aedf69f3127e24f8"
```

## Recommended Next Steps

1. **Test network connectivity:**
   ```powershell
   Test-NetConnection -ComputerName 52.0.207.242 -Port 22
   ```

2. **Try AWS Systems Manager** (if available):
   ```powershell
   aws ssm start-session --target i-0aedf69f3127e24f8
   ```

3. **Verify key file:**
   - Ensure `server_saver_key` matches `Burnt-Beats-KEY` or is the correct key
   - Check file permissions

4. **Try from WSL** (if `172.29.128.1` is your WSL IP):
   ```bash
   # In WSL
   ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242
   ```

5. **Check Windows Firewall:**
   ```powershell
   Get-NetFirewallRule | Where-Object {$_.DisplayName -like "*SSH*" -or $_.DisplayName -like "*22*"}
   ```

---

**Status:** Security group allows your IP, but connection still timing out  
**Likely Causes:** Network-level blocking, key mismatch, or instance firewall
