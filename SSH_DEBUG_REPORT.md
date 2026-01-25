# SSH Connection Debug Report

**Date:** January 24, 2026  
**Issue:** SSH connection timing out to 52.0.207.242

## AWS Instance Status

### Instance Details
- **Instance ID:** `i-0aedf69f3127e24f8`
- **Status:** ✅ Running
- **Public IP:** `52.0.207.242`
- **Private IP:** `172.31.90.134`
- **Instance Type:** `t3.large`
- **Key Name:** `Burnt-Beats-KEY`
- **Security Group:** `sg-0381e5cf859d3feb4`

## Root Cause: Security Group Restriction

The security group **only allows SSH (port 22) from specific IP addresses:**

### Current SSH Rules (Port 22)
1. `172.29.128.1/32` - WSL SSH access
2. `68.251.50.12/32` - SSH access

**Your current IP address is NOT in this list**, which is why SSH connections are timing out.

## Solution Options

### Option 1: Add Your Current IP to Security Group (Recommended)

```powershell
# Get your current IP
$MY_IP = (Invoke-WebRequest -Uri https://api.ipify.org -UseBasicParsing).Content
Write-Host "Your IP: $MY_IP"

# Add your IP to security group
aws ec2 authorize-security-group-ingress `
  --group-id sg-0381e5cf859d3feb4 `
  --protocol tcp `
  --port 22 `
  --cidr "$MY_IP/32" `
  --description "SSH access for deployment"
```

### Option 2: Temporarily Allow All IPs (Less Secure)

```powershell
aws ec2 authorize-security-group-ingress `
  --group-id sg-0381e5cf859d3feb4 `
  --protocol tcp `
  --port 22 `
  --cidr "0.0.0.0/0" `
  --description "Temporary SSH access - REMOVE AFTER DEPLOYMENT"
```

**⚠️ Warning:** This allows SSH from anywhere. **Remove this rule after deployment!**

### Option 3: Use Allowed IP Address

If you have access to one of the allowed IPs:
- `172.29.128.1` (WSL)
- `68.251.50.12`

You can SSH from that location.

## Current Security Group Rules

### Port 22 (SSH)
- ✅ `172.29.128.1/32` - WSL ssh
- ✅ `68.251.50.12/32` - ssh

### Port 80 (HTTP)
- ✅ `0.0.0.0/0` - All IPs (public web access)

### Port 443 (HTTPS)
- ✅ `0.0.0.0/0` - All IPs (public web access)

### Port 8000 (Admin)
- ✅ `68.251.50.12/32` - Admin port (restricted)

## Verification Commands

After adding your IP, verify:

```powershell
# Check your IP was added
aws ec2 describe-security-groups `
  --group-ids sg-0381e5cf859d3feb4 `
  --query "SecurityGroups[0].IpPermissions[?FromPort==\`22\`].IpRanges" `
  --output json

# Test SSH connection
ssh -i "C:\Users\sammy\.ssh\server_saver_key" ubuntu@52.0.207.242
```

## Cleanup (After Deployment)

**Important:** Remove temporary SSH access rules after deployment:

```powershell
# List current rules to find the one to remove
aws ec2 describe-security-groups `
  --group-ids sg-0381e5cf859d3feb4 `
  --query "SecurityGroups[0].IpPermissions[?FromPort==\`22\`]" `
  --output json

# Remove specific IP (replace YOUR_IP with actual IP)
aws ec2 revoke-security-group-ingress `
  --group-id sg-0381e5cf859d3feb4 `
  --protocol tcp `
  --port 22 `
  --cidr "YOUR_IP/32"
```

## Next Steps

1. **Add your IP to security group** (use Option 1 above)
2. **Test SSH connection**
3. **Deploy code** once SSH works
4. **Remove temporary SSH access** after deployment (if added)

---

**Status:** ⚠️ Security group restriction identified  
**Action Required:** Add your IP to security group to enable SSH access
