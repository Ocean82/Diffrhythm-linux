# SSH Connection Fix - Complete Solution

**Date:** January 24, 2026  
**Instance:** i-0aedf69f3127e24f8 (52.0.207.242)

## Critical Discovery

**ALL connections to the instance are failing:**
- ❌ SSH (port 22): Connection timeout
- ❌ HTTP (port 80): Connection timeout  
- ❌ HTTPS (port 443): Connection timeout
- ✅ General internet: Working (8.8.8.8, google.com work)

**This indicates:** The issue is **NOT** port-specific blocking, but rather:
1. **ISP/Router blocking the specific AWS IP address**
2. **Instance-level firewall (ufw) blocking all inbound traffic**
3. **Network routing issue to this specific destination**

## Investigation Results

### ✅ Confirmed Working
- Instance is running
- SSH service started on instance
- Security group allows your IP (68.251.50.12/32)
- Route table has internet gateway route
- General internet connectivity works

### ❌ Confirmed Failing
- All TCP connections to 52.0.207.242 fail (22, 80, 443)
- SSM status: "ConnectionLost" (needs IAM role)
- No Windows Firewall rules blocking outbound

## Solutions (In Order of Preference)

### Solution 1: Enable AWS Systems Manager (SSM) ⭐ BEST OPTION

**Why:** Works through firewalls, no SSH needed, more secure

**Steps:**

```powershell
# 1. Create IAM role for SSM
aws iam create-role `
  --role-name EC2-SSM-Role `
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ec2.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# 2. Attach SSM managed policy
aws iam attach-role-policy `
  --role-name EC2-SSM-Role `
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

# 3. Create instance profile
aws iam create-instance-profile --instance-profile-name EC2-SSM-Profile
aws iam add-role-to-instance-profile `
  --instance-profile-name EC2-SSM-Profile `
  --role-name EC2-SSM-Role

# 4. Attach to instance
aws ec2 associate-iam-instance-profile `
  --instance-id i-0aedf69f3127e24f8 `
  --iam-instance-profile Name=EC2-SSM-Profile

# 5. Wait 5-10 minutes for SSM to connect, then:
aws ssm start-session --target i-0aedf69f3127e24f8
```

**Time:** 10-15 minutes setup + 5-10 minutes wait

### Solution 2: Check Instance Firewall (ufw)

**If you can access via EC2 Instance Connect or another method:**

```bash
# Check ufw status
sudo ufw status

# If enabled and blocking, allow SSH:
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Or disable ufw temporarily:
sudo ufw disable
```

**Access Methods:**
- AWS Console → EC2 → Instance Connect (browser-based)
- AWS Systems Manager (after setting up IAM role)

### Solution 3: Use EC2 Instance Connect

**AWS provides browser-based SSH:**

1. Go to AWS Console → EC2 → Instances
2. Select instance `i-0aedf69f3127e24f8`
3. Click "Connect" → "EC2 Instance Connect"
4. This opens a browser-based terminal
5. Check and fix ufw if needed

### Solution 4: Check Router/ISP Settings

**Router Configuration:**
1. Log into router (192.168.1.254)
2. Check firewall/security settings
3. Look for:
   - IP address blocking
   - Geo-blocking (blocking AWS regions)
   - Outbound connection restrictions

**ISP Contact:**
- Ask if they block AWS IP ranges
- Request unblocking (may require business account)
- Check if they have "security" features blocking cloud services

### Solution 5: Use Mobile Hotspot / VPN

**Quick Workaround:**
1. Connect to mobile hotspot
2. Get new public IP
3. Add new IP to security group:
   ```powershell
   $NEW_IP = (Invoke-WebRequest -Uri https://api.ipify.org -UseBasicParsing).Content
   aws ec2 authorize-security-group-ingress `
     --group-id sg-0381e5cf859d3feb4 `
     --ip-permissions IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges="[{CidrIp=$NEW_IP/32,Description=Mobile-Hotspot}]"
   ```
4. Test connection from new network

### Solution 6: Deploy via S3 + SSM Run Command

**After enabling SSM (Solution 1):**

```powershell
# 1. Create S3 bucket
$BUCKET = "diffrhythm-deploy-$(Get-Date -Format yyyyMMddHHmmss)"
aws s3 mb s3://$BUCKET

# 2. Upload files
cd d:\EMBERS-BANK\DiffRhythm-LINUX
aws s3 cp backend/ s3://$BUCKET/backend/ --recursive
aws s3 cp test_server_implementation.py s3://$BUCKET/
aws s3 cp test_payment_flow.py s3://$BUCKET/

# 3. Deploy to instance via SSM
aws ssm send-command `
  --instance-ids i-0aedf69f3127e24f8 `
  --document-name "AWS-RunShellScript" `
  --parameters "commands=[
    'cd /home/ubuntu/app',
    'aws s3 sync s3://$BUCKET/backend/ backend/',
    'aws s3 cp s3://$BUCKET/test_server_implementation.py .',
    'aws s3 cp s3://$BUCKET/test_payment_flow.py .',
    'sudo systemctl restart burntbeats-api'
  ]"
```

## Immediate Action Plan

### Step 1: Try EC2 Instance Connect (5 minutes)
1. AWS Console → EC2 → Instances
2. Select instance → Connect → EC2 Instance Connect
3. Check `sudo ufw status`
4. If blocking, run: `sudo ufw allow 22/tcp && sudo ufw allow 80/tcp && sudo ufw allow 443/tcp`

### Step 2: Enable SSM (15 minutes)
Follow Solution 1 above to set up SSM access

### Step 3: If Still Failing
- Try mobile hotspot (Solution 5)
- Contact ISP about AWS IP blocking
- Check router settings

## Diagnostic Commands

### Check Instance Firewall (via EC2 Instance Connect)
```bash
sudo ufw status verbose
sudo iptables -L -n -v
sudo systemctl status ssh
```

### Check Network Connectivity (from instance)
```bash
# From inside instance (via EC2 Instance Connect or SSM)
curl -I http://google.com
netstat -tuln | grep 22
ss -tuln | grep 22
```

### Test Security Group (from AWS)
```powershell
# Verify your IP is in security group
aws ec2 describe-security-groups --group-ids sg-0381e5cf859d3feb4 --query "SecurityGroups[0].IpPermissions[?FromPort==\`22\`]" --output json
```

## Most Likely Cause

Based on investigation:
1. **Instance firewall (ufw)** - Most likely if SSH service is running but connections fail
2. **ISP blocking AWS IPs** - Possible if router/ISP has security features
3. **Router blocking specific IP** - Less likely but possible

## Recommended Next Steps

1. **Try EC2 Instance Connect** (immediate, no setup needed)
2. **Enable SSM** (best long-term solution)
3. **Check ufw** (via EC2 Instance Connect)
4. **If still failing, try mobile hotspot** (quick workaround)

---

**Status:** All connections to instance failing (not just SSH)  
**Best Solution:** Enable SSM or use EC2 Instance Connect to check/fix ufw  
**Time to Fix:** 5-15 minutes depending on solution chosen
