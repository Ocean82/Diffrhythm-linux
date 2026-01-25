# Deployment Solution - SSH Blocked

**Date:** January 24, 2026  
**Issue:** SSH connection timing out due to network-level blocking

## Diagnosis Summary

### ✅ What's Working
- Instance is **running** (i-0aedf69f3127e24f8)
- Security group **allows your IP** (68.251.50.12/32)
- Public IP is correct (52.0.207.242)

### ❌ What's Blocked
- **Port 22 is blocked** from your network (ISP/router likely blocking outbound SSH)
- **SSM not available** (instance has no IAM role for Systems Manager)
- **Ping also fails** (network-level blocking)

## Recommended Solutions

### Solution 1: Use WSL (If Available) ⭐ RECOMMENDED

If your WSL IP is `172.29.128.1`, it's already allowed in the security group:

```bash
# In WSL
cd /mnt/d/EMBERS-BANK/DiffRhythm-LINUX

# Deploy backend
scp -r backend/ ubuntu@52.0.207.242:/home/ubuntu/app/

# Deploy test scripts
scp test_server_implementation.py ubuntu@52.0.207.242:/home/ubuntu/app/
scp test_payment_flow.py ubuntu@52.0.207.242:/home/ubuntu/app/

# SSH and restart service
ssh ubuntu@52.0.207.242
sudo systemctl restart burntbeats-api
cd /home/ubuntu/app
python3 test_server_implementation.py
```

### Solution 2: Use Mobile Hotspot / Different Network

1. Connect to mobile hotspot or different network
2. Get new IP address
3. Add new IP to security group:
   ```powershell
   $NEW_IP = (Invoke-WebRequest -Uri https://api.ipify.org -UseBasicParsing).Content
   aws ec2 authorize-security-group-ingress --group-id sg-0381e5cf859d3feb4 --protocol tcp --port 22 --cidr "$NEW_IP/32"
   ```
4. Deploy from new network

### Solution 3: Enable SSM and Use Systems Manager

**Step 1: Create IAM Role for SSM**

```powershell
# Create IAM role (one-time setup)
aws iam create-role --role-name EC2-SSM-Role --assume-role-policy-document '{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "ec2.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}'

# Attach SSM policy
aws iam attach-role-policy --role-name EC2-SSM-Role --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

# Create instance profile
aws iam create-instance-profile --instance-profile-name EC2-SSM-Profile
aws iam add-role-to-instance-profile --instance-profile-name EC2-SSM-Profile --role-name EC2-SSM-Role

# Attach to instance
aws ec2 associate-iam-instance-profile --instance-id i-0aedf69f3127e24f8 --iam-instance-profile Name=EC2-SSM-Profile
```

**Step 2: Wait for SSM to connect (5-10 minutes)**

**Step 3: Use SSM to deploy**

```powershell
# Start SSM session
aws ssm start-session --target i-0aedf69f3127e24f8

# In SSM session, deploy files via S3 or direct copy
```

### Solution 4: S3 + SSM Run Command (After Enabling SSM)

```powershell
# 1. Create S3 bucket
$BUCKET = "diffrhythm-deploy-$(Get-Date -Format yyyyMMdd)"
aws s3 mb s3://$BUCKET

# 2. Upload files
cd d:\EMBERS-BANK\DiffRhythm-LINUX
aws s3 cp backend/ s3://$BUCKET/backend/ --recursive
aws s3 cp test_server_implementation.py s3://$BUCKET/
aws s3 cp test_payment_flow.py s3://$BUCKET/

# 3. Run command on instance (after SSM is enabled)
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

### Solution 5: Use AWS CloudShell

1. Open AWS Console → CloudShell
2. Upload files to CloudShell
3. Deploy from CloudShell (different network, may work)

## Quick Fix: Try WSL First

**Check if WSL is available and has IP 172.29.128.1:**

```powershell
# Check WSL IP
wsl hostname -I

# If it's 172.29.128.1, you're good to go!
# Deploy from WSL (see Solution 1 above)
```

## Alternative: Temporary Port Change

If you have console access or another way in:

1. Change SSH port on server to 2222
2. Update security group to allow port 2222
3. Connect via new port

**This requires initial server access though.**

## Immediate Action Plan

1. **Try WSL first** (easiest if available)
2. **If WSL doesn't work**, try mobile hotspot
3. **If neither works**, enable SSM (takes 10-15 minutes setup)
4. **Deploy via SSM** once connected

## After Successful Connection

Once you can connect (via WSL, different network, or SSM):

```bash
# Deploy files
scp -r backend/ ubuntu@52.0.207.242:/home/ubuntu/app/
scp test_server_implementation.py ubuntu@52.0.207.242:/home/ubuntu/app/
scp test_payment_flow.py ubuntu@52.0.207.242:/home/ubuntu/app/

# SSH to server
ssh ubuntu@52.0.207.242

# Restart service
sudo systemctl restart burntbeats-api

# Run tests
cd /home/ubuntu/app
python3 test_server_implementation.py
```

---

**Status:** Network blocking SSH from current location  
**Best Option:** Try WSL first, then mobile hotspot, then enable SSM
