# Final Deployment Instructions

**Date:** January 24, 2026  
**SSH Issue:** Network blocking port 22 from Windows

## SSH Debug Results

### Findings
- ✅ Instance running: i-0aedf69f3127e24f8
- ✅ Security group allows: 68.251.50.12/32 (your IP) and 172.29.128.1/32
- ❌ Port 22 blocked from Windows network
- ✅ WSL IP: 192.168.92.175 (added to security group)

## Deployment Options

### Option 1: Deploy from WSL ⭐ RECOMMENDED

WSL IP has been added to security group. Try:

```bash
# In WSL
cd /mnt/d/EMBERS-BANK/DiffRhythm-LINUX

# Test SSH connection
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242 "echo 'Connection test'"

# If successful, deploy:
scp -r backend/ ubuntu@52.0.207.242:/home/ubuntu/app/
scp test_server_implementation.py ubuntu@52.0.207.242:/home/ubuntu/app/
scp test_payment_flow.py ubuntu@52.0.207.242:/home/ubuntu/app/

# SSH and restart
ssh -i ~/.ssh/server_saver_key ubuntu@52.0.207.242
sudo systemctl restart burntbeats-api
cd /home/ubuntu/app
python3 test_server_implementation.py
```

### Option 2: Use Mobile Hotspot

1. Connect to mobile hotspot
2. Get new IP: `(Invoke-WebRequest -Uri https://api.ipify.org -UseBasicParsing).Content`
3. Add to security group:
   ```powershell
   $NEW_IP = (Invoke-WebRequest -Uri https://api.ipify.org -UseBasicParsing).Content
   aws ec2 authorize-security-group-ingress --group-id sg-0381e5cf859d3feb4 --protocol tcp --port 22 --cidr "$NEW_IP/32"
   ```
4. Deploy from mobile network

### Option 3: Enable AWS Systems Manager

**One-time setup (10-15 minutes):**

```powershell
# 1. Create IAM role
aws iam create-role --role-name EC2-SSM-Role --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

# 2. Attach SSM policy
aws iam attach-role-policy --role-name EC2-SSM-Role --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

# 3. Create instance profile
aws iam create-instance-profile --instance-profile-name EC2-SSM-Profile
aws iam add-role-to-instance-profile --instance-profile-name EC2-SSM-Profile --role-name EC2-SSM-Role

# 4. Attach to instance
aws ec2 associate-iam-instance-profile --instance-id i-0aedf69f3127e24f8 --iam-instance-profile Name=EC2-SSM-Profile

# 5. Wait 5-10 minutes for SSM agent to connect

# 6. Start session
aws ssm start-session --target i-0aedf69f3127e24f8
```

## Current Security Group Rules (Port 22)

- ✅ `172.29.128.1/32` - WSL ssh
- ✅ `68.251.50.12/32` - ssh (your Windows IP - blocked by network)
- ✅ `192.168.92.175/32` - WSL SSH access (just added)

## Quick Test Commands

### Test from WSL
```bash
# In WSL
ssh -i ~/.ssh/server_saver_key -v ubuntu@52.0.207.242
```

### Test Network Connectivity
```powershell
# From Windows
Test-NetConnection -ComputerName 52.0.207.242 -Port 22

# From WSL
nc -zv 52.0.207.242 22
```

## Next Steps

1. **Try WSL deployment first** (easiest)
2. **If WSL fails**, try mobile hotspot
3. **If both fail**, enable SSM and deploy via Systems Manager
4. **After successful deployment**, restart service and run tests

---

**Status:** WSL IP added to security group  
**Action:** Try deploying from WSL first
