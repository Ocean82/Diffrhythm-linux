# SSH Connection Issue - Summary & Solutions

**Date:** January 24, 2026  
**Instance:** i-0aedf69f3127e24f8 (52.0.207.242)

## Problem Identified

**SSH connection timing out** despite:
- ✅ Instance is running
- ✅ Security group allows your IP (68.251.50.12/32)
- ✅ Port 22 is open in security group

**Root Cause:** Network-level blocking of outbound SSH (port 22) from your current location.

## AWS Instance Details

- **Instance ID:** i-0aedf69f3127e24f8
- **Status:** Running ✅
- **Public IP:** 52.0.207.242
- **Security Group:** sg-0381e5cf859d3feb4
- **Key Name:** Burnt-Beats-KEY
- **Your IP:** 68.251.50.12 (already allowed)

## Current Security Group Rules (Port 22)

- `172.29.128.1/32` - WSL ssh
- `68.251.50.12/32` - ssh (your IP - but network blocks it)
- `192.168.92.175/32` - WSL IP (attempted to add)

## Solutions (In Order of Recommendation)

### 1. Use Mobile Hotspot ⭐ FASTEST

**Steps:**
1. Connect to mobile hotspot
2. Get new IP and add to security group
3. Deploy via SSH

**Commands:**
```powershell
# Get new IP
$NEW_IP = (Invoke-WebRequest -Uri https://api.ipify.org -UseBasicParsing).Content

# Add to security group (use JSON file for proper escaping)
# Create file: sg-rule.json
# {
#   "IpProtocol": "tcp",
#   "FromPort": 22,
#   "ToPort": 22,
#   "IpRanges": [{"CidrIp": "NEW_IP/32", "Description": "Mobile-Hotspot"}]
# }

aws ec2 authorize-security-group-ingress --group-id sg-0381e5cf859d3feb4 --ip-permissions file://sg-rule.json

# Deploy
scp -i "C:\Users\sammy\.ssh\server_saver_key" -r backend/ ubuntu@52.0.207.242:/home/ubuntu/app/
```

### 2. Enable AWS Systems Manager ⭐ BEST LONG-TERM

**Allows deployment without SSH:**

See `SSH_SOLUTION.md` for complete SSM setup instructions.

**Quick setup:**
1. Create IAM role with SSM permissions
2. Attach to instance
3. Wait 5-10 minutes
4. Use `aws ssm start-session --target i-0aedf69f3127e24f8`

### 3. Use AWS CloudShell

1. Open AWS Console
2. Click CloudShell (top right)
3. Upload files and deploy from CloudShell

## Deployment Commands (Once SSH Works)

```bash
# Deploy backend
scp -i "C:\Users\sammy\.ssh\server_saver_key" -r backend/ ubuntu@52.0.207.242:/home/ubuntu/app/

# Deploy test scripts
scp -i "C:\Users\sammy\.ssh\server_saver_key" test_server_implementation.py ubuntu@52.0.207.242:/home/ubuntu/app/
scp -i "C:\Users\sammy\.ssh\server_saver_key" test_payment_flow.py ubuntu@52.0.207.242:/home/ubuntu/app/

# SSH and restart
ssh -i "C:\Users\sammy\.ssh\server_saver_key" ubuntu@52.0.207.242
sudo systemctl restart burntbeats-api
cd /home/ubuntu/app
python3 test_server_implementation.py
```

## Files Ready for Deployment

✅ `backend/api.py` (29.17 KB)
✅ `backend/payment_verification.py` (3.04 KB)
✅ `backend/config.py` (3.85 KB)
✅ `test_server_implementation.py` (8.07 KB)
✅ `test_payment_flow.py` (8.39 KB)

## Next Steps

1. **Try mobile hotspot** (quickest solution)
2. **Or enable SSM** (better for future deployments)
3. **Deploy code** once connection works
4. **Restart service** and run tests

---

**Status:** Network blocking SSH  
**Action:** Use mobile hotspot or enable SSM to deploy
