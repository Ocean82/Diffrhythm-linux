# SSH Connection Solution

**Date:** January 24, 2026  
**Issue:** SSH timeout to 52.0.207.242

## Diagnosis Results

### ✅ Instance Status
- **Instance ID:** i-0aedf69f3127e24f8
- **Status:** Running
- **Public IP:** 52.0.207.242
- **Your IP:** 68.251.50.12 ✅ (Allowed in security group)

### ❌ Network Test Results
- **Port 22 TCP Test:** FAILED (Connection timeout)
- **Ping Test:** FAILED (Timeout)
- **SSM Status:** ConnectionLost

### Root Cause
Port 22 is **not reachable** from your current network. This could be due to:
1. **ISP/Router blocking outbound SSH** (port 22)
2. **Windows Firewall** blocking outbound connections
3. **Network ACL** restrictions (checking...)
4. **Instance firewall** (ufw) blocking connections

## Solutions

### Solution 1: Use AWS Systems Manager (Recommended)

If SSM can be configured, this bypasses SSH entirely:

```powershell
# Check if SSM agent is installed (requires IAM role with SSM permissions)
aws ssm describe-instance-information --filters "Key=InstanceIds,Values=i-0aedf69f3127e24f8"

# If available, start session
aws ssm start-session --target i-0aedf69f3127e24f8
```

**To enable SSM:**
1. Instance needs IAM role with `AmazonSSMManagedInstanceCore` policy
2. SSM Agent must be installed (usually pre-installed on Amazon Linux/Ubuntu)

### Solution 2: Change SSH Port (If You Have Access)

If you can access the instance another way, change SSH to a non-blocked port:

```bash
# On server (via console or existing access)
sudo nano /etc/ssh/sshd_config
# Change: Port 22 to Port 2222 (or another port)

sudo systemctl restart sshd

# Update security group to allow new port
aws ec2 authorize-security-group-ingress --group-id sg-0381e5cf859d3feb4 --protocol tcp --port 2222 --cidr 68.251.50.12/32
```

### Solution 3: Use Different Network

Try from:
- **Mobile hotspot** (different ISP)
- **WSL** (if your WSL IP is 172.29.128.1, it's already allowed)
- **AWS CloudShell** (from AWS Console)
- **VPN** to different network

### Solution 4: Port Forward via AWS Systems Manager

If SSM works, use port forwarding:

```powershell
# Forward port 22 to local port 2222
aws ssm start-session `
  --target i-0aedf69f3127e24f8 `
  --document-name AWS-StartPortForwardingSession `
  --parameters '{"portNumber":["22"],"localPortNumber":["2222"]}'

# In another terminal, SSH to localhost
ssh -i "C:\Users\sammy\.ssh\server_saver_key" -p 2222 ubuntu@localhost
```

### Solution 5: Deploy via AWS CodeDeploy / S3

Upload files to S3, then pull on server:

```powershell
# Upload to S3
aws s3 cp backend/ s3://your-bucket/deployment/backend/ --recursive

# On server (via SSM or console)
aws s3 sync s3://your-bucket/deployment/backend/ /home/ubuntu/app/backend/
```

### Solution 6: Use EC2 Instance Connect

If available in your region:

```powershell
# Connect via browser-based terminal
aws ec2-instance-connect send-ssh-public-key `
  --instance-id i-0aedf69f3127e24f8 `
  --availability-zone us-east-1a `
  --instance-os-user ubuntu `
  --ssh-public-key file://~/.ssh/server_saver_key.pub
```

## Immediate Workaround: Deploy via Alternative Method

Since SSH is blocked, use one of these:

### Option A: AWS Systems Manager (If Available)

```powershell
# Check SSM status
aws ssm describe-instance-information --filters "Key=InstanceIds,Values=i-0aedf69f3127e24f8"

# If available, start session
aws ssm start-session --target i-0aedf69f3127e24f8

# Then deploy files via SSM session
```

### Option B: S3 + SSM Run Command

```powershell
# 1. Upload files to S3
aws s3 mb s3://diffrhythm-deployment-$(Get-Date -Format yyyyMMdd)  # Create bucket
aws s3 cp backend/ s3://diffrhythm-deployment-$(Get-Date -Format yyyyMMdd)/backend/ --recursive

# 2. Run command on instance to download
aws ssm send-command `
  --instance-ids i-0aedf69f3127e24f8 `
  --document-name "AWS-RunShellScript" `
  --parameters 'commands=["aws s3 sync s3://BUCKET_NAME/backend/ /home/ubuntu/app/backend/", "sudo systemctl restart burntbeats-api"]'
```

### Option C: Use WSL (If 172.29.128.1 is Your WSL IP)

```bash
# In WSL
cd /mnt/d/EMBERS-BANK/DiffRhythm-LINUX
scp -r backend/ ubuntu@52.0.207.242:/home/ubuntu/app/
```

## Next Steps

1. **Try AWS Systems Manager first:**
   ```powershell
   aws ssm start-session --target i-0aedf69f3127e24f8
   ```

2. **If SSM doesn't work, try WSL** (if 172.29.128.1 is your WSL IP)

3. **If neither works, use S3 + SSM Run Command** to deploy files

4. **After deployment, restart service:**
   ```bash
   sudo systemctl restart burntbeats-api
   ```

---

**Status:** SSH blocked at network level  
**Recommended:** Use AWS Systems Manager or S3 deployment method
