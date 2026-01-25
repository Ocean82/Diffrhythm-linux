# Stripe Payment Investigation via AWS CLI

**Date:** January 23, 2026  
**Server:** 52.0.207.242  
**Method:** AWS CLI (SSH unavailable)

## AWS CLI Commands to Investigate Stripe

### 1. Get Instance ID
```bash
aws ec2 describe-instances --filters "Name=ip-address,Values=52.0.207.242" --query "Reservations[0].Instances[0].InstanceId" --output text
```

### 2. Check Stripe Configuration via SSM
```bash
INSTANCE_ID=$(aws ec2 describe-instances --filters "Name=ip-address,Values=52.0.207.242" --query "Reservations[0].Instances[0].InstanceId" --output text)

aws ssm send-command \
  --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["cat /home/ubuntu/app/backend/.env | grep -i stripe"]' \
  --output text

# Get command output
COMMAND_ID=<from above output>
aws ssm get-command-invocation --command-id $COMMAND_ID --instance-id $INSTANCE_ID
```

### 3. Find Payment Files
```bash
aws ssm send-command \
  --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["find /home/ubuntu/app/backend -name \"*payment*\" -o -name \"*stripe*\""]' \
  --output text
```

### 4. Check Payment Code
```bash
aws ssm send-command \
  --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["cat /home/ubuntu/app/backend/src/api/payments.py | head -100"]' \
  --output text
```

### 5. Check Pricing Logic
```bash
aws ssm send-command \
  --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["grep -r \"calculate.*price\\|price.*duration\" /home/ubuntu/app/backend/src --include=\"*.py\""]' \
  --output text
```

### 6. Check Service Logs
```bash
aws ssm send-command \
  --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["sudo journalctl -u burntbeats-api --no-pager -n 200 | grep -i stripe"]' \
  --output text
```

## Alternative: Use AWS Systems Manager Session Manager

If SSM agent is installed and configured:

```bash
# Start interactive session
aws ssm start-session --target <INSTANCE_ID>

# Then run commands interactively:
cat /home/ubuntu/app/backend/.env | grep -i stripe
find /home/ubuntu/app/backend -name '*payment*'
cat /home/ubuntu/app/backend/src/api/payments.py
```

## Prerequisites

1. **SSM Agent** must be installed on EC2 instance
2. **IAM Role** with SSM permissions attached to instance
3. **AWS CLI** configured with appropriate credentials

## Quick Investigation Script

Save this as `investigate_stripe_aws.sh`:

```bash
#!/bin/bash
INSTANCE_ID=$(aws ec2 describe-instances --filters "Name=ip-address,Values=52.0.207.242" --query "Reservations[0].Instances[0].InstanceId" --output text)

echo "Instance ID: $INSTANCE_ID"

# Check Stripe config
echo "Checking Stripe configuration..."
aws ssm send-command \
  --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["cat /home/ubuntu/app/backend/.env | grep -i stripe"]' \
  --output text

# Wait and get results
sleep 5
COMMAND_ID=$(aws ssm list-commands --instance-id $INSTANCE_ID --max-items 1 --query "Commands[0].CommandId" --output text)
aws ssm get-command-invocation --command-id $COMMAND_ID --instance-id $INSTANCE_ID --query "StandardOutputContent" --output text
```
