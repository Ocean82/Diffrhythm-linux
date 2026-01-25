# SSH Key Setup Instructions

**Date:** January 24, 2026  
**Instance:** `i-0aedf69f3127e24f8` (BurntBeats-Production)  
**Required Key:** `Burnt-Beats-KEY.pem`

## Current Status

❌ **Burnt-Beats-KEY.pem not found in local directories**

The deploy script requires `Burnt-Beats-KEY.pem` which is the key pair associated with this EC2 instance.

## How to Download the Key

### Option 1: AWS Console (Recommended)

1. Go to **AWS Console** → **EC2** → **Key Pairs**
2. Find **Burnt-Beats-KEY** in the list
3. Click **Actions** → **Download**
4. Save the file as `Burnt-Beats-KEY.pem`

**Note:** If the key was created when launching the instance, you can only download it once. If you don't have it, you'll need to:
- Create a new key pair, or
- Use an existing key that was saved when the instance was created

### Option 2: AWS CLI

```bash
# If the key is available via AWS CLI
aws ec2 describe-key-pairs --key-names Burnt-Beats-KEY
```

## Where to Place the Key

The deploy script will check these locations (in order):

1. `C:\Users\sammy\OneDrive\Desktop\AWS ITEMS\Burnt-Beats-KEY.pem` ⭐ **Recommended**
2. `C:\Users\sammy\OneDrive\Desktop\KEYS\Burnt-Beats-KEY.pem`
3. `C:\Users\sammy\.ssh\Burnt-Beats-KEY.pem`

Or set the `SSH_KEY` environment variable:

```bash
export SSH_KEY="C:/path/to/Burnt-Beats-KEY.pem"
bash scripts/deploy-to-server.sh
```

## Testing the Key

Once you have the key, test it:

```bash
# Test connection
ssh -i "C:/Users/sammy/OneDrive/Desktop/AWS ITEMS/Burnt-Beats-KEY.pem" \
    ubuntu@ec2-52-0-207-242.compute-1.amazonaws.com

# Or use the test script
bash scripts/test-keys.sh
```

## Alternative: Use Existing Key

If you have access to the instance via another method (AWS Systems Manager, etc.), you can:

1. Add your public key to `~/.ssh/authorized_keys` on the server
2. Then use your own key for deployment

## Files Found (But Not Working)

The following keys were found but don't work with this instance:

- `C:\Users\sammy\OneDrive\Desktop\AWS ITEMS\aws SSH key.pem` ❌
- `C:\Users\sammy\OneDrive\Desktop\AWS ITEMS\aws-ssh.pem` ❌
- `C:\Users\sammy\OneDrive\Desktop\KEYS\aws ssh.pem` ❌ (permission issues)
- `C:\Users\sammy\OneDrive\Desktop\KEYS\bb-key.pem` ❌ (permission issues)
- `C:\Users\sammy\OneDrive\Desktop\KEYS\working_ssh_key.pem` ❌

## Next Steps

1. **Download Burnt-Beats-KEY.pem** from AWS Console
2. **Place it in:** `C:\Users\sammy\OneDrive\Desktop\AWS ITEMS\Burnt-Beats-KEY.pem`
3. **Test connection:**
   ```bash
   bash scripts/test-keys.sh
   ```
4. **Deploy:**
   ```bash
   bash scripts/deploy-to-server.sh
   ```

## Troubleshooting

### Permission Denied Errors

If you see "Permission denied" when accessing keys in OneDrive:

- OneDrive may be syncing or files may be locked
- Try copying the key to `C:\Users\sammy\.ssh\` instead
- Or set `SSH_KEY` environment variable to point directly to the file

### Key Not Found

If the key doesn't exist in AWS Console:

- The key may have been deleted
- You may need to create a new key pair and associate it with the instance
- Or use AWS Systems Manager Session Manager to access the instance

---

**Once Burnt-Beats-KEY.pem is in place, the deployment should work correctly.**
