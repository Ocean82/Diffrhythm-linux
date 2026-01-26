# Environment Variables Comparison Report
**Date**: January 26, 2026  
**Local File**: `C:\Users\sammy\OneDrive\Desktop\.env` (Phoenix Project)  
**Server File**: `/opt/diffrhythm/.env` (DiffRhythm-LINUX Backend)

## Executive Summary

The local `.env` file contains **LIVE Stripe keys** and other configuration that should be added to the server's `.env` file. The server currently has **empty Stripe keys**, which explains why payment verification may not be working correctly.

## Critical Missing Items on Server

### 1. Stripe Configuration (CRITICAL - Currently Empty)
**Status**: ⚠️ **MISSING - REQUIRED FOR PAYMENTS**

| Key | Local Value | Server Value | Action |
|-----|-------------|--------------|--------|
| `STRIPE_SECRET_KEY` | `***REMOVED***` | *(empty)* | ✅ **ADD** |
| `STRIPE_PUBLISHABLE_KEY` | `***REMOVED***` | *(empty)* | ✅ **ADD** |
| `STRIPE_WEBHOOK_SECRET` | `***REMOVED***` | *(empty)* | ✅ **ADD** |
| `STRIPE_ACCOUNT_ID` | `***REMOVED***` | *(missing)* | ✅ **ADD** |

**Impact**: Payment verification will fail without these keys.

## Configuration Differences

### 2. CORS Configuration
| Key | Local Value | Server Value | Recommendation |
|-----|-------------|--------------|----------------|
| `CORS_ORIGINS` | `*` | `*` | ✅ Same |
| `CORS_ALLOW_ORIGINS` | `https://burntbeats.com,http://localhost:3000,http://localhost:5173,http://52.0.207.242` | *(missing)* | ⚠️ Consider adding for frontend integration |

### 3. Rate Limiting
| Key | Local Value | Server Value | Status |
|-----|-------------|--------------|--------|
| `ENABLE_RATE_LIMIT` | *(not in local)* | `false` | ✅ Currently disabled (was causing issues) |

### 4. Payment Requirement
| Key | Local Value | Server Value | Status |
|-----|-------------|--------------|--------|
| `REQUIRE_PAYMENT_FOR_GENERATION` | *(not in local)* | `false` | ✅ Currently disabled for testing |

## Keys in Local File (Not Needed for DiffRhythm Backend)

These keys are specific to the Phoenix project and **should NOT** be added to the DiffRhythm server:

- Database configuration (PostgreSQL RDS)
- Redis cache URL
- AWS credentials (unless needed for S3 storage)
- Email configuration (SMTP)
- Clerk authentication
- RVC voice cloning paths
- MVP pipeline configuration
- Session secrets (JWT, CSRF)
- Client/Frontend URLs

## Keys in Server File (DiffRhythm-Specific)

These are correctly configured for the DiffRhythm backend and should remain:

- `MODEL_CACHE_DIR` - Model storage path
- `HUGGINGFACE_HUB_CACHE` - HuggingFace cache
- `CPU_STEPS` - Generation quality settings
- `CPU_CFG_STRENGTH` - Generation quality settings
- `DEVICE` - CPU configuration
- `GENERATION_TIMEOUT` - Timeout settings

## Recommended Actions

### Immediate (Critical)

1. **Add Stripe Keys to Server** ⚠️ **REQUIRED**
   ```bash
   # SSH to server and edit /opt/diffrhythm/.env
   sudo nano /opt/diffrhythm/.env
   ```
   
   Add these lines:
   ```env
   STRIPE_SECRET_KEY=***REMOVED***
   STRIPE_PUBLISHABLE_KEY=***REMOVED***
   STRIPE_WEBHOOK_SECRET=***REMOVED***
   STRIPE_ACCOUNT_ID=***REMOVED***
   ```

2. **Restart Docker Container** after adding keys
   ```bash
   cd /opt/diffrhythm
   sudo docker-compose -f docker-compose.prod.yml restart diffrhythm-api
   ```

### Optional (For Better Integration)

3. **Add CORS Origins** (if frontend needs specific origins)
   ```env
   CORS_ALLOW_ORIGINS=https://burntbeats.com,http://localhost:3000,http://localhost:5173,http://52.0.207.242
   ```

## Security Notes

⚠️ **IMPORTANT**: 
- The Stripe keys in the local file are **LIVE production keys**
- Ensure the server `.env` file has proper permissions (not world-readable)
- Consider using environment variable injection in Docker instead of plain text files
- Rotate keys if they've been exposed

## Verification Steps

After adding the Stripe keys:

1. Verify keys are loaded:
   ```bash
   sudo docker exec diffrhythm-api env | grep STRIPE
   ```

2. Test payment verification:
   ```bash
   # Use the test script from previous investigation
   python3 /tmp/test_api_payment.py
   ```

3. Check container logs:
   ```bash
   sudo docker logs diffrhythm-api | grep -i stripe
   ```

## Summary

| Category | Status | Action Required |
|----------|--------|-----------------|
| Stripe Keys | ⚠️ **MISSING** | ✅ **ADD IMMEDIATELY** |
| CORS Configuration | ⚠️ Basic | ⚠️ Optional enhancement |
| Rate Limiting | ✅ Disabled | ✅ OK for now |
| Payment Requirement | ✅ Disabled | ✅ OK for testing |

---

**Priority**: 🔴 **HIGH** - Stripe keys must be added for payment functionality  
**Impact**: Payment verification will fail without these keys
