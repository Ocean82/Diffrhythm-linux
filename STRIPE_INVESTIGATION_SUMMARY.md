# Stripe Payment System Investigation Summary

**Date:** January 23, 2026  
**Status:** Investigation tools prepared, ready for server access

## Current Status

Based on service logs from earlier investigation:
- ⚠️ **Stripe not configured** - Service logs show warnings
- Service on port 8001 has payment code but Stripe keys missing
- Payment endpoints may exist but are non-functional

## Investigation Tools Created

1. **Investigation Script**: `scripts/investigate_stripe_payments.sh`
   - Comprehensive script to check all aspects of Stripe integration
   - Can be run directly on server

2. **Investigation Guide**: `STRIPE_PAYMENT_INVESTIGATION.md`
   - Step-by-step manual investigation commands
   - Expected findings and key files to check

3. **Implementation Guide**: `STRIPE_PAYMENT_IMPLEMENTATION_GUIDE.md`
   - Expected implementation structure
   - Pricing model recommendations
   - Testing checklist

## Quick Investigation Commands

When SSH access is available, run these commands:

```bash
# 1. Check Stripe configuration
cat /home/ubuntu/app/backend/.env | grep -i stripe

# 2. Find payment files
find /home/ubuntu/app/backend -name '*payment*' -o -name '*stripe*'

# 3. Check payment code
cat /home/ubuntu/app/backend/src/api/payments.py
cat /home/ubuntu/app/backend/src/api/v1/payments.py

# 4. Check pricing logic
grep -r 'calculate.*price\|price.*duration' /home/ubuntu/app/backend/src --include='*.py'

# 5. Check service logs
sudo journalctl -u burntbeats-api | grep -i stripe

# 6. Test payment endpoints
curl http://127.0.0.1:8001/api/v1/payments/checkout
```

## Expected Issues to Find

1. **Missing Stripe Keys**
   - `STRIPE_SECRET_KEY` not set
   - `STRIPE_PUBLISHABLE_KEY` not set
   - `STRIPE_WEBHOOK_SECRET` not set

2. **Payment Code Not Integrated**
   - Generation endpoint doesn't check payment
   - No price calculation before generation
   - Payment verification missing

3. **Pricing Logic Missing**
   - No function to calculate price based on duration
   - No pricing tiers defined
   - No integration with Stripe API

## Recommended Pricing Model

Based on song duration:
- **95 seconds (base)**: $9.99
- **96-180 seconds**: $9.99 + ($0.10 × (duration - 95))
- **181-285 seconds**: $9.99 + ($0.10 × (duration - 95))

Or simpler tiered:
- **95 seconds**: $9.99
- **96-180 seconds**: $14.99
- **181-285 seconds**: $19.99

## Next Steps

1. **When SSH access is available:**
   - Run investigation script: `bash scripts/investigate_stripe_payments.sh`
   - Review payment code files
   - Check environment configuration

2. **Fix Issues Found:**
   - Add Stripe keys to `.env`
   - Implement pricing calculation
   - Integrate payment verification in generation endpoint
   - Configure webhook handler

3. **Test Implementation:**
   - Test payment calculation
   - Test Stripe checkout flow
   - Test webhook handling
   - Test end-to-end payment → generation flow

## Files to Review

1. `/home/ubuntu/app/backend/src/api/payments.py`
2. `/home/ubuntu/app/backend/src/api/v1/payments.py`
3. `/home/ubuntu/app/backend/src/api/v1/generation.py`
4. `/home/ubuntu/app/backend/.env`
5. `/home/ubuntu/app/backend/src/services/payment_service.py` (if exists)

---

**Status:** ⏳ **AWAITING SERVER ACCESS**  
**Action:** Run investigation script when SSH connection is available
