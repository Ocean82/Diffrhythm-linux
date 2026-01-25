# Stripe Pricing Implementation Summary

**Date:** January 23, 2026  
**Status:** ✅ Pricing System Updated

## New Pricing Model

### Pricing Tiers Implemented

| Product | Price | Duration | Notes |
|---------|-------|----------|-------|
| Single 2-Minute Song | $2.00 | Up to 120s | Base price |
| Extended Song (up to 4 min) | $3.50 | 121-240s | Extended duration |
| Commercial License Add-On | $15.00 | Any song | Add-on option |
| Bulk Pack (10 Songs) | $18.00 | 10 songs | 10% discount |
| Bulk Pack (50 Songs) | $80.00 | 50 songs | 20% discount |

## Implementation Details

### Files Updated

1. **`/home/ubuntu/app/backend/src/utils/pricing.py`**
   - ✅ Updated to duration-based pricing
   - ✅ Added support for extended songs
   - ✅ Added commercial license add-on
   - ✅ Added bulk pack pricing (10 and 50 songs)
   - ✅ Maintains backward compatibility

2. **`/home/ubuntu/app/backend/src/schemas/payments.py`**
   - ✅ Updated `CreatePaymentIntentRequest` with new fields:
     - `duration_seconds` (for new generation)
     - `is_extended` (for extended songs)
     - `commercial_license` (boolean flag)
     - `bulk_pack_size` (10 or 50)
   - ✅ Added `CalculatePriceRequest` schema
   - ✅ Added `CalculatePriceResponse` schema

### Pricing Functions

**`calculate_price_cents()`**
```python
calculate_price_cents(
    duration_seconds: int,
    is_extended: bool = False,
    commercial_license: bool = False,
    bulk_pack_size: Optional[int] = None
) -> int
```

**Examples:**
- 120s single song: $2.00 (200 cents)
- 180s extended: $3.50 (350 cents)
- 120s with commercial: $17.00 (1700 cents = $2.00 + $15.00)
- Bulk 10 songs: $18.00 (1800 cents)
- Bulk 50 songs: $80.00 (8000 cents)

**`calculate_price_for_duration()`**
Returns all pricing options for a given duration:
- Base price
- Price with commercial license
- Bulk pack prices
- All in cents and dollars

**`validate_price_for_duration()`**
Validates that charged price matches duration and options.

## Next Steps Required

### 1. Update Payment Endpoints

**File:** `/home/ubuntu/app/backend/src/api/v1/payments.py`

- Update `create_payment_intent` to:
  - Accept `duration_seconds` instead of requiring `song_id`
  - Calculate price using new pricing function
  - Validate price against duration
  - Handle commercial license flag
  - Handle bulk pack purchases

- Add new endpoint `calculate_price`:
  ```python
  @router.get("/calculate-price")
  async def calculate_price(
      duration: int,
      is_extended: bool = False,
      commercial_license: bool = False,
      bulk_pack_size: Optional[int] = None
  ):
      """Calculate price for a song based on duration"""
      from ...utils.pricing import calculate_price_for_duration
      return calculate_price_for_duration(duration)
  ```

### 2. Update Generation Endpoint

**File:** `/home/ubuntu/app/backend/src/api/v1/generation.py`

- Add payment verification before generation:
  - Require `payment_intent_id` in request
  - Verify payment intent is confirmed
  - Verify price matches requested duration
  - Store payment info with job

### 3. Update Webhook Handler

**File:** `/home/ubuntu/app/backend/src/api/stripe_webhooks.py`

- Update to verify duration-based pricing
- Handle commercial license metadata
- Handle bulk pack metadata
- Grant appropriate access based on purchase type

### 4. Configure Stripe Keys

**File:** `/home/ubuntu/app/backend/.env`

```bash
STRIPE_SECRET_KEY=sk_live_...
STRIPE_PUBLISHABLE_KEY=pk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
```

### 5. Testing Checklist

- [ ] Test single 2-minute song pricing
- [ ] Test extended song pricing
- [ ] Test commercial license add-on
- [ ] Test bulk pack 10 songs
- [ ] Test bulk pack 50 songs
- [ ] Test price calculation endpoint
- [ ] Test payment intent creation
- [ ] Test payment → generation flow
- [ ] Test webhook handling

## Testing Commands

```bash
# Test pricing functions
cd /home/ubuntu/app/backend
python3 /tmp/test_pricing.py

# Expected output:
# 120s single: $2.00
# 180s extended: $3.50
# 120s with commercial: $17.00
# Bulk 10: $18.00
# Bulk 50: $80.00
```

## Pricing Validation

The system validates:
- Duration must be 95-240 seconds
- Extended songs: 121-240 seconds
- Single songs: up to 120 seconds
- Bulk packs: 10 or 50 songs only
- Price must match duration and options

---

**Status:** ✅ **PRICING LOGIC UPDATED**  
**Next:** Update payment endpoints to use new pricing
