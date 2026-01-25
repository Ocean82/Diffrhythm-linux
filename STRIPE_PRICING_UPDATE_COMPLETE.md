# Stripe Pricing Update - Complete

**Date:** January 23, 2026  
**Status:** ✅ Pricing System Updated

## New Pricing Model Implemented

### Pricing Tiers

1. **Single 2-Minute Song**: $2.00
   - Duration: Up to 120 seconds (2 minutes)
   - Price: $2.00 (200 cents)

2. **Extended Song (up to 4 min)**: $3.50
   - Duration: 121-240 seconds (2-4 minutes)
   - Price: $3.50 (350 cents)

3. **Commercial License Add-On**: $15.00
   - Can be added to any song purchase
   - Price: $15.00 (1500 cents)

4. **Bulk Pack (10 Songs)**: $18.00
   - 10 songs at discounted rate
   - Price: $18.00 (1800 cents) - 10% discount
   - Per song: $1.80 (vs $2.00 regular)

5. **Bulk Pack (50 Songs)**: $80.00
   - 50 songs at discounted rate
   - Price: $80.00 (8000 cents) - 20% discount
   - Per song: $1.60 (vs $2.00 regular)

## Files Updated

### 1. `/home/ubuntu/app/backend/src/utils/pricing.py`
- Updated to calculate prices based on duration
- Added support for extended songs
- Added commercial license add-on
- Added bulk pack pricing
- Maintains backward compatibility with legacy file-size pricing

### 2. `/home/ubuntu/app/backend/src/schemas/payments.py`
- Updated `CreatePaymentIntentRequest` to include:
  - `duration_seconds` (for new generation)
  - `is_extended` (for extended songs)
  - `commercial_license` (for commercial license add-on)
  - `bulk_pack_size` (for bulk packs: 10 or 50)
- Added `CalculatePriceRequest` and `CalculatePriceResponse` schemas

## Pricing Functions

### `calculate_price_cents()`
```python
calculate_price_cents(
    duration_seconds: int,
    is_extended: bool = False,
    commercial_license: bool = False,
    bulk_pack_size: Optional[int] = None
) -> int
```

**Examples:**
- 120s single: $2.00
- 180s extended: $3.50
- 120s with commercial: $2.00 + $15.00 = $17.00
- Bulk 10: $18.00
- Bulk 50: $80.00

### `calculate_price_for_duration()`
Returns comprehensive pricing options for a given duration:
- Base price
- Price with commercial license
- Bulk pack prices
- All in cents and dollars

### `validate_price_for_duration()`
Validates that the charged price matches the duration and options.

## Next Steps

1. **Update Payment Endpoints** (`src/api/v1/payments.py`)
   - Update `create_payment_intent` to use new pricing
   - Add `calculate_price` endpoint
   - Validate prices using duration-based logic

2. **Update Generation Endpoint** (`src/api/v1/generation.py`)
   - Integrate payment verification before generation
   - Calculate price based on requested duration
   - Require payment intent confirmation

3. **Update Webhook Handler** (`src/api/stripe_webhooks.py`)
   - Verify duration-based pricing in webhook
   - Handle commercial license flags
   - Handle bulk pack purchases

4. **Configure Stripe Keys**
   - Add `STRIPE_SECRET_KEY` to `.env`
   - Add `STRIPE_PUBLISHABLE_KEY` to `.env`
   - Add `STRIPE_WEBHOOK_SECRET` to `.env`

5. **Test Payment Flow**
   - Test single song purchase
   - Test extended song purchase
   - Test commercial license add-on
   - Test bulk pack purchases
   - Test payment → generation flow

## Testing

Run pricing tests:
```bash
python3 -c "
import sys
sys.path.insert(0, '/home/ubuntu/app/backend')
from src.utils.pricing import calculate_price_cents

# Test cases
print('Single 2-min (120s):', calculate_price_cents(120) / 100)
print('Extended (180s):', calculate_price_cents(180, is_extended=True) / 100)
print('With commercial (120s):', calculate_price_cents(120, commercial_license=True) / 100)
print('Bulk 10:', calculate_price_cents(120, bulk_pack_size=10) / 100)
print('Bulk 50:', calculate_price_cents(120, bulk_pack_size=50) / 100)
"
```

Expected output:
- Single 2-min (120s): 2.0
- Extended (180s): 3.5
- With commercial (120s): 17.0
- Bulk 10: 18.0
- Bulk 50: 80.0

---

**Status:** ✅ **PRICING SYSTEM UPDATED**  
**Next:** Update payment endpoints and integrate with generation flow
