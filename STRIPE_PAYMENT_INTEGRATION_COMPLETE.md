# Stripe Payment Integration - Complete

**Date:** January 23, 2026  
**Status:** ✅ **PAYMENT ENDPOINTS UPDATED**

## Implementation Summary

### ✅ Completed Updates

1. **Payment Endpoint Updated** (`src/api/v1/payments.py`)
   - ✅ Uses new duration-based pricing functions
   - ✅ Supports commercial license add-on
   - ✅ Supports bulk pack purchases (10 and 50 songs)
   - ✅ Validates price against duration
   - ✅ Handles both existing song purchases and new generation prepayment
   - ✅ Added `calculate-price` endpoint

2. **Generation Endpoint Updated** (`src/api/v1/generation.py`)
   - ✅ Calculates price based on requested duration
   - ✅ Verifies payment intent before generation
   - ✅ Validates price matches duration
   - ✅ Stores payment info with job metadata

3. **Pricing System** (`src/utils/pricing.py`)
   - ✅ Duration-based pricing implemented
   - ✅ All pricing tiers supported
   - ✅ Tested and verified

4. **Payment Schemas** (`src/schemas/payments.py`)
   - ✅ Updated with new fields
   - ✅ Supports duration, extended, commercial license, bulk packs

## New Endpoints

### 1. Calculate Price
**GET** `/api/v1/payments/calculate-price`

**Query Parameters:**
- `duration` (required): Song duration in seconds (95-240)
- `is_extended` (optional): Whether extended song (default: false)
- `commercial_license` (optional): Include commercial license (default: false)
- `bulk_pack_size` (optional): Bulk pack size 10 or 50 (default: none)

**Response:**
```json
{
  "duration_seconds": 120,
  "song_type": "single",
  "base_price_cents": 200,
  "base_price_dollars": 2.00,
  "with_commercial_license_cents": 1700,
  "with_commercial_license_dollars": 17.00,
  "commercial_license_addon_cents": 1500,
  "commercial_license_addon_dollars": 15.00,
  "bulk_10_price_cents": 1800,
  "bulk_10_price_dollars": 18.00,
  "bulk_50_price_cents": 8000,
  "bulk_50_price_dollars": 80.00,
  "currency": "usd"
}
```

### 2. Create Payment Intent (Updated)
**POST** `/api/v1/payments/create-intent`

**Request Body:**
```json
{
  "duration_seconds": 120,
  "amount_cents": 200,
  "is_extended": false,
  "commercial_license": false,
  "bulk_pack_size": null,
  "currency": "usd"
}
```

**Response:**
```json
{
  "client_secret": "pi_xxx_secret_xxx",
  "payment_intent_id": "pi_xxx",
  "amount_cents": 200,
  "currency": "usd"
}
```

### 3. Verify Payment Intent
**GET** `/api/v1/payments/verify-payment/{payment_intent_id}`

**Response:**
```json
{
  "payment_intent_id": "pi_xxx",
  "status": "succeeded",
  "amount_cents": 200,
  "currency": "usd",
  "metadata": {...},
  "ready_for_generation": true
}
```

## Generation Flow with Payment

### Step 1: Calculate Price
```bash
GET /api/v1/payments/calculate-price?duration=120
```

### Step 2: Create Payment Intent
```bash
POST /api/v1/payments/create-intent
{
  "duration_seconds": 120,
  "amount_cents": 200,
  "currency": "usd"
}
```

### Step 3: Frontend Confirms Payment
- Frontend uses `client_secret` to confirm payment with Stripe
- Payment status becomes "succeeded"

### Step 4: Generate Song
```bash
POST /api/v1/generate
{
  "text_prompt": "A happy pop song",
  "duration": 120,
  "payment_intent_id": "pi_xxx"
}
```

**Note:** `payment_intent_id` should be added to `GenerateSongRequest` schema.

## Configuration Required

### 1. Add Stripe Keys to `.env`

**File:** `/home/ubuntu/app/backend/.env`

```bash
STRIPE_SECRET_KEY=sk_live_...
STRIPE_PUBLISHABLE_KEY=pk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
REQUIRE_PAYMENT_FOR_GENERATION=true
```

### 2. Update GenerateSongRequest Schema

**File:** `/home/ubuntu/app/backend/src/schemas/generation.py`

Add optional `payment_intent_id` field:
```python
payment_intent_id: Optional[str] = Field(
    None,
    description="Stripe payment intent ID (required if payment is enabled)",
    examples=["pi_xxx"]
)
```

## Testing

### Test Price Calculation
```bash
curl http://127.0.0.1:8001/api/v1/payments/calculate-price?duration=120
```

### Test Payment Intent Creation
```bash
curl -X POST http://127.0.0.1:8001/api/v1/payments/create-intent \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer TOKEN" \
  -d '{
    "duration_seconds": 120,
    "amount_cents": 200,
    "currency": "usd"
  }'
```

### Test Payment Verification
```bash
curl http://127.0.0.1:8001/api/v1/payments/verify-payment/pi_xxx \
  -H "Authorization: Bearer TOKEN"
```

## Pricing Examples

| Duration | Type | Base Price | With Commercial | Bulk 10 | Bulk 50 |
|----------|------|------------|-----------------|---------|---------|
| 120s | Single | $2.00 | $17.00 | $18.00 | $80.00 |
| 180s | Extended | $3.50 | $18.50 | $18.00 | $80.00 |
| 240s | Extended | $3.50 | $18.50 | $18.00 | $80.00 |

## Next Steps

1. **Add `payment_intent_id` to GenerateSongRequest schema**
2. **Configure Stripe keys in `.env`**
3. **Set `REQUIRE_PAYMENT_FOR_GENERATION=true` in settings**
4. **Test complete payment → generation flow**
5. **Update frontend to use new payment flow**

---

**Status:** ✅ **PAYMENT ENDPOINTS UPDATED**  
**Next:** Configure Stripe keys and test end-to-end flow
