# Stripe Payment System Investigation Report

**Date:** January 23, 2026  
**Server:** ubuntu@52.0.207.242  
**Status:** ✅ Investigation Complete

## Executive Summary

### ❌ **CRITICAL ISSUE FOUND**

**Current Implementation:** Pricing is based on **FILE SIZE (MB)**, not **SONG DURATION**  
**Required:** Pricing should be based on **SONG DURATION (seconds)**

### Current Pricing Model

- **Formula:** `Price = File Size (MB) × $1.00`
- **Minimum:** $0.99
- **Maximum:** $99.99
- **Location:** `/home/ubuntu/app/backend/src/utils/pricing.py`

### Payment Flow

1. **Generation Endpoint** (`/api/v1/generate`)
   - ❌ **NO payment required before generation**
   - User can generate songs without paying
   - Payment happens AFTER song is created

2. **Payment Endpoint** (`/api/v1/payments/create-intent`)
   - Used to purchase **already-generated** songs
   - Requires `song_id` (song must exist first)
   - Validates price against file size

3. **Webhook Handler** (`/api/webhooks/stripe`)
   - Handles payment success/failure
   - Grants download access after payment

## Detailed Findings

### 1. Pricing Logic (`src/utils/pricing.py`)

```python
PRICE_PER_MB_CENTS = 100  # $1.00 per MB
MIN_PRICE_CENTS = 99      # $0.99 minimum
MAX_PRICE_CENTS = 9999    # $99.99 maximum

def calculate_price_cents(file_size_mb: Optional[float]) -> int:
    """Calculate price based on FILE SIZE"""
    calculated_price = int(round(file_size_mb * PRICE_PER_MB_CENTS))
    # ... min/max validation
```

**Problem:** This calculates price from file size, not duration.

### 2. Generation Endpoint (`src/api/v1/generation.py`)

```python
@router.post("/generate")
async def generate_song(...):
    # NO payment check
    # NO price calculation
    # Directly creates generation job
```

**Problem:** Generation happens without payment verification.

### 3. Payment Endpoint (`src/api/v1/payments.py`)

```python
@router.post("/create-intent")
async def create_payment_intent(request: CreatePaymentIntentRequest):
    # Requires song_id (song must already exist)
    # Validates price against file_size_mb
```

**Problem:** Payment is for existing songs, not generation.

### 4. Stripe Configuration

**Status:** ❌ **NOT CONFIGURED**
- No Stripe keys in `.env` file
- Service logs show: "⚠️ Stripe not configured"
- Payment endpoints will fail

## Required Changes

### 1. Update Pricing to Use Duration

**File:** `src/utils/pricing.py`

```python
# NEW: Pricing based on duration
PRICE_PER_SECOND_CENTS = 10  # $0.10 per second
BASE_PRICE_95S_CENTS = 999  # $9.99 for 95-second base song
MIN_PRICE_CENTS = 999       # $9.99 minimum
MAX_PRICE_CENTS = 9999     # $99.99 maximum

def calculate_price_cents(duration_seconds: int) -> int:
    """
    Calculate price based on song duration
    
    Pricing:
    - 95 seconds (base): $9.99
    - 96-285 seconds: $9.99 + ($0.10 × (duration - 95))
    """
    if duration_seconds == 95:
        return BASE_PRICE_95S_CENTS
    
    if 96 <= duration_seconds <= 285:
        additional_seconds = duration_seconds - 95
        price = BASE_PRICE_95S_CENTS + (additional_seconds * PRICE_PER_SECOND_CENTS)
        return min(price, MAX_PRICE_CENTS)
    
    raise ValueError(f"Invalid duration: {duration_seconds}. Must be 95 or 96-285.")
```

### 2. Require Payment Before Generation

**File:** `src/api/v1/generation.py`

```python
@router.post("/generate")
async def generate_song(...):
    # NEW: Calculate price based on duration
    from ...utils.pricing import calculate_price_cents
    
    price_cents = calculate_price_cents(body.duration)
    
    # NEW: Require payment intent before generation
    # Option 1: Require payment_intent_id in request
    # Option 2: Create payment intent and return it, require confirmation
    
    # Verify payment before starting generation
    if not verify_payment_intent(payment_intent_id, price_cents):
        raise HTTPException(status_code=402, detail="Payment required")
    
    # Then proceed with generation
    ...
```

### 3. Add Duration-Based Price Endpoint

**New Endpoint:** `GET /api/v1/payments/calculate-price?duration=95`

```python
@router.get("/calculate-price")
async def calculate_price(duration: int):
    """Calculate price for a song based on duration"""
    price_cents = calculate_price_cents(duration)
    return {
        "duration_seconds": duration,
        "price_cents": price_cents,
        "price_dollars": price_cents / 100,
        "currency": "usd"
    }
```

### 4. Configure Stripe Keys

**File:** `/home/ubuntu/app/backend/.env`

```bash
STRIPE_SECRET_KEY=sk_live_...
STRIPE_PUBLISHABLE_KEY=pk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
```

## Recommended Pricing Model

Based on song duration:

| Duration | Price | Formula |
|----------|-------|---------|
| 95 seconds | $9.99 | Base price |
| 96-180 seconds | $9.99 + $0.10/sec | $9.99 + (duration - 95) × $0.10 |
| 181-285 seconds | $9.99 + $0.10/sec | $9.99 + (duration - 95) × $0.10 |
| Max | $99.99 | Capped at $99.99 |

**Examples:**
- 95s: $9.99
- 120s: $9.99 + (120-95) × $0.10 = $12.49
- 180s: $9.99 + (180-95) × $0.10 = $18.49
- 285s: $9.99 + (285-95) × $0.10 = $28.99

## Implementation Checklist

- [ ] Update `pricing.py` to calculate based on duration
- [ ] Add payment verification to generation endpoint
- [ ] Create price calculation endpoint
- [ ] Update payment intent creation to use duration
- [ ] Add Stripe keys to `.env`
- [ ] Test payment flow end-to-end
- [ ] Update frontend to show duration-based pricing
- [ ] Update webhook handler to verify duration-based pricing

## Current Issues Summary

1. ❌ **Pricing based on file size, not duration**
2. ❌ **No payment required before generation**
3. ❌ **Stripe keys not configured**
4. ❌ **Payment flow is post-generation, not pre-generation**

## Next Steps

1. **Immediate:** Update pricing logic to use duration
2. **Critical:** Require payment before generation
3. **Required:** Configure Stripe keys
4. **Testing:** Verify complete payment → generation flow

---

**Status:** ⚠️ **REQUIRES FIXES**  
**Priority:** 🔴 **HIGH** - Payment system not correctly implemented for duration-based pricing
