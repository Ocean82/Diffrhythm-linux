# Stripe Payment Implementation Guide

**Date:** January 23, 2026  
**Purpose:** Ensure Stripe payment system correctly charges users based on song duration

## Expected Implementation

### Pricing Model

Based on song duration:
- **95 seconds (base song)**: $X.XX
- **96-180 seconds (medium)**: $Y.YY  
- **181-285 seconds (extended)**: $Z.ZZ

Or tiered pricing:
- **95 seconds**: Base price
- **Per additional second**: $0.XX per second after 95

### Integration Points

1. **Generation Endpoint** (`/api/v1/generate`)
   - Should calculate price before generation
   - Should create Stripe payment intent
   - Should verify payment before starting generation

2. **Payment Endpoint** (`/api/v1/payments/checkout`)
   - Should accept song duration
   - Should calculate price
   - Should create Stripe checkout session

3. **Webhook Handler** (`/api/v1/payments/webhook`)
   - Should verify Stripe webhook signature
   - Should handle payment success/failure
   - Should update job status based on payment

## Required Configuration

### Environment Variables

```bash
STRIPE_SECRET_KEY=sk_live_...
STRIPE_PUBLISHABLE_KEY=pk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
```

### Pricing Configuration

```python
# Example pricing structure
PRICING = {
    "base_95s": 9.99,  # USD for 95-second songs
    "per_second": 0.10,  # USD per second after 95
    "max_duration": 285,  # Maximum duration
}
```

## Implementation Checklist

- [ ] Stripe keys configured in `.env`
- [ ] Stripe package installed (`pip install stripe`)
- [ ] Payment calculation function implemented
- [ ] Payment intent created before generation
- [ ] Webhook handler configured
- [ ] Webhook endpoint accessible from Stripe
- [ ] Payment verification in generation endpoint
- [ ] Error handling for failed payments
- [ ] Logging for payment events

## Testing

1. **Test Payment Calculation**
   ```python
   def calculate_price(duration: int) -> float:
       if duration == 95:
           return 9.99
       elif 96 <= duration <= 285:
           return 9.99 + (duration - 95) * 0.10
       else:
           raise ValueError("Invalid duration")
   ```

2. **Test Stripe Integration**
   - Create test payment intent
   - Verify webhook receives events
   - Test payment success flow
   - Test payment failure flow

3. **Test End-to-End**
   - User requests generation
   - System calculates price
   - User pays via Stripe
   - Webhook confirms payment
   - Generation starts
   - User receives song

## Common Issues

1. **Stripe not configured**
   - Check `.env` file for keys
   - Verify keys are valid
   - Check service logs

2. **Webhook not receiving events**
   - Verify webhook URL is correct
   - Check webhook secret matches
   - Ensure endpoint is accessible

3. **Payment not verified before generation**
   - Check generation endpoint verifies payment
   - Ensure payment intent is confirmed
   - Verify webhook updates job status

## Next Steps

1. Run investigation script on server
2. Review payment code implementation
3. Verify pricing logic matches requirements
4. Test payment flow
5. Fix any issues found
6. Document final implementation
