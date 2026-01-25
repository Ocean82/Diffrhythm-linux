#!/usr/bin/env python3
"""Test payment schema"""
import sys
sys.path.insert(0, '/home/ubuntu/app/backend')

try:
    from src.schemas.payments import CreatePaymentIntentRequest, CalculatePriceRequest
    print("✅ Schema import successful")
    
    # Test with duration
    req1 = CreatePaymentIntentRequest(
        duration_seconds=120,
        amount_cents=200,
        is_extended=False
    )
    print(f"✅ Request with duration: {req1.duration_seconds}s, ${req1.amount_cents/100:.2f}")
    
    # Test with extended
    req2 = CreatePaymentIntentRequest(
        duration_seconds=180,
        amount_cents=350,
        is_extended=True
    )
    print(f"✅ Request extended: {req2.duration_seconds}s, ${req2.amount_cents/100:.2f}")
    
    # Test with commercial license
    req3 = CreatePaymentIntentRequest(
        duration_seconds=120,
        amount_cents=1700,
        commercial_license=True
    )
    print(f"✅ Request with commercial: ${req3.amount_cents/100:.2f}")
    
    # Test bulk pack
    req4 = CreatePaymentIntentRequest(
        duration_seconds=120,
        amount_cents=1800,
        bulk_pack_size=10
    )
    print(f"✅ Request bulk 10: ${req4.amount_cents/100:.2f}")
    
    print("\n✅ All schema tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
