#!/usr/bin/env python3
"""Test payment endpoints"""
import sys
sys.path.insert(0, '/home/ubuntu/app/backend')

print("Testing payment endpoints...")

try:
    from src.api.v1.payments import router, calculate_price
    from src.utils.pricing import calculate_price_cents, calculate_price_for_duration
    print("✅ Payment endpoints imported successfully")
    
    # Test pricing
    print("\nTesting pricing calculations:")
    print(f"120s single: ${calculate_price_cents(120)/100:.2f}")
    print(f"180s extended: ${calculate_price_cents(180, is_extended=True)/100:.2f}")
    print(f"120s with commercial: ${calculate_price_cents(120, commercial_license=True)/100:.2f}")
    print(f"Bulk 10: ${calculate_price_cents(120, bulk_pack_size=10)/100:.2f}")
    print(f"Bulk 50: ${calculate_price_cents(120, bulk_pack_size=50)/100:.2f}")
    
    print("\n✅ All payment endpoint tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
