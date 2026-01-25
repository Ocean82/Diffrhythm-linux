#!/usr/bin/env python3
"""Test pricing functions"""
import sys
sys.path.insert(0, '/home/ubuntu/app/backend')

from src.utils.pricing import calculate_price_cents, calculate_price_for_duration

print("Testing pricing functions:")
print(f"120s single: ${calculate_price_cents(120)/100:.2f}")
print(f"180s extended: ${calculate_price_cents(180, is_extended=True)/100:.2f}")
print(f"120s with commercial: ${calculate_price_cents(120, commercial_license=True)/100:.2f}")
print(f"Bulk 10: ${calculate_price_cents(120, bulk_pack_size=10)/100:.2f}")
print(f"Bulk 50: ${calculate_price_cents(120, bulk_pack_size=50)/100:.2f}")

print("\nPrice calculation for 120s:")
result = calculate_price_for_duration(120)
print(f"Base: ${result['base_price_dollars']:.2f}")
print(f"With commercial: ${result['with_commercial_license_dollars']:.2f}")
