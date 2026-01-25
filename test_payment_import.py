#!/usr/bin/env python3
"""Test payment endpoint import"""
import sys
sys.path.insert(0, '/home/ubuntu/app/backend')

try:
    print("Testing imports...")
    from src.api.v1.payments import router
    print(f"✅ Router imported: {router}")
    
    print("\nRouter routes:")
    for route in router.routes:
        if hasattr(route, 'path'):
            print(f"  - {route.methods if hasattr(route, 'methods') else 'N/A'}: {route.path}")
        else:
            print(f"  - Route: {route}")
    
    print("\nTesting calculate_price function...")
    from src.api.v1.payments import calculate_price
    print(f"✅ calculate_price function: {calculate_price}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
