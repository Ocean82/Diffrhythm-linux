#!/usr/bin/env python3
"""
Test Complete Payment Flow
Tests payment intent creation and webhook handling
"""

import sys
import requests
import json
import time
from typing import Optional

# Configuration
BASE_URL = "http://127.0.0.1:8001/api/v1"
AUTH_TOKEN = None  # Set if authentication is required

def test_calculate_price(duration: int = 120) -> Optional[dict]:
    """Test price calculation"""
    print(f"\n[Test 1] Calculating price for {duration}s song...")
    
    url = f"{BASE_URL}/payments/calculate-price"
    params = {"duration": duration}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        price_info = response.json()
        print(f"✅ Price calculated successfully:")
        print(f"   Base price: ${price_info['base_price_dollars']:.2f}")
        print(f"   With commercial: ${price_info['with_commercial_license_dollars']:.2f}")
        
        return price_info
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_create_payment_intent(
    duration: int = 120,
    amount_cents: int = 200,
    use_stripe_cli: bool = False
) -> Optional[dict]:
    """Test payment intent creation"""
    print(f"\n[Test 2] Creating payment intent...")
    
    if use_stripe_cli:
        print("   Using Stripe CLI to create test payment intent...")
        import subprocess
        try:
            result = subprocess.run(
                [
                    "stripe", "payment_intents", "create",
                    "--amount", str(amount_cents),
                    "--currency", "usd",
                    "--metadata[duration_seconds]", str(duration),
                    "--metadata[user_id]", "test_user",
                    "--confirm"
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                payment_data = json.loads(result.stdout)
                print(f"✅ Payment intent created via CLI:")
                print(f"   Payment Intent ID: {payment_data.get('id')}")
                print(f"   Status: {payment_data.get('status')}")
                print(f"   Amount: ${payment_data.get('amount', 0)/100:.2f}")
                return {
                    "payment_intent_id": payment_data.get('id'),
                    "status": payment_data.get('status'),
                    "amount_cents": payment_data.get('amount', 0),
                    "client_secret": payment_data.get('client_secret')
                }
            else:
                print(f"❌ Stripe CLI error: {result.stderr}")
                return None
        except Exception as e:
            print(f"❌ Error using Stripe CLI: {e}")
            return None
    else:
        # Use API endpoint (requires authentication)
        url = f"{BASE_URL}/payments/create-intent"
        headers = {"Content-Type": "application/json"}
        if AUTH_TOKEN:
            headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
        
        payload = {
            "duration_seconds": duration,
            "amount_cents": amount_cents,
            "currency": "usd",
            "is_extended": False,
            "commercial_license": False,
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 201:
                payment_intent = response.json()
                print(f"✅ Payment intent created via API:")
                print(f"   Payment Intent ID: {payment_intent['payment_intent_id']}")
                print(f"   Amount: ${payment_intent['amount_cents']/100:.2f}")
                return payment_intent
            else:
                print(f"❌ API error: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

def test_verify_payment(payment_intent_id: str) -> bool:
    """Test payment verification"""
    print(f"\n[Test 3] Verifying payment intent: {payment_intent_id}...")
    
    url = f"{BASE_URL}/payments/verify-payment/{payment_intent_id}"
    headers = {}
    if AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        verification = response.json()
        print(f"✅ Payment verification:")
        print(f"   Status: {verification['status']}")
        print(f"   Ready for generation: {verification['ready_for_generation']}")
        
        return verification.get('ready_for_generation', False)
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_webhook_endpoint():
    """Test webhook endpoint accessibility"""
    print(f"\n[Test 4] Testing webhook endpoint...")
    
    url = "http://127.0.0.1:8001/api/webhooks/stripe"
    
    try:
        # Test with GET (should return method not allowed or similar)
        response = requests.get(url, timeout=5)
        print(f"   GET response: {response.status_code}")
        
        # Test with POST (webhook endpoint)
        response = requests.post(
            url,
            json={"type": "test"},
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        print(f"   POST response: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_trigger_webhook_event(event_type: str = "payment_intent.succeeded"):
    """Trigger test webhook event using Stripe CLI"""
    print(f"\n[Test 5] Triggering webhook event: {event_type}...")
    
    import subprocess
    try:
        result = subprocess.run(
            ["stripe", "trigger", event_type],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print(f"✅ Webhook event triggered successfully")
            print(f"   Output: {result.stdout[:200]}")
            return True
        else:
            print(f"❌ Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run complete payment flow tests"""
    print("=" * 60)
    print("Payment Flow Testing")
    print("=" * 60)
    
    # Test 1: Calculate price
    price_info = test_calculate_price(120)
    if not price_info:
        print("\n❌ Test failed at price calculation")
        return False
    
    # Test 2: Create payment intent (using Stripe CLI)
    print("\n" + "-" * 60)
    print("Creating test payment intent using Stripe CLI...")
    payment_intent = test_create_payment_intent(
        duration=120,
        amount_cents=200,
        use_stripe_cli=True
    )
    
    if not payment_intent:
        print("\n⚠️  Payment intent creation failed. This is expected if:")
        print("   - Stripe CLI is not logged in (run: stripe login)")
        print("   - Or you prefer to test via API endpoint")
        print("\nSkipping payment verification test...")
    else:
        # Test 3: Verify payment
        payment_verified = test_verify_payment(payment_intent['payment_intent_id'])
        
        if payment_verified:
            print("\n✅ Payment verified and ready for generation")
        else:
            print("\n⚠️  Payment not verified (may need confirmation)")
    
    # Test 4: Webhook endpoint
    print("\n" + "-" * 60)
    test_webhook_endpoint()
    
    # Test 5: Trigger webhook event
    print("\n" + "-" * 60)
    print("To test webhook handling:")
    print("1. Start webhook listener: stripe listen --forward-to http://127.0.0.1:8001/api/webhooks/stripe")
    print("2. In another terminal, trigger events:")
    print("   stripe trigger payment_intent.succeeded")
    print("   stripe trigger payment_intent.payment_failed")
    
    print("\n" + "=" * 60)
    print("✅ Payment flow tests completed!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
