#!/usr/bin/env python3
"""
Webhook Delivery Test
Tests Stripe webhook delivery and processing
"""

import sys
import requests
import json
import time
import subprocess
from typing import Optional, Dict

# Configuration
BASE_URL = "http://127.0.0.1:8001"
WEBHOOK_URL = f"{BASE_URL}/api/webhooks/stripe"

def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_test(test_name: str, status: bool, details: str = ""):
    """Print test result"""
    icon = "✅" if status else "❌"
    print(f"\n{icon} {test_name}")
    if details:
        print(f"   {details}")

def test_webhook_endpoint_accessible() -> bool:
    """Test 1: Check if webhook endpoint is accessible"""
    print_section("Test 1: Webhook Endpoint Accessibility")
    
    print(f"Testing webhook endpoint: {WEBHOOK_URL}")
    
    try:
        # Test with GET (should return method not allowed)
        response = requests.get(WEBHOOK_URL, timeout=5)
        print(f"   GET response: {response.status_code}")
        
        # Test with POST (webhook endpoint)
        response = requests.post(
            WEBHOOK_URL,
            json={"type": "test"},
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        print(f"   POST response: {response.status_code}")
        
        if response.status_code in [400, 401, 403]:
            print_test("Webhook endpoint accessible", True, "Endpoint exists and validates requests")
            return True
        elif response.status_code == 200:
            print_test("Webhook endpoint accessible", True, "Endpoint accepts requests")
            return True
        else:
            print_test("Webhook endpoint unexpected response", False, f"Status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_test("Webhook endpoint not accessible", False, "Cannot connect to server")
        return False
    except Exception as e:
        print_test("Webhook endpoint test failed", False, str(e))
        return False

def test_stripe_cli_listener() -> Optional[str]:
    """Test 2: Start Stripe CLI listener and get signing secret"""
    print_section("Test 2: Stripe CLI Listener")
    
    print("Checking if Stripe CLI is available...")
    
    try:
        # Check if stripe CLI is installed
        result = subprocess.run(
            ["stripe", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            print_test("Stripe CLI not found", False, "Install from: https://stripe.com/docs/stripe-cli")
            return None
        
        print_test("Stripe CLI found", True, result.stdout.strip())
        
        print("\n⚠️  To test webhook delivery:")
        print("   1. In a separate terminal, run:")
        print(f"      stripe listen --forward-to {WEBHOOK_URL}")
        print("   2. Copy the webhook signing secret (whsec_...)")
        print("   3. Update server .env file with the secret")
        print("   4. Restart the service")
        print("   5. In another terminal, trigger events:")
        print("      stripe trigger payment_intent.succeeded")
        print("      stripe trigger payment_intent.payment_failed")
        
        return "manual_setup_required"
    except FileNotFoundError:
        print_test("Stripe CLI not installed", False, "Install from: https://stripe.com/docs/stripe-cli")
        return None
    except Exception as e:
        print_test("Stripe CLI check failed", False, str(e))
        return None

def test_trigger_webhook_event(event_type: str = "payment_intent.succeeded") -> bool:
    """Test 3: Trigger webhook event using Stripe CLI"""
    print_section(f"Test 3: Trigger Webhook Event ({event_type})")
    
    print(f"Triggering {event_type} event...")
    print("   (This requires Stripe CLI listener to be running)")
    
    try:
        result = subprocess.run(
            ["stripe", "trigger", event_type],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            output = result.stdout
            print_test("Webhook event triggered", True)
            print(f"   Output: {output[:200]}")
            
            # Try to extract payment intent ID
            if "pi_" in output:
                print("   Payment intent ID found in output")
            
            return True
        else:
            error = result.stderr or result.stdout
            print_test("Webhook event trigger failed", False, error[:200])
            print("\n   Troubleshooting:")
            print("   1. Ensure 'stripe login' has been run")
            print("   2. Check Stripe CLI listener is running")
            print("   3. Verify webhook endpoint is accessible")
            return False
    except FileNotFoundError:
        print_test("Stripe CLI not found", False, "Install Stripe CLI first")
        return False
    except Exception as e:
        print_test("Error triggering webhook", False, str(e))
        return False

def test_webhook_signature_verification() -> bool:
    """Test 4: Verify webhook signature validation"""
    print_section("Test 4: Webhook Signature Verification")
    
    print("Testing signature verification...")
    print("   (Sending request without valid signature)")
    
    try:
        # Send request without valid signature
        response = requests.post(
            WEBHOOK_URL,
            json={
                "type": "payment_intent.succeeded",
                "data": {
                    "object": {
                        "id": "pi_test_123",
                        "status": "succeeded"
                    }
                }
            },
            headers={
                "Content-Type": "application/json",
                "stripe-signature": "invalid_signature"
            },
            timeout=5
        )
        
        if response.status_code in [400, 401, 403]:
            print_test("Signature verification working", True, "Invalid signature correctly rejected")
            return True
        elif response.status_code == 200:
            print_test("Signature verification may be disabled", False, "Endpoint accepted invalid signature")
            return False
        else:
            print_test("Unexpected response", False, f"Status: {response.status_code}")
            return False
    except Exception as e:
        print_test("Signature verification test failed", False, str(e))
        return False

def check_stripe_dashboard_config() -> bool:
    """Test 5: Check if webhook is configured in Stripe Dashboard"""
    print_section("Test 5: Stripe Dashboard Configuration")
    
    print("⚠️  Manual verification required:")
    print("\n   1. Go to https://dashboard.stripe.com")
    print("   2. Navigate to Developers → Webhooks")
    print("   3. Verify endpoint exists: https://burntbeats.com/api/webhooks/stripe")
    print("   4. Check events are selected:")
    print("      ✅ payment_intent.succeeded")
    print("      ✅ payment_intent.payment_failed")
    print("      ✅ payment_intent.canceled")
    print("   5. Verify webhook signing secret matches server .env file")
    print("   6. Check endpoint status is 'Active'")
    
    response = input("\n   Is webhook configured in Stripe Dashboard? (y/n): ")
    return response.lower() == 'y'

def main():
    """Run webhook delivery tests"""
    print("\n" + "=" * 70)
    print("  WEBHOOK DELIVERY TEST")
    print("=" * 70)
    print("\nThis test verifies webhook delivery and processing:")
    print("  1. Endpoint accessibility")
    print("  2. Stripe CLI listener setup")
    print("  3. Event triggering")
    print("  4. Signature verification")
    print("  5. Dashboard configuration")
    
    results = {
        "endpoint_accessible": False,
        "stripe_cli_available": False,
        "event_trigger": False,
        "signature_verification": False,
        "dashboard_config": False
    }
    
    # Test 1: Endpoint accessibility
    results["endpoint_accessible"] = test_webhook_endpoint_accessible()
    
    # Test 2: Stripe CLI
    cli_result = test_stripe_cli_listener()
    results["stripe_cli_available"] = cli_result is not None
    
    # Test 3: Trigger event (if CLI available)
    if cli_result:
        print("\n⚠️  To test event triggering:")
        print("   1. Start listener: stripe listen --forward-to http://127.0.0.1:8001/api/webhooks/stripe")
        print("   2. Run this test again, or manually trigger: stripe trigger payment_intent.succeeded")
        
        # Ask user if they want to test now
        test_now = input("\n   Do you have Stripe CLI listener running? (y/n): ")
        if test_now.lower() == 'y':
            results["event_trigger"] = test_trigger_webhook_event()
        else:
            print("   Skipping event trigger test")
    
    # Test 4: Signature verification
    results["signature_verification"] = test_webhook_signature_verification()
    
    # Test 5: Dashboard configuration
    results["dashboard_config"] = check_stripe_dashboard_config()
    
    # Summary
    print_section("Test Summary")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name.replace('_', ' ').title()}")
    
    print(f"\n  Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n✅ All webhook tests passed!")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed or require manual setup")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
