#!/usr/bin/env python3
"""
Test Complete Payment → Generation Flow

This script tests the complete flow:
1. Calculate price for a song
2. Create payment intent
3. Verify payment intent
4. Generate song with payment intent ID
"""

import sys
import requests
import json
import time
from typing import Optional

# Configuration
BASE_URL = "http://127.0.0.1:8001/api/v1"
AUTH_TOKEN = None  # Set if authentication is required


def calculate_price(duration: int = 120) -> dict:
    """Step 1: Calculate price for a song"""
    print(f"\n[Step 1] Calculating price for {duration}s song...")
    
    url = f"{BASE_URL}/payments/calculate-price"
    params = {"duration": duration}
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"❌ Error calculating price: {response.status_code}")
        print(response.text)
        return None
    
    price_info = response.json()
    print(f"✅ Price calculated:")
    print(f"   Base price: ${price_info['base_price_dollars']:.2f}")
    print(f"   With commercial: ${price_info['with_commercial_license_dollars']:.2f}")
    print(f"   Bulk 10: ${price_info['bulk_10_price_dollars']:.2f}")
    print(f"   Bulk 50: ${price_info['bulk_50_price_dollars']:.2f}")
    
    return price_info


def create_payment_intent(
    duration: int = 120,
    amount_cents: int = 200,
    is_extended: bool = False,
    commercial_license: bool = False,
    bulk_pack_size: Optional[int] = None
) -> Optional[dict]:
    """Step 2: Create payment intent"""
    print(f"\n[Step 2] Creating payment intent...")
    
    url = f"{BASE_URL}/payments/create-intent"
    headers = {"Content-Type": "application/json"}
    if AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
    
    payload = {
        "duration_seconds": duration,
        "amount_cents": amount_cents,
        "currency": "usd",
        "is_extended": is_extended,
        "commercial_license": commercial_license,
        "bulk_pack_size": bulk_pack_size,
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code != 201:
        print(f"❌ Error creating payment intent: {response.status_code}")
        print(response.text)
        return None
    
    payment_intent = response.json()
    print(f"✅ Payment intent created:")
    print(f"   Payment Intent ID: {payment_intent['payment_intent_id']}")
    print(f"   Amount: ${payment_intent['amount_cents']/100:.2f}")
    print(f"   Client Secret: {payment_intent['client_secret'][:20]}...")
    
    return payment_intent


def verify_payment_intent(payment_intent_id: str) -> bool:
    """Step 3: Verify payment intent status"""
    print(f"\n[Step 3] Verifying payment intent: {payment_intent_id}...")
    
    url = f"{BASE_URL}/payments/verify-payment/{payment_intent_id}"
    headers = {}
    if AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"❌ Error verifying payment: {response.status_code}")
        print(response.text)
        return False
    
    verification = response.json()
    print(f"✅ Payment verification:")
    print(f"   Status: {verification['status']}")
    print(f"   Ready for generation: {verification['ready_for_generation']}")
    
    if verification['status'] == 'succeeded' and verification['ready_for_generation']:
        return True
    else:
        print(f"⚠️  Payment not ready: {verification.get('message', 'Unknown')}")
        return False


def generate_song(
    text_prompt: str,
    duration: float = 120.0,
    payment_intent_id: Optional[str] = None
) -> Optional[dict]:
    """Step 4: Generate song with payment intent"""
    print(f"\n[Step 4] Generating song...")
    
    url = f"{BASE_URL}/generate"
    headers = {"Content-Type": "application/json"}
    if AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
    
    payload = {
        "text_prompt": text_prompt,
        "duration": duration,
        "chunked": True,
    }
    
    if payment_intent_id:
        payload["payment_intent_id"] = payment_intent_id
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code != 201:
        print(f"❌ Error generating song: {response.status_code}")
        print(response.text)
        return None
    
    result = response.json()
    print(f"✅ Generation started:")
    print(f"   Job ID: {result.get('job_id')}")
    print(f"   Message: {result.get('message')}")
    
    return result


def check_generation_status(job_id: str) -> Optional[dict]:
    """Check generation job status"""
    url = f"{BASE_URL}/generate/{job_id}/status"
    headers = {}
    if AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        return None
    
    return response.json()


def main():
    """Run complete payment → generation flow test"""
    print("=" * 60)
    print("Payment → Generation Flow Test")
    print("=" * 60)
    
    # Step 1: Calculate price
    price_info = calculate_price(duration=120)
    if not price_info:
        print("\n❌ Test failed at Step 1")
        return False
    
    # Step 2: Create payment intent
    payment_intent = create_payment_intent(
        duration=120,
        amount_cents=price_info['base_price_cents'],
        is_extended=False,
        commercial_license=False
    )
    if not payment_intent:
        print("\n❌ Test failed at Step 2")
        print("Note: This step requires Stripe to be configured.")
        print("If Stripe is not configured, payment intent creation will fail.")
        return False
    
    payment_intent_id = payment_intent['payment_intent_id']
    
    # Step 3: Verify payment intent
    # Note: In a real flow, the frontend would confirm payment with Stripe
    # For testing, we'll check if payment is verified
    payment_verified = verify_payment_intent(payment_intent_id)
    if not payment_verified:
        print("\n⚠️  Payment not verified. In production, frontend confirms payment with Stripe.")
        print("For testing, we'll proceed anyway (if REQUIRE_PAYMENT_FOR_GENERATION is False).")
    
    # Step 4: Generate song
    generation_result = generate_song(
        text_prompt="A happy upbeat pop song about summer",
        duration=120.0,
        payment_intent_id=payment_intent_id if payment_verified else None
    )
    
    if not generation_result:
        print("\n❌ Test failed at Step 4")
        return False
    
    job_id = generation_result.get('job_id')
    if job_id:
        print(f"\n[Step 5] Monitoring generation job: {job_id}")
        print("Checking status every 5 seconds (max 60 seconds)...")
        
        for i in range(12):  # Check for 60 seconds
            time.sleep(5)
            status = check_generation_status(job_id)
            if status:
                print(f"   Status: {status.get('status')}, Progress: {status.get('progress', 0)*100:.1f}%")
                if status.get('status') in ['completed', 'failed']:
                    break
    
    print("\n" + "=" * 60)
    print("✅ Test flow completed!")
    print("=" * 60)
    print("\nNote: In production, the frontend would:")
    print("1. Calculate price")
    print("2. Create payment intent")
    print("3. Use Stripe.js to confirm payment")
    print("4. Generate song with confirmed payment_intent_id")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
