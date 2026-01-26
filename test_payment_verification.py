#!/usr/bin/env python3
"""
Test payment verification on server
"""
import requests
import json

def test_payment_verification():
    """Test payment verification via API"""
    api_url = 'http://52.0.207.242:8000/api/v1/generate'
    
    # Test 1: Without payment (should work since REQUIRE_PAYMENT_FOR_GENERATION=false)
    print("=== Test 1: Generation without payment ===")
    test_data_1 = {
        'lyrics': '[00:00.00]Test song lyrics for verification',
        'style_prompt': 'pop song',
        'audio_length': 95,  # Minimum required length
        'preset': 'preview'  # Fast preset for testing
    }
    
    try:
        response = requests.post(
            api_url,
            json=test_data_1,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"[SUCCESS] Job ID: {result.get('job_id')}")
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"[ERROR] {e}")
    
    # Test 2: With invalid payment intent (should fail gracefully)
    print("\n=== Test 2: Generation with invalid payment intent ===")
    test_data_2 = {
        'lyrics': '[00:00.00]Test song lyrics',
        'style_prompt': 'pop song',
        'audio_length': 95,  # Minimum required length
        'payment_intent_id': 'pi_test_invalid_123'
    }
    
    try:
        response = requests.post(
            api_url,
            json=test_data_2,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        if response.status_code == 400:
            print("[SUCCESS] Correctly rejected invalid payment intent")
    except Exception as e:
        print(f"[ERROR] {e}")
    
    # Test 3: Health check
    print("\n=== Test 3: Health check ===")
    try:
        response = requests.get('http://52.0.207.242:8000/api/v1/health', timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    test_payment_verification()
