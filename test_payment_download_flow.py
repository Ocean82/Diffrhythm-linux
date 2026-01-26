"""
Test script for payment-before-download flow
"""
import requests
import time
import json

BASE_URL = "http://52.0.207.242:8000"

def test_payment_download_flow():
    """Test the new payment-before-download flow"""
    
    print("=" * 80)
    print("Payment-Before-Download Flow Test")
    print("=" * 80)
    print()
    
    # Test 1: Generate song without payment (should succeed)
    print("Test 1: Generate song without payment")
    print("-" * 80)
    
    generate_data = {
        "lyrics": "[00:00.00]Test song\n[00:05.00]This is a test",
        "style_prompt": "upbeat pop",
        "audio_length": 95,
        "preset": "preview"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/generate",
        json=generate_data,
        timeout=10
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        job_id = result["job_id"]
        print(f"[SUCCESS] Job created: {job_id}")
        print(f"Response: {json.dumps(result, indent=2)}")
    else:
        print(f"[ERROR] Generation failed: {response.text}")
        return
    
    print()
    
    # Test 2: Check job status
    print("Test 2: Check job status")
    print("-" * 80)
    
    response = requests.get(f"{BASE_URL}/api/v1/status/{job_id}", timeout=10)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        status = response.json()
        print(f"Job Status: {status['status']}")
        print(f"Response: {json.dumps(status, indent=2)}")
    else:
        print(f"[ERROR] Status check failed: {response.text}")
    
    print()
    
    # Test 3: Try to download without payment (should fail if payment required)
    print("Test 3: Try to download without payment")
    print("-" * 80)
    
    # Wait a bit for job to complete (or check status first)
    print("Note: This test assumes job is completed. If not, wait for completion first.")
    print()
    
    response = requests.get(
        f"{BASE_URL}/api/v1/download/{job_id}",
        timeout=10,
        allow_redirects=False
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 402:
        print("[SUCCESS] Download correctly requires payment")
        print(f"Response: {response.text}")
    elif response.status_code == 200:
        print("[INFO] Download succeeded (payment may not be required)")
    elif response.status_code == 400:
        print(f"[INFO] Job not completed yet: {response.text}")
    else:
        print(f"[INFO] Response: {response.text}")
    
    print()
    
    # Test 4: Health check
    print("Test 4: Health check")
    print("-" * 80)
    
    response = requests.get(f"{BASE_URL}/api/v1/health", timeout=10)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        health = response.json()
        print(f"Response: {json.dumps(health, indent=2)}")
    
    print()
    print("=" * 80)
    print("Test Complete")
    print("=" * 80)

if __name__ == "__main__":
    test_payment_download_flow()
