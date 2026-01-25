#!/usr/bin/env python3
"""
Test server configuration - verify authentication bypass and API format
"""
import requests
import json

BASE_URL = "http://127.0.0.1:8001"
API_BASE = f"{BASE_URL}/api/v1"

def test_generate_endpoint():
    """Test generate endpoint with server's API format"""
    print("Testing generate endpoint...")
    
    request_data = {
        "text_prompt": "A happy upbeat pop song about summer",
        "genre": "pop",
        "style": "upbeat",
        "duration": 95.0
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/generate",
            json=request_data,
            timeout=10
        )
        
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200 or response.status_code == 201:
            print("✅ Generate endpoint working!")
            return True
        elif response.status_code == 401:
            print("❌ Authentication still required")
            return False
        elif response.status_code == 402:
            print("⚠️  Payment required (expected)")
            return True
        else:
            print(f"⚠️  Unexpected status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_generate_endpoint()
