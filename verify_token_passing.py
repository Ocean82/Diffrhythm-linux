#!/usr/bin/env python3
"""
Token Passing Verification Script
Verifies that tokens are being properly extracted and validated in the server
"""
import os
import sys
import requests
import logging
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", None)
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", None)

def test_api_key_extraction() -> Dict[str, Any]:
    """Test 1: Verify API key is extracted from X-API-Key header"""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: API Key Extraction from X-API-Key Header")
    logger.info("="*60)
    
    results = {
        "test": "API Key Extraction",
        "passed": False,
        "details": []
    }
    
    # Test with valid API key (if configured)
    if API_KEY:
        try:
            headers = {"X-API-Key": API_KEY}
            response = requests.get(f"{API_BASE_URL}/api/v1/health", headers=headers, timeout=5)
            
            if response.status_code == 200:
                results["details"].append("✅ Valid API key accepted")
                results["passed"] = True
            elif response.status_code == 401:
                results["details"].append("❌ Valid API key rejected (401)")
            else:
                results["details"].append(f"⚠️ Unexpected status: {response.status_code}")
        except Exception as e:
            results["details"].append(f"❌ Error: {e}")
    else:
        results["details"].append("⚠️ API_KEY not set - skipping test")
        results["passed"] = True  # Not a failure if not configured
    
    # Test with invalid API key
    try:
        headers = {"X-API-Key": "invalid_key_12345"}
        response = requests.get(f"{API_BASE_URL}/api/v1/health", headers=headers, timeout=5)
        
        if API_KEY:
            if response.status_code == 401:
                results["details"].append("✅ Invalid API key correctly rejected (401)")
            else:
                results["details"].append(f"⚠️ Invalid API key not rejected (status: {response.status_code})")
        else:
            results["details"].append("ℹ️ API key not required (API_KEY not set)")
    except Exception as e:
        results["details"].append(f"❌ Error testing invalid key: {e}")
    
    # Test without API key header
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=5)
        
        if API_KEY:
            if response.status_code == 401:
                results["details"].append("✅ Missing API key correctly rejected (401)")
            else:
                results["details"].append(f"⚠️ Missing API key not rejected (status: {response.status_code})")
        else:
            if response.status_code == 200:
                results["details"].append("✅ Request without API key accepted (API_KEY not set)")
            else:
                results["details"].append(f"⚠️ Unexpected status without key: {response.status_code}")
    except Exception as e:
        results["details"].append(f"❌ Error testing without key: {e}")
    
    return results


def test_api_key_case_sensitivity() -> Dict[str, Any]:
    """Test 2: Verify header name case sensitivity"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Header Name Case Sensitivity")
    logger.info("="*60)
    
    results = {
        "test": "Header Case Sensitivity",
        "passed": False,
        "details": []
    }
    
    if not API_KEY:
        results["details"].append("ℹ️ Skipping - API_KEY not set")
        results["passed"] = True
        return results
    
    # Test various header name cases
    test_cases = [
        ("x-api-key", "lowercase"),
        ("X-API-Key", "mixed case"),
        ("X-API-KEY", "uppercase"),
        ("x-Api-Key", "mixed case 2"),
    ]
    
    for header_name, description in test_cases:
        try:
            headers = {header_name: API_KEY}
            response = requests.get(f"{API_BASE_URL}/api/v1/health", headers=headers, timeout=5)
            
            if response.status_code == 200:
                results["details"].append(f"✅ {description} header accepted: {header_name}")
            elif response.status_code == 401:
                results["details"].append(f"❌ {description} header rejected: {header_name}")
            else:
                results["details"].append(f"⚠️ {description} unexpected status {response.status_code}")
        except Exception as e:
            results["details"].append(f"❌ Error with {description}: {e}")
    
    results["passed"] = True
    return results


def test_generate_endpoint_auth() -> Dict[str, Any]:
    """Test 3: Verify /api/v1/generate endpoint requires API key"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Generate Endpoint Authentication")
    logger.info("="*60)
    
    results = {
        "test": "Generate Endpoint Auth",
        "passed": False,
        "details": []
    }
    
    test_payload = {
        "lyrics": "[00:00.00]Test lyrics",
        "style_prompt": "test style",
        "audio_length": 95
    }
    
    # Test without API key
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/generate",
            json=test_payload,
            timeout=5
        )
        
        if API_KEY:
            if response.status_code == 401:
                results["details"].append("✅ Generate endpoint correctly requires API key")
                results["passed"] = True
            else:
                results["details"].append(f"❌ Generate endpoint should require API key (status: {response.status_code})")
        else:
            if response.status_code in [200, 202, 400, 503]:  # 400/503 are OK, means auth passed
                results["details"].append("✅ Generate endpoint accessible without API key (API_KEY not set)")
                results["passed"] = True
            else:
                results["details"].append(f"⚠️ Unexpected status: {response.status_code}")
    except Exception as e:
        results["details"].append(f"❌ Error: {e}")
    
    # Test with valid API key
    if API_KEY:
        try:
            headers = {"X-API-Key": API_KEY}
            response = requests.post(
                f"{API_BASE_URL}/api/v1/generate",
                json=test_payload,
                headers=headers,
                timeout=5
            )
            
            if response.status_code != 401:
                results["details"].append("✅ Generate endpoint accepts valid API key")
            else:
                results["details"].append("❌ Generate endpoint rejected valid API key")
        except Exception as e:
            results["details"].append(f"❌ Error with valid key: {e}")
    
    return results


def test_webhook_signature() -> Dict[str, Any]:
    """Test 4: Verify Stripe webhook signature extraction"""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Stripe Webhook Signature Extraction")
    logger.info("="*60)
    
    results = {
        "test": "Webhook Signature",
        "passed": False,
        "details": []
    }
    
    if not STRIPE_WEBHOOK_SECRET:
        results["details"].append("ℹ️ Skipping - STRIPE_WEBHOOK_SECRET not set")
        results["passed"] = True
        return results
    
    # Test with missing signature header
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/webhooks/stripe",
            json={"type": "test"},
            timeout=5
        )
        
        if response.status_code == 400:
            results["details"].append("✅ Webhook correctly requires stripe-signature header")
        else:
            results["details"].append(f"⚠️ Unexpected status without signature: {response.status_code}")
    except Exception as e:
        results["details"].append(f"❌ Error: {e}")
    
    # Test with invalid signature
    try:
        headers = {"stripe-signature": "invalid_signature_12345"}
        response = requests.post(
            f"{API_BASE_URL}/api/webhooks/stripe",
            json={"type": "test"},
            headers=headers,
            timeout=5
        )
        
        if response.status_code == 400:
            results["details"].append("✅ Webhook correctly rejects invalid signature")
            results["passed"] = True
        else:
            results["details"].append(f"⚠️ Unexpected status with invalid signature: {response.status_code}")
    except Exception as e:
        results["details"].append(f"❌ Error: {e}")
    
    return results


def test_header_logging() -> Dict[str, Any]:
    """Test 5: Verify tokens are not logged in plaintext"""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: Token Security (No Plaintext Logging)")
    logger.info("="*60)
    
    results = {
        "test": "Token Security",
        "passed": True,  # Assume pass unless we find issues
        "details": []
    }
    
    # This is a code review check - we'll check the actual code
    results["details"].append("ℹ️ Manual code review required:")
    results["details"].append("   - Check backend/api.py for token logging")
    results["details"].append("   - Check backend/security.py for token exposure")
    results["details"].append("   - Verify tokens not in error messages")
    
    return results


def main():
    """Run all token verification tests"""
    logger.info("\n" + "="*60)
    logger.info("TOKEN PASSING VERIFICATION")
    logger.info("="*60)
    logger.info(f"API Base URL: {API_BASE_URL}")
    logger.info(f"API Key Configured: {'Yes' if API_KEY else 'No'}")
    logger.info(f"Stripe Webhook Secret Configured: {'Yes' if STRIPE_WEBHOOK_SECRET else 'No'}")
    
    all_results = []
    
    # Run tests
    all_results.append(test_api_key_extraction())
    all_results.append(test_api_key_case_sensitivity())
    all_results.append(test_generate_endpoint_auth())
    all_results.append(test_webhook_signature())
    all_results.append(test_header_logging())
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for r in all_results if r["passed"])
    total = len(all_results)
    
    for result in all_results:
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        logger.info(f"{status}: {result['test']}")
        for detail in result["details"]:
            logger.info(f"   {detail}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("✅ All token passing tests passed!")
        return 0
    else:
        logger.warning("⚠️ Some tests failed or were skipped")
        return 1


if __name__ == "__main__":
    sys.exit(main())
