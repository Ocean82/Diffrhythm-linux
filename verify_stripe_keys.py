#!/usr/bin/env python3
"""
Verify Stripe Keys Configuration
Checks if Stripe keys are properly formatted and configured
"""

import re
import sys
from pathlib import Path

def verify_stripe_key_format(key: str, key_type: str) -> tuple[bool, str]:
    """Verify Stripe key format"""
    if not key or key.strip() == "":
        return False, f"{key_type} is empty"
    
    key = key.strip()
    
    if key_type == "SECRET_KEY":
        if not re.match(r'^sk_(test|live)_[a-zA-Z0-9]{24,}$', key):
            return False, f"Invalid secret key format. Must start with sk_test_ or sk_live_"
        return True, "Valid secret key format"
    
    elif key_type == "PUBLISHABLE_KEY":
        if not re.match(r'^pk_(test|live)_[a-zA-Z0-9]{24,}$', key):
            return False, f"Invalid publishable key format. Must start with pk_test_ or pk_live_"
        return True, "Valid publishable key format"
    
    elif key_type == "WEBHOOK_SECRET":
        if not re.match(r'^whsec_[a-zA-Z0-9]{32,}$', key):
            return False, f"Invalid webhook secret format. Must start with whsec_"
        return True, "Valid webhook secret format"
    
    return False, "Unknown key type"

def check_env_file(file_path: str) -> dict:
    """Check .env file for Stripe configuration"""
    results = {
        "file_exists": False,
        "stripe_secret_key": None,
        "stripe_publishable_key": None,
        "stripe_webhook_secret": None,
        "require_payment": None,
        "errors": [],
        "warnings": [],
        "valid": False
    }
    
    env_file = Path(file_path)
    
    if not env_file.exists():
        results["errors"].append(f"File not found: {file_path}")
        return results
    
    results["file_exists"] = True
    
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse key-value pairs
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                
                if key == "STRIPE_SECRET_KEY":
                    results["stripe_secret_key"] = value
                    is_valid, message = verify_stripe_key_format(value, "SECRET_KEY")
                    if not is_valid:
                        results["errors"].append(f"STRIPE_SECRET_KEY: {message}")
                    else:
                        # Check if it's a placeholder
                        if value.endswith('...') or 'YOUR' in value.upper() or 'PLACEHOLDER' in value.upper():
                            results["warnings"].append("STRIPE_SECRET_KEY appears to be a placeholder")
                        else:
                            results["warnings"].append(f"STRIPE_SECRET_KEY: {message} (Mode: {'Test' if 'test' in value else 'Live'})")
                
                elif key == "STRIPE_PUBLISHABLE_KEY":
                    results["stripe_publishable_key"] = value
                    is_valid, message = verify_stripe_key_format(value, "PUBLISHABLE_KEY")
                    if not is_valid:
                        results["errors"].append(f"STRIPE_PUBLISHABLE_KEY: {message}")
                    else:
                        if value.endswith('...') or 'YOUR' in value.upper() or 'PLACEHOLDER' in value.upper():
                            results["warnings"].append("STRIPE_PUBLISHABLE_KEY appears to be a placeholder")
                        else:
                            results["warnings"].append(f"STRIPE_PUBLISHABLE_KEY: {message} (Mode: {'Test' if 'test' in value else 'Live'})")
                
                elif key == "STRIPE_WEBHOOK_SECRET":
                    results["stripe_webhook_secret"] = value
                    is_valid, message = verify_stripe_key_format(value, "WEBHOOK_SECRET")
                    if not is_valid:
                        results["errors"].append(f"STRIPE_WEBHOOK_SECRET: {message}")
                    else:
                        if value.endswith('...') or 'YOUR' in value.upper() or 'PLACEHOLDER' in value.upper():
                            results["warnings"].append("STRIPE_WEBHOOK_SECRET appears to be a placeholder")
                        else:
                            results["warnings"].append(f"STRIPE_WEBHOOK_SECRET: {message}")
                
                elif key == "REQUIRE_PAYMENT_FOR_GENERATION":
                    results["require_payment"] = value.lower() in ['true', '1', 'yes']
        
        # Check if all keys are present
        if not results["stripe_secret_key"]:
            results["errors"].append("STRIPE_SECRET_KEY not found in .env file")
        if not results["stripe_publishable_key"]:
            results["errors"].append("STRIPE_PUBLISHABLE_KEY not found in .env file")
        if not results["stripe_webhook_secret"]:
            results["errors"].append("STRIPE_WEBHOOK_SECRET not found in .env file")
        
        # Determine if configuration is valid
        results["valid"] = (
            results["stripe_secret_key"] and
            results["stripe_publishable_key"] and
            results["stripe_webhook_secret"] and
            len(results["errors"]) == 0
        )
        
    except Exception as e:
        results["errors"].append(f"Error reading file: {str(e)}")
    
    return results

def main():
    """Main function"""
    if len(sys.argv) > 1:
        env_file = sys.argv[1]
    else:
        env_file = r"C:\Users\sammy\OneDrive\Desktop\.env"
    
    print("=" * 60)
    print("Stripe Keys Configuration Verification")
    print("=" * 60)
    print(f"\nChecking file: {env_file}\n")
    
    results = check_env_file(env_file)
    
    if not results["file_exists"]:
        print("❌ File not found!")
        sys.exit(1)
    
    print("Configuration Status:")
    print("-" * 60)
    
    # Secret Key
    if results["stripe_secret_key"]:
        print(f"[OK] STRIPE_SECRET_KEY: Found")
        print(f"   Value: {results['stripe_secret_key'][:20]}...")
    else:
        print("[ERROR] STRIPE_SECRET_KEY: Not found")
    
    # Publishable Key
    if results["stripe_publishable_key"]:
        print(f"[OK] STRIPE_PUBLISHABLE_KEY: Found")
        print(f"   Value: {results['stripe_publishable_key'][:20]}...")
    else:
        print("[ERROR] STRIPE_PUBLISHABLE_KEY: Not found")
    
    # Webhook Secret
    if results["stripe_webhook_secret"]:
        print(f"[OK] STRIPE_WEBHOOK_SECRET: Found")
        print(f"   Value: {results['stripe_webhook_secret'][:20]}...")
    else:
        print("[ERROR] STRIPE_WEBHOOK_SECRET: Not found")
    
    # Require Payment
    if results["require_payment"] is not None:
        print(f"[OK] REQUIRE_PAYMENT_FOR_GENERATION: {results['require_payment']}")
    else:
        print("[WARN] REQUIRE_PAYMENT_FOR_GENERATION: Not set (defaults to false)")
    
    # Errors
    if results["errors"]:
        print("\n[ERROR] Errors:")
        for error in results["errors"]:
            print(f"   - {error}")
    
    # Warnings
    if results["warnings"]:
        print("\n[WARN] Warnings:")
        for warning in results["warnings"]:
            print(f"   - {warning}")
    
    # Summary
    print("\n" + "=" * 60)
    if results["valid"]:
        print("[OK] Configuration is VALID")
        print("\nNext steps:")
        print("1. Copy keys to server: /home/ubuntu/app/backend/.env")
        print("2. Restart service: sudo systemctl restart burntbeats-api")
        print("3. Verify: curl http://127.0.0.1:8001/api/v1/payments/calculate-price?duration=120")
    else:
        print("❌ Configuration has ERRORS")
        print("\nPlease fix the errors above before proceeding.")
    
    print("=" * 60)
    
    sys.exit(0 if results["valid"] else 1)

if __name__ == "__main__":
    main()
