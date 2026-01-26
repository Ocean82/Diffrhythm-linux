# Webhook Route Fix
**Date**: January 26, 2026  
**Issue**: Webhook route exists in code but not registered with FastAPI

## Problem

- **Route Definition**: Exists at line 716 in `/opt/diffrhythm/backend/api.py`
- **Route Registration**: Not found in FastAPI app routes (0 webhook routes)
- **Symptom**: Returns 404 Not Found
- **Impact**: Stripe webhooks cannot be received

## Investigation

### Code Verification
- ✅ Route definition exists: `@app.post("/api/webhooks/stripe", tags=["Webhooks"])`
- ✅ Function defined: `async def stripe_webhook(...)`
- ✅ Syntax: No syntax errors (py_compile passes)
- ❌ Route registration: Not found in app.routes

### Possible Causes

1. **Import Error**: Error during module import stops execution before route registration
2. **Execution Error**: Error in code before webhook route definition
3. **FastAPI Version Issue**: Route registration issue with FastAPI 0.128.0
4. **Route Ordering**: Route defined after an error that prevents registration

## Solution

### Option 1: Check for Import/Execution Errors
Verify no errors occur during module import that would prevent route registration.

### Option 2: Move Route Definition
Move webhook route definition earlier in the file, before potential error points.

### Option 3: Rebuild Container
Rebuild Docker container to ensure latest code is used.

### Option 4: Verify Route Registration
Add explicit route registration check or logging.

## Next Steps

1. Check for import/execution errors in api.py
2. Verify route is defined before any error points
3. Test route registration programmatically
4. Rebuild container if needed

---

**Status**: 🔍 **INVESTIGATING**
