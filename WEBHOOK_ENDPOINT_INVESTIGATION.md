# Webhook Endpoint Investigation
**Date**: January 26, 2026  
**Issue**: `/api/webhooks/stripe` returns 404 Not Found

## Investigation

### Code Verification
- **File**: `/opt/diffrhythm/backend/api.py`
- **Line**: 705-706
- **Route Definition**: `@app.post("/api/webhooks/stripe", tags=["Webhooks"])`
- **Status**: ✅ Code exists on server

### Test Results
- **Request**: `POST /api/webhooks/stripe`
- **Response**: `404 Not Found`
- **Headers**: `Content-Type: application/json`, `stripe-signature: test`
- **Status**: ❌ Route not registered

### Possible Causes

1. **Route Registration Issue**
   - Route defined but not registered with FastAPI
   - Syntax error preventing route registration
   - Route defined after app initialization

2. **FastAPI Route Ordering**
   - Route might be shadowed by another route
   - Middleware blocking the route

3. **Container Code Mismatch**
   - Server code different from local
   - Container using cached/old code

## Next Steps

1. Verify route is registered in FastAPI app
2. Check for syntax errors in api.py
3. Test route registration programmatically
4. Rebuild container if needed

---

**Status**: 🔍 **INVESTIGATING**
