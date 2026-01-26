# Rate Limiter Fix Plan
**Date**: January 26, 2026  
**Status**: 🔧 **ANALYZING**

## Current Issue

The rate limiter was temporarily disabled (`ENABLE_RATE_LIMIT=false`) due to a parameter mismatch error:
```
Exception: parameter `request` must be an instance of starlette.requests.Request
```

## Root Cause

The `slowapi` limiter decorator `@limiter.limit()` expects the decorated function to have a `request: Request` parameter. However, the `generate_music` function has:
```python
async def generate_music(
    gen_request: GenerationRequest,
    request: Request
):
```

The issue is that `slowapi` needs the `request` parameter to be in a specific position or accessed correctly.

## Solution Options

### Option 1: Use Dependency Injection (Recommended)
Remove the decorator and check rate limits manually inside the function:
```python
async def generate_music(
    gen_request: GenerationRequest,
    request: Request
):
    # Check rate limit manually
    if Config.ENABLE_RATE_LIMIT:
        try:
            limiter.check_rate_limit(request)
        except RateLimitExceeded:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
    # ... rest of function
```

### Option 2: Fix Decorator Usage
Use `@app.post()` with dependency injection:
```python
@app.post("/api/v1/generate")
@limiter.limit(f"{Config.RATE_LIMIT_PER_HOUR}/hour")
async def generate_music(
    gen_request: GenerationRequest,
    request: Request = Depends(lambda: Request)
):
    # ... function body
```

### Option 3: Conditional Decorator
Only apply decorator when rate limiting is enabled:
```python
def apply_rate_limit(func):
    if Config.ENABLE_RATE_LIMIT:
        return limiter.limit(f"{Config.RATE_LIMIT_PER_HOUR}/hour")(func)
    return func

@apply_rate_limit
async def generate_music(...):
    # ... function body
```

## Recommended Approach

**Option 1** is the most reliable because:
1. It's explicit and easy to understand
2. Works regardless of FastAPI/slowapi version
3. Allows conditional rate limiting without decorator complexity
4. Easier to test and debug

## Implementation Steps

1. Remove `@limiter.limit()` decorators from `generate_music` and `generate_music_alias`
2. Add manual rate limit check at the start of `generate_music`
3. Test rate limiting functionality
4. Re-enable `ENABLE_RATE_LIMIT=true` in `.env`
5. Restart container and verify

## Testing

After implementation:
1. Test with rate limit enabled
2. Make multiple requests quickly
3. Verify 429 error when limit exceeded
4. Verify normal operation when under limit

---

**Status**: 🔧 **READY TO IMPLEMENT**
