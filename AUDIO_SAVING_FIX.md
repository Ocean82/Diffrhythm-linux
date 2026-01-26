# Audio Saving Fix - Thread Safety Issue
**Date**: January 26, 2026  
**Issue**: Jobs failing with "Failed to save audio with any available method"

## Problem Identified

### Root Cause
The `set_timeout()` function uses Python's `signal` module, which can only be used in the main thread. When audio saving runs in a worker thread (as it does in the FastAPI background worker), calling `signal.signal()` raises:

```
ValueError: signal only works in main thread of the main interpreter
```

This causes all three audio saving methods (torchaudio, scipy, soundfile) to fail, resulting in job failures.

### Error Pattern
```
Method 1: Attempting save with torchaudio...
  WARN torchaudio failed: signal only works in main thread of the main interpreter
Method 2: Attempting save with scipy.io.wavfile...
  WARN scipy failed: signal only works in main thread of the main interpreter
Method 3: Attempting save with soundfile...
  WARN soundfile failed: signal only works in main thread of the main interpreter
```

## Solution

### Fix Applied
Modified `infer/infer.py` to:
1. Import `threading` module
2. Add `is_main_thread()` function to check if we're in the main thread
3. Update `set_timeout()` and `clear_timeout()` to only use signals in the main thread
4. Skip timeout mechanism in worker threads (audio saving should be fast enough without timeout)

### Code Changes
```python
import threading

def is_main_thread():
    """Check if we're in the main thread"""
    return threading.current_thread() is threading.main_thread()

def set_timeout(seconds):
    """Set a timeout alarm (Unix only, main thread only)"""
    if HAS_SIGNAL_ALARM and is_main_thread():
        try:
            signal.signal(signal.SIGALRM, audio_timeout_handler)
            signal.alarm(seconds)
        except ValueError:
            # Signal can only be used in main thread
            pass

def clear_timeout():
    """Clear the timeout alarm (Unix only, main thread only)"""
    if HAS_SIGNAL_ALARM and is_main_thread():
        try:
            signal.alarm(0)
        except ValueError:
            # Signal can only be used in main thread
            pass
```

## Deployment

1. ✅ Fixed `infer/infer.py` locally
2. ✅ Deployed to server via SCP
3. ✅ Restarted container
4. ⏳ Testing audio saving

## Expected Result

Audio saving should now work in worker threads:
- torchaudio will attempt to save (without timeout)
- If it fails, scipy will attempt
- If that fails, soundfile will attempt
- All methods will work without signal-based timeout errors

---

**Status**: ✅ **FIX DEPLOYED**  
**Testing**: ⏳ **IN PROGRESS**
