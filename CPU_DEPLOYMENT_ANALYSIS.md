# CPU-Only Deployment Analysis

## Performance Reality

### Single Song Generation (95s)
- **Time**: 15-25 minutes
- **CPU Usage**: 100% on all cores
- **RAM**: 8-16GB

### Concurrent Requests
- **1 user**: 20 min wait ⚠️
- **5 users**: 100 min wait ❌
- **10 users**: 200 min wait ❌

## Viable CPU Strategies

### Strategy 1: Pre-Generation
Generate songs in advance, not on-demand:
```python
# Batch generate overnight
songs = [
    {"style": "folk", "lyrics": "..."},
    {"style": "pop", "lyrics": "..."},
]
# Run batch: 10 songs = 3-4 hours
```

### Strategy 2: Cloud GPU API
Use your backend as proxy to cloud GPU:
```python
# Your backend -> Modal/Replicate -> Return audio
# Cost: $0.01-0.05 per song
# Time: 1-2 minutes
```

### Strategy 3: Hybrid
- CPU for testing/development
- GPU for production
- Queue system handles both

## Recommended: Use Cloud GPU

### Option A: Modal.com
```python
# modal_deploy.py
import modal

stub = modal.Stub("diffrhythm")
image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

@stub.function(gpu="T4", timeout=600)
def generate_song(lyrics, style_prompt):
    # Your inference code
    return audio_bytes

# Deploy: modal deploy modal_deploy.py
# Cost: ~$0.02 per song
```

### Option B: Replicate
- Upload model to Replicate
- Pay per inference: $0.01-0.05
- No infrastructure management

### Option C: AWS Lambda + EFS
- Not viable - Lambda has 15min timeout
- Song generation takes 20+ min on CPU

## Bottom Line

**CPU-only is NOT production-ready** for on-demand generation.

**Your options**:
1. Rent GPU ($0.50/hour = $360/month)
2. Pay-per-use GPU API ($0.02/song)
3. Pre-generate songs (batch processing)
4. Accept 20+ minute wait times (unusable)

**Recommendation**: Use Modal/Replicate for $0.02/song, no infrastructure needed.
