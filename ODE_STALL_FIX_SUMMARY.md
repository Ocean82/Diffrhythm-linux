# ODE Integration Stall Fix Summary

## Problem
Song generation was stalling during ODE (Ordinary Differential Equation) integration, particularly on CPU.

## Root Causes Identified

1. **Large Model Size**: The DiffRhythm model is ~1B parameters (dim=2048, depth=16, heads=32)
2. **CPU FP16 Slowdown**: Half-precision (float16) operations are emulated on CPU, causing significant slowdowns
3. **No Progress Tracking**: Original code had minimal progress reporting
4. **No Timeout Mechanism**: ODE integration could run indefinitely
5. **High Step Count**: Default 32 steps × 2 forward passes (CFG) = 64 transformer forward passes

## Changes Made

### 1. `model/cfm.py` - ODE Integration Improvements

- **Added `ODEProgressTracker` class**: Provides detailed progress tracking with:
  - Step-by-step progress reporting
  - Elapsed time and ETA calculation
  - Configurable timeout support
  - Interruption capability

- **Added manual Euler integration**: `_manual_euler_integration()` method provides better control than `torchdiffeq.odeint`:
  - Progress tracking per step
  - Timeout checking
  - Memory cleanup for CPU
  - Graceful interruption support

- **Added timeout support**: `set_ode_timeout()` method allows setting integration timeout

### 2. `infer/infer.py` - Inference Improvements

- **Auto-detect CPU/GPU**: Automatically selects optimal settings based on hardware
- **Reduced default CPU steps**: 8 steps (down from 32) for reasonable quality with acceptable time
- **Added command-line arguments**:
  - `--steps`: Override ODE integration steps
  - `--cfg-strength`: Override CFG strength
  - `--timeout`: Set explicit timeout in seconds
- **Better time estimates**: Shows estimated completion time based on detected hardware

### 3. `infer/infer_utils.py` - Dtype Optimization

- **CPU float32 mode**: Uses float32 on CPU for better performance (FP16 emulation is slow)
- **GPU float16 mode**: Keeps float16 on GPU for efficiency
- **Consistent dtype handling**: All tensor operations now respect device-appropriate dtype

### 4. New Diagnostic Tool: `diagnose_ode_stall.py`

Run this to diagnose issues:
```bash
python diagnose_ode_stall.py
```

Features:
- System information check
- Model size analysis
- Transformer benchmark
- Performance recommendations

## Usage Examples

### Quick Preview (fastest, lower quality)
```bash
python -m infer.infer --lrc-path lyrics.lrc --ref-prompt "pop music" --steps 4 --audio-length 95
```

### Balanced Quality (recommended for CPU)
```bash
python -m infer.infer --lrc-path lyrics.lrc --ref-prompt "pop music" --steps 8 --audio-length 95
```

### High Quality (slow on CPU, recommended for GPU)
```bash
python -m infer.infer --lrc-path lyrics.lrc --ref-prompt "pop music" --steps 16 --audio-length 95
```

### With Explicit Timeout (30 minutes)
```bash
python -m infer.infer --lrc-path lyrics.lrc --ref-prompt "pop music" --timeout 1800
```

## Performance Expectations

| Hardware | Steps | Estimated Time (95s audio) |
|----------|-------|---------------------------|
| RTX 4090 | 32    | 1-2 minutes               |
| RTX 3090 | 32    | 2-3 minutes               |
| RTX 3060 | 32    | 3-5 minutes               |
| CPU (modern) | 8 | 6-10 minutes             |
| CPU (modern) | 16 | 12-20 minutes           |
| CPU (older) | 8  | 15-30 minutes            |

## Troubleshooting

### Still Stalling?

1. **Run diagnostics**: `python diagnose_ode_stall.py`

2. **Reduce steps further**:
   ```bash
   python -m infer.infer --steps 4 ...
   ```

3. **Set explicit timeout**:
   ```bash
   python -m infer.infer --timeout 600 ...  # 10 minutes
   ```

4. **Check system resources**:
   - CPU usage (should be high during inference)
   - RAM usage (should have at least 8GB free)
   - Disk I/O (shouldn't be a bottleneck)

5. **Environment variables for CPU optimization**:
   ```bash
   export OMP_NUM_THREADS=4
   export MKL_NUM_THREADS=4
   ```

### Quality vs Speed Trade-off

| Steps | Quality | Notes |
|-------|---------|-------|
| 4     | Low     | Quick preview, noticeable artifacts |
| 8     | Medium  | Good for testing, some artifacts |
| 16    | Good    | Recommended minimum for final output |
| 32    | Best    | Original default, highest quality |

## Technical Details

The ODE integration solves the flow matching equation:
```
dx/dt = f(x, t)
```

Each step requires:
1. Forward pass through transformer for `pred`
2. Forward pass through transformer for `null_pred` (CFG)
3. Compute: `output = pred + (pred - null_pred) * cfg_strength`

With a 1B parameter model, each forward pass is computationally expensive, especially on CPU without optimized FP16 support.
