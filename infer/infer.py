#!/usr/bin/env python3
"""
Fixed version of infer.py with proper validation and error handling
"""
import argparse
import os
import time
import random

import torch
import torchaudio
import numpy as np
from einops import rearrange
import sys

# Cross-platform signal handling
if sys.platform != 'win32':
    import signal
    import threading
    HAS_SIGNAL_ALARM = True
else:
    HAS_SIGNAL_ALARM = False
    import threading

print("Current working directory:", os.getcwd())

# Audio saving timeout handler
class TimeoutException(Exception):
    pass

def audio_timeout_handler(signum, frame):
    raise TimeoutException("Audio saving timed out")

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

try:
    from infer.infer_utils import (
        decode_audio,
        get_lrc_token,
        get_negative_style_prompt,
        get_reference_latent,
        get_style_prompt,
        prepare_model,
    )
except (ImportError, ModuleNotFoundError):
    from infer_utils import (
        decode_audio,
        get_lrc_token,
        get_negative_style_prompt,
        get_reference_latent,
        get_style_prompt,
        prepare_model,
    )


def safe_normalize_audio(output, target_amplitude=0.95, min_threshold=1e-8):
    """
    Safely normalize audio with proper validation

    Args:
        output: Audio tensor to normalize
        target_amplitude: Target peak amplitude (0-1)
        min_threshold: Minimum amplitude threshold for non-silent audio

    Returns:
        Normalized int16 audio tensor
    """
    # Get maximum absolute value
    max_val = torch.max(torch.abs(output))

    print(f"   Audio stats: max_abs={max_val:.8f}, shape={output.shape}")

    # Check if audio is essentially silent
    if max_val < min_threshold:
        print(f"   WARN WARNING: Audio is silent (max={max_val:.2e}), returning zeros")
        return torch.zeros_like(output, dtype=torch.int16)

    # Check for NaN or Inf
    if torch.isnan(output).any():
        print("   ERROR ERROR: Audio contains NaN values!")
        return torch.zeros_like(output, dtype=torch.int16)

    if torch.isinf(output).any():
        print("   ERROR ERROR: Audio contains Inf values!")
        return torch.zeros_like(output, dtype=torch.int16)

    # Normalize to target amplitude
    normalized = output * (target_amplitude / max_val)

    # Convert to int16 with proper clamping
    normalized = normalized.clamp(-1, 1).mul(32767).to(torch.int16)

    print(
        f"   OK Audio normalized successfully: range {normalized.min()} to {normalized.max()}"
    )

    return normalized


def validate_audio_tensor(audio):
    """
    Validate audio tensor for saving
    
    Args:
        audio: Audio tensor to validate
        
    Returns:
        bool: True if valid
        
    Raises:
        ValueError: If tensor is invalid
    """
    print(f"   Validating audio tensor...")
    print(f"     Shape: {audio.shape}")
    print(f"     Dtype: {audio.dtype}")
    
    # Check dimensions
    if audio.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {audio.dim()}D")
    
    # Check dtype
    if audio.dtype not in [torch.float32, torch.int16]:
        raise ValueError(f"Expected float32 or int16, got {audio.dtype}")
    
    # Check size
    if audio.numel() == 0:
        raise ValueError("Empty audio tensor")
    
    print(f"     OK Audio tensor is valid")
    return True


def save_audio_robust(audio, output_path, sample_rate=44100, timeout=60):
    """
    Save audio with multiple fallback methods
    
    Args:
        audio: Audio tensor [channels, samples] int16 or float32
        output_path: Path to save WAV file
        sample_rate: Sample rate in Hz
        timeout: Timeout in seconds
        
    Returns:
        bool: True if successful
    """
    print(f"   Saving audio to {output_path}...")
    
    # Validate tensor
    try:
        validate_audio_tensor(audio)
    except ValueError as e:
        print(f"   ERROR Audio validation failed: {e}")
        raise
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Method 1: Try torchaudio (native PyTorch method)
    print(f"   Method 1: Attempting save with torchaudio...")
    try:
        # Set timeout (Unix only)
        set_timeout(timeout)

        try:
            torchaudio.save(output_path, audio.unsqueeze(0) if audio.dim() == 1 else audio, sample_rate=sample_rate)
            clear_timeout()

            # Verify file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"     OK Saved with torchaudio ({file_size:,} bytes)")
                return True
        except TimeoutException:
            clear_timeout()
            print(f"     WARN torchaudio save timed out after {timeout}s")
    except Exception as e:
        clear_timeout()
        print(f"     WARN torchaudio failed: {e}")
    
    # Method 2: Try scipy
    print(f"   Method 2: Attempting save with scipy.io.wavfile...")
    try:
        import scipy.io.wavfile as wavfile

        # Convert to numpy
        if audio.dtype == torch.int16:
            audio_np = audio.cpu().numpy()
        else:  # float32
            # Normalize float32 to int16 range
            audio_np = audio.cpu().numpy()
            audio_np = (audio_np * 32767).astype(np.int16)

        # Ensure correct shape [samples, channels]
        if audio_np.shape[0] < audio_np.shape[1]:
            audio_np = audio_np.T

        set_timeout(timeout)

        try:
            wavfile.write(output_path, sample_rate, audio_np)
            clear_timeout()

            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"     OK Saved with scipy.io.wavfile ({file_size:,} bytes)")
                return True
        except TimeoutException:
            clear_timeout()
            print(f"     WARN scipy save timed out after {timeout}s")
    except ImportError:
        clear_timeout()
        print(f"     WARN scipy not available")
    except Exception as e:
        clear_timeout()
        print(f"     WARN scipy failed: {e}")
    
    # Method 3: Try soundfile
    print(f"   Method 3: Attempting save with soundfile...")
    try:
        import soundfile as sf

        # Convert to numpy float32
        if audio.dtype == torch.int16:
            audio_np = audio.cpu().numpy().astype(np.float32) / 32768.0
        else:
            audio_np = audio.cpu().numpy()

        # Ensure correct shape [samples, channels]
        if audio_np.shape[0] < audio_np.shape[1]:
            audio_np = audio_np.T

        set_timeout(timeout)

        try:
            sf.write(output_path, audio_np, sample_rate)
            clear_timeout()

            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"     OK Saved with soundfile ({file_size:,} bytes)")
                return True
        except TimeoutException:
            clear_timeout()
            print(f"     WARN soundfile save timed out after {timeout}s")
    except ImportError:
        clear_timeout()
        print(f"     WARN soundfile not available")
    except Exception as e:
        clear_timeout()
        print(f"     WARN soundfile failed: {e}")

    # All methods failed
    clear_timeout()
    raise RuntimeError(f"Failed to save audio with any available method")


def validate_latents(latents, step_name=""):
    """
    Validate latent tensors for common issues

    Args:
        latents: List of latent tensors or single tensor
        step_name: Name of the step for logging

    Returns:
        bool: True if valid, False if invalid
    """
    if not isinstance(latents, (list, tuple)):
        latents = [latents]

    print(f"   Validating {len(latents)} latent tensor(s) from {step_name}...")

    for i, latent in enumerate(latents):
        print(f"     Latent {i}: shape={latent.shape}, dtype={latent.dtype}")

        # Check for NaN/Inf
        has_nan = torch.isnan(latent).any()
        has_inf = torch.isinf(latent).any()

        if has_nan:
            print(f"     ERROR ERROR: Latent {i} contains NaN values!")
            return False

        if has_inf:
            print(f"     ERROR ERROR: Latent {i} contains Inf values!")
            return False

        # Check if all zeros
        is_all_zeros = (latent.abs() < 1e-8).all()
        if is_all_zeros:
            print(f"     ERROR ERROR: Latent {i} is all zeros!")
            return False

        # Check statistics
        min_val = latent.min().item()
        max_val = latent.max().item()
        mean_val = latent.mean().item()
        std_val = latent.std().item()

        print(
            f"     Stats: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}, std={std_val:.6f}"
        )

        # Check for reasonable variance
        if std_val < 1e-6:
            print(
                f"     WARN WARNING: Latent {i} has very low variance (std={std_val:.2e})"
            )

        print(f"     OK Latent {i} validation passed")

    return True


def inference(
    cfm_model,
    vae_model,
    cond,
    text,
    duration,
    style_prompt,
    negative_style_prompt,
    start_time,
    pred_frames,
    batch_infer_num,
    song_duration,
    chunked=False,
    steps=32,
    cfg_strength=4.0,
):
    print("   Starting inference with CFM sampling...")
    print(f"   Duration: {duration}, Batch size: {batch_infer_num}")

    with torch.inference_mode():
        # CFM sampling with validation
        try:
            latents, trajectory = cfm_model.sample(
                cond=cond,
                text=text,
                duration=duration,
                style_prompt=style_prompt,
                max_duration=duration,
                song_duration=song_duration,
                negative_style_prompt=negative_style_prompt,
                steps=steps,
                cfg_strength=cfg_strength,
                start_time=start_time,
                latent_pred_segments=pred_frames,
                batch_infer_num=batch_infer_num,
            )
            print(f"   OK CFM sampling completed, got {len(latents)} latent(s)")

        except Exception as e:
            print(f"   ERROR ERROR during CFM sampling: {e}")
            raise

        # Validate latents
        if not validate_latents(latents, "CFM sampling"):
            raise ValueError("CFM sampling produced invalid latents")

        outputs = []
        for i, latent in enumerate(latents):
            print(f"   Processing latent {i+1}/{len(latents)}...")

            # Prepare latent for VAE
            latent = latent.to(torch.float32)
            latent = latent.transpose(1, 2)  # [b d t]
            print(f"     Latent prepared for VAE: shape={latent.shape}")

            # VAE decoding with validation
            try:
                output = decode_audio(latent, vae_model, chunked=chunked)
                print(f"     OK VAE decode successful: shape={output.shape}")

            except Exception as e:
                print(f"     ERROR ERROR during VAE decode: {e}")
                raise

            # Validate audio output
            if torch.isnan(output).any():
                print("     ERROR ERROR: VAE output contains NaN!")
                raise ValueError("VAE decode produced NaN values")

            if torch.isinf(output).any():
                print("     ERROR ERROR: VAE output contains Inf!")
                raise ValueError("VAE decode produced Inf values")

            # Rearrange audio batch to a single sequence
            output = rearrange(output, "b d n -> d (b n)")
            print(f"     Audio rearranged: shape={output.shape}")

            # Safe normalization
            output = safe_normalize_audio(output.to(torch.float32))
            if torch.max(torch.abs(output)).item() == 0:
                raise ValueError("VAE decode produced silent audio")

            outputs.append(output.cpu())
            print(f"     OK Latent {i+1} processed successfully")

        print(f"   OK All {len(outputs)} outputs processed successfully")
        return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lrc-path",
        type=str,
        help="lyrics of target song",
    )
    parser.add_argument(
        "--ref-prompt",
        type=str,
        help="reference prompt as style prompt for target song",
        required=False,
    )
    parser.add_argument(
        "--ref-audio-path",
        type=str,
        help="reference audio as style prompt for target song",
        required=False,
    )
    parser.add_argument(
        "--chunked",
        action="store_true",
        help="whether to use chunked decoding",
    )
    parser.add_argument(
        "--audio-length",
        type=int,
        default=95,
        help="length of generated song, supported values are exactly 95 or any value between 96 and 285 (inclusive).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="output directory for generated song",
    )
    parser.add_argument(
        "--edit",
        action="store_true",
        help="whether to open edit mode",
    )
    parser.add_argument(
        "--ref-song",
        type=str,
        required=False,
        help="reference prompt as latent prompt for editing",
    )
    parser.add_argument(
        "--edit-segments",
        type=str,
        required=False,
        help="Time segments to edit (in seconds). Format: `[[start1,end1],...]`. "
        "Use `-1` for audio start/end (e.g., `[[-1,25], [50.0,-1]]`).",
    )
    parser.add_argument(
        "--batch-infer-num",
        type=int,
        default=1,
        required=False,
        help="number of songs per batch",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        required=False,
        help="Override ODE integration steps (default: 32 GPU, 8 CPU). Lower = faster but lower quality",
    )
    parser.add_argument(
        "--cfg-strength",
        type=float,
        default=None,
        required=False,
        help="Override CFG strength (default: 4.0 GPU, 2.0 CPU)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        required=False,
        help="ODE integration timeout in seconds (default: auto-calculated for CPU)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        required=False,
        choices=["preview", "draft", "standard", "high", "maximum", "ultra"],
        help="Quality preset (overrides --steps and --cfg-strength)",
    )
    parser.add_argument(
        "--auto-master",
        action="store_true",
        help="Automatically apply mastering to output",
    )
    parser.add_argument(
        "--master-preset",
        type=str,
        default="balanced",
        choices=["subtle", "balanced", "loud", "broadcast"],
        help="Mastering preset (used with --auto-master)",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA adapter directory for fine-tuning (requires PEFT library)",
    )
    args = parser.parse_args()

    # Validation
    if not (args.ref_prompt or args.ref_audio_path):
        parser.error("either ref_prompt or ref_audio_path should be provided")
    if args.ref_prompt and args.ref_audio_path:
        parser.error("only one of them should be provided")
    if args.edit and (not args.ref_song or not args.edit_segments):
        parser.error("reference song and edit segments should be provided for editing")

    # Load quality presets if specified
    if args.preset:
        try:
            from infer.quality_presets import get_preset
        except ImportError:
            from quality_presets import get_preset

        preset = get_preset(args.preset)
        print(f"\nOK Using quality preset: {args.preset}")
        print(f"  - Steps: {preset.steps}")
        print(f"  - CFG Strength: {preset.cfg_strength}")
        print(f"  - {preset.description}")

        # Override steps and cfg if not explicitly set
        if args.steps is None:
            args.steps = preset.steps
        if args.cfg_strength is None:
            args.cfg_strength = preset.cfg_strength

    # Device detection
    if torch.cuda.is_available():
        device = "cuda"
        print(f"OK Using GPU: {torch.cuda.get_device_name(0)}")
        default_steps = 32
        default_cfg = 4.0
        ode_timeout = args.timeout  # Use user-specified or None
    else:
        device = "cpu"
        print(f"WARN Using CPU - inference will be slow!")

        # CPU optimization: reduce steps for faster inference
        default_steps = 16  # Reduced from 32 - balanced for quality/latency
        default_cfg = 2.0  # Reduced from 4.0 for CPU

        # Calculate estimated time
        estimated_per_step = 45  # ~45 seconds per step on CPU for 1B model
        user_steps = args.steps if args.steps else default_steps
        estimated_total = user_steps * estimated_per_step

        print(f"\n{'='*60}")
        print(f"CPU OPTIMIZATION SETTINGS")
        print(f"{'='*60}")
        print(f"  - ODE steps: {user_steps} (default: {default_steps}, original: 32)")
        print(f"  - CFG strength: {args.cfg_strength if args.cfg_strength else default_cfg}")
        print(f"  - Estimated time: {estimated_total//60}m {estimated_total%60}s")
        print(f"  - Timeout: {args.timeout if args.timeout else estimated_total * 2}s")
        print(f"")
        print(f"  TIP: For faster inference, use GPU or reduce audio-length")
        print(f"  TIP: Use --steps 4 for quick preview (lower quality)")
        print(f"  TIP: Use --steps 16 for better quality (slower)")
        print(f"{'='*60}\n")

        ode_timeout = args.timeout if args.timeout else estimated_total * 2

    # Apply user overrides or defaults
    cpu_optimized_steps = args.steps if args.steps else default_steps
    cpu_optimized_cfg = args.cfg_strength if args.cfg_strength else default_cfg

    # Audio length validation
    audio_length = args.audio_length
    if audio_length == 95:
        max_frames = 2048
    elif 95 < audio_length <= 285:
        max_frames = 6144
    else:
        raise ValueError(
            f"Invalid audio_length: {audio_length}. "
            "Supported values are exactly 95 or any value between 96 and 285 (inclusive)."
        )

    print(f"Audio length: {audio_length}s, Max frames: {max_frames}")

    # Model preparation
    print("Loading models...")
    try:
        cfm, tokenizer, muq, vae = prepare_model(max_frames, device, lora_adapter_path=args.lora_path)
        print("OK All models loaded successfully")
    except Exception as e:
        print(f"ERROR ERROR loading models: {e}")
        raise

    # Lyrics processing
    if args.lrc_path:
        with open(args.lrc_path, "r", encoding="utf-8") as f:
            lrc = f.read()
        print(f"OK Lyrics loaded from {args.lrc_path}")
    else:
        lrc = ""
        print("WARN No lyrics provided")

    try:
        lrc_prompt, start_time, end_frame, song_duration = get_lrc_token(
            max_frames, lrc, tokenizer, audio_length, device
        )
        print(f"OK Lyrics processed: end_frame={end_frame}")
    except Exception as e:
        print(f"ERROR ERROR processing lyrics: {e}")
        raise

    # Style processing
    try:
        if args.ref_audio_path:
            style_prompt = get_style_prompt(muq, args.ref_audio_path)
            print(f"OK Style from audio: {args.ref_audio_path}")
        else:
            style_prompt = get_style_prompt(muq, prompt=args.ref_prompt)
            print(f"OK Style from prompt: {args.ref_prompt}")
    except Exception as e:
        print(f"ERROR ERROR processing style: {e}")
        raise

    try:
        negative_style_prompt = get_negative_style_prompt(device)
        print("OK Negative style prompt loaded")
    except Exception as e:
        print(f"ERROR ERROR loading negative style: {e}")
        raise

    try:
        latent_prompt, pred_frames = get_reference_latent(
            device, max_frames, args.edit, args.edit_segments, args.ref_song, vae
        )
        print("OK Reference latent prepared")
    except Exception as e:
        print(f"ERROR ERROR preparing reference latent: {e}")
        raise

    # Generation
    print("\n" + "=" * 60)
    print("STARTING GENERATION")
    print("=" * 60)

    if device == "cpu":
        print("WARN Running on CPU - this will take significant time")
        print("WARN Progress will be shown for each ODE step")
        print("WARN You can Ctrl+C to interrupt if needed")
    else:
        print("OK Running on GPU - should complete in 1-3 minutes")

    # Set ODE timeout for CPU
    if device == "cpu":
        cfm.set_ode_timeout(ode_timeout)

    s_t = time.time()
    try:
        generated_songs = inference(
            cfm_model=cfm,
            vae_model=vae,
            cond=latent_prompt,
            text=lrc_prompt,
            duration=end_frame,
            style_prompt=style_prompt,
            negative_style_prompt=negative_style_prompt,
            start_time=start_time,
            pred_frames=pred_frames,
            chunked=args.chunked,
            batch_infer_num=args.batch_infer_num,
            song_duration=song_duration,
            steps=cpu_optimized_steps,
            cfg_strength=cpu_optimized_cfg,
        )

        e_t = time.time() - s_t
        print(f"OK Generation completed in {e_t:.2f} seconds ({e_t/60:.1f} minutes)")

    except Exception as e:
        print(f"ERROR ERROR during generation: {e}")
        import traceback

        traceback.print_exc()
        raise

    # Validation of generated songs
    if not generated_songs:
        raise ValueError("No songs were generated!")

    print(f"OK Generated {len(generated_songs)} song(s)")

    # Select output
    generated_song = random.sample(generated_songs, 1)[0]
    print(
        f"Selected song tensor: shape={generated_song.shape}, dtype={generated_song.dtype}"
    )

    # Save output
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output_fixed.wav")

    try:
        # Use robust audio saving with fallback methods
        save_audio_robust(generated_song, output_path, sample_rate=44100, timeout=60)

        # Verify file was created and has reasonable size
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"OK Audio saved: {output_path}")
            print(f"OK File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")

            if file_size < 1000:
                print("WARN WARNING: File size is very small, audio may be silent")
            elif file_size > 1000 and file_size < 10000000:
                print("OK File size is reasonable for 95-second song")
            else:
                print(f"WARN File size is larger than expected, but may be valid")
        else:
            print("ERROR ERROR: Output file was not created")

    except TimeoutException as e:
        print(f"ERROR ERROR: Audio saving timed out: {e}")
        print("WARN This indicates a codec or backend issue")
        print("WARN Try installing: pip install scipy soundfile")
        raise
    except Exception as e:
        print(f"ERROR ERROR saving audio: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Auto-mastering if requested
    final_output_path = output_path
    if args.auto_master:
        print("\n" + "=" * 60)
        print("APPLYING MASTERING")
        print("=" * 60)
        try:
            try:
                from post_processing.mastering import master_audio_file
            except ImportError:
                import sys
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from post_processing.mastering import master_audio_file

            mastered_path = os.path.join(output_dir, "output_mastered.wav")
            master_audio_file(output_path, mastered_path, preset=args.master_preset, verbose=True)
            final_output_path = mastered_path
            print(f"OK Mastered audio saved to: {mastered_path}")
        except Exception as e:
            print(f"WARN Mastering failed: {e}")
            print("  Raw output still available at:", output_path)

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE!")
    print("=" * 60)
    print(f"Output: {final_output_path}")
    if args.auto_master and final_output_path != output_path:
        print(f"Raw output: {output_path}")
    print(f"Duration: {audio_length}s")
    print(f"Generation time: {e_t:.1f}s ({e_t/60:.1f}min)")
    if not args.auto_master:
        print("\nTIP: Add --auto-master for automatic audio mastering")
