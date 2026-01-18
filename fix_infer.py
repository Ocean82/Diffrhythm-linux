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
from einops import rearrange

print("Current working directory:", os.getcwd())

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
        print(f"   ⚠ WARNING: Audio is silent (max={max_val:.2e}), returning zeros")
        return torch.zeros_like(output, dtype=torch.int16)
    
    # Check for NaN or Inf
    if torch.isnan(output).any():
        print(f"   ✗ ERROR: Audio contains NaN values!")
        return torch.zeros_like(output, dtype=torch.int16)
    
    if torch.isinf(output).any():
        print(f"   ✗ ERROR: Audio contains Inf values!")
        return torch.zeros_like(output, dtype=torch.int16)
    
    # Normalize to target amplitude
    normalized = output * (target_amplitude / max_val)
    
    # Convert to int16 with proper clamping
    normalized = normalized.clamp(-1, 1).mul(32767).to(torch.int16)
    
    print(f"   ✓ Audio normalized successfully: range {normalized.min()} to {normalized.max()}")
    
    return normalized


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
            print(f"     ✗ ERROR: Latent {i} contains NaN values!")
            return False
            
        if has_inf:
            print(f"     ✗ ERROR: Latent {i} contains Inf values!")
            return False
        
        # Check if all zeros
        is_all_zeros = (latent.abs() < 1e-8).all()
        if is_all_zeros:
            print(f"     ✗ ERROR: Latent {i} is all zeros!")
            return False
        
        # Check statistics
        min_val = latent.min().item()
        max_val = latent.max().item()
        mean_val = latent.mean().item()
        std_val = latent.std().item()
        
        print(f"     Stats: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}, std={std_val:.6f}")
        
        # Check for reasonable variance
        if std_val < 1e-6:
            print(f"     ⚠ WARNING: Latent {i} has very low variance (std={std_val:.2e})")
        
        print(f"     ✓ Latent {i} validation passed")
    
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
):
    print(f"   Starting inference with CFM sampling...")
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
                steps=32,
                cfg_strength=4.0,
                start_time=start_time,
                latent_pred_segments=pred_frames,
                batch_infer_num=batch_infer_num
            )
            print(f"   ✓ CFM sampling completed, got {len(latents)} latent(s)")
            
        except Exception as e:
            print(f"   ✗ ERROR during CFM sampling: {e}")
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
                print(f"     ✓ VAE decode successful: shape={output.shape}")
                
            except Exception as e:
                print(f"     ✗ ERROR during VAE decode: {e}")
                raise

            # Validate audio output
            if torch.isnan(output).any():
                print(f"     ✗ ERROR: VAE output contains NaN!")
                raise ValueError("VAE decode produced NaN values")
                
            if torch.isinf(output).any():
                print(f"     ✗ ERROR: VAE output contains Inf!")
                raise ValueError("VAE decode produced Inf values")

            # Rearrange audio batch to a single sequence
            output = rearrange(output, "b d n -> d (b n)")
            print(f"     Audio rearranged: shape={output.shape}")
            
            # Safe normalization
            output = safe_normalize_audio(output.to(torch.float32))
            
            outputs.append(output.cpu())
            print(f"     ✓ Latent {i+1} processed successfully")

        print(f"   ✓ All {len(outputs)} outputs processed successfully")
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
             "Use `-1` for audio start/end (e.g., `[[-1,25], [50.0,-1]]`)."
    )
    parser.add_argument(
        "--batch-infer-num",
        type=int,
        default=1,
        required=False,
        help="number of songs per batch",
    )
    args = parser.parse_args()

    # Validation
    assert (
        args.ref_prompt or args.ref_audio_path
    ), "either ref_prompt or ref_audio_path should be provided"
    assert not (
        args.ref_prompt and args.ref_audio_path
    ), "only one of them should be provided"
    if args.edit:
        assert (
            args.ref_song and args.edit_segments
        ), "reference song and edit segments should be provided for editing"

    device = "cpu"
    print(f"Using device: {device}")

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
        cfm, tokenizer, muq, vae = prepare_model(max_frames, device)
        print("✓ All models loaded successfully")
    except Exception as e:
        print(f"✗ ERROR loading models: {e}")
        raise

    # Lyrics processing
    if args.lrc_path:
        with open(args.lrc_path, "r", encoding='utf-8') as f:
            lrc = f.read()
        print(f"✓ Lyrics loaded from {args.lrc_path}")
    else:
        lrc = ""
        print("⚠ No lyrics provided")

    try:
        lrc_prompt, start_time, end_frame, song_duration = get_lrc_token(
            max_frames, lrc, tokenizer, audio_length, device
        )
        print(f"✓ Lyrics processed: end_frame={end_frame}")
    except Exception as e:
        print(f"✗ ERROR processing lyrics: {e}")
        raise

    # Style processing
    try:
        if args.ref_audio_path:
            style_prompt = get_style_prompt(muq, args.ref_audio_path)
            print(f"✓ Style from audio: {args.ref_audio_path}")
        else:
            style_prompt = get_style_prompt(muq, prompt=args.ref_prompt)
            print(f"✓ Style from prompt: {args.ref_prompt}")
    except Exception as e:
        print(f"✗ ERROR processing style: {e}")
        raise

    try:
        negative_style_prompt = get_negative_style_prompt(device)
        print("✓ Negative style prompt loaded")
    except Exception as e:
        print(f"✗ ERROR loading negative style: {e}")
        raise

    try:
        latent_prompt, pred_frames = get_reference_latent(
            device, max_frames, args.edit, args.edit_segments, args.ref_song, vae
        )
        print("✓ Reference latent prepared")
    except Exception as e:
        print(f"✗ ERROR preparing reference latent: {e}")
        raise

    # Generation
    print("\n" + "="*60)
    print("STARTING GENERATION")
    print("="*60)
    print("⚠ This will take 5-15 minutes on CPU...")
    
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
            song_duration=song_duration
        )
        
        e_t = time.time() - s_t
        print(f"✓ Generation completed in {e_t:.2f} seconds ({e_t/60:.1f} minutes)")
        
    except Exception as e:
        print(f"✗ ERROR during generation: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Validation of generated songs
    if not generated_songs:
        raise ValueError("No songs were generated!")
    
    print(f"✓ Generated {len(generated_songs)} song(s)")
    
    # Select output
    generated_song = random.sample(generated_songs, 1)[0]
    print(f"Selected song tensor: shape={generated_song.shape}, dtype={generated_song.dtype}")

    # Save output
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output_fixed.wav")
    
    try:
        torchaudio.save(output_path, generated_song, sample_rate=44100)
        
        # Verify file was created and has reasonable size
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✓ Audio saved: {output_path}")
            print(f"✓ File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
            
            if file_size < 1000:
                print("⚠ WARNING: File size is very small, audio may be silent")
            else:
                print("✓ File size looks reasonable")
        else:
            print("✗ ERROR: Output file was not created")
            
    except Exception as e:
        print(f"✗ ERROR saving audio: {e}")
        raise

    print("\n" + "="*60)
    print("GENERATION COMPLETE!")
    print("="*60)
    print(f"Output: {output_path}")
    print(f"Duration: {audio_length}s")
    print(f"Generation time: {e_t:.1f}s ({e_t/60:.1f}min)")
    print("Please check the audio file for quality.")