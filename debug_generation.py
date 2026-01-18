#!/usr/bin/env python3
"""
Debug script to identify where the output breakdown occurs
"""
import os
import sys
import torch
import torchaudio
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

# Import required modules
from infer.infer_utils import (
    prepare_model, get_lrc_token, get_style_prompt, 
    get_negative_style_prompt, get_reference_latent, decode_audio
)


def load_models_and_setup():
    """Load models and prepare test data"""
    print("\n[1/7] Importing modules...")
    print("   ✓ Imports successful")

    print("\n[2/7] Loading models...")
    cfm, tokenizer, muq, vae = prepare_model(max_frames=2048, device='cpu')
    print("   ✓ All models loaded")
    
    return cfm, tokenizer, muq, vae


def prepare_test_inputs(tokenizer, muq, vae):
    """Prepare test lyrics and style inputs"""
    print("\n[3/7] Creating test lyrics...")
    test_lyrics = """[00:00.00]Hello world
[00:05.00]This is a test
[00:10.00]Can you hear me
[00:15.00]Testing one two three"""

    print("\n[4/7] Processing lyrics...")
    lrc_emb, start_time_val, end_frame, song_duration = get_lrc_token(
        max_frames=2048,
        text=test_lyrics,
        tokenizer=tokenizer,
        max_secs=95.0,
        device='cpu'
    )
    print(f"   ✓ Lyrics tokenized - shape: {lrc_emb.shape}")
    print(f"   ✓ End frame: {end_frame}")
    print(f"   ✓ Song duration: {song_duration}")
    
    print("\n[5/7] Creating style embedding...")
    style = get_style_prompt(muq, prompt="simple pop song")
    print(f"   ✓ Style embedding created - shape: {style.shape}")
    print(f"   ✓ Style stats: min={style.min():.4f}, "
          f"max={style.max():.4f}, mean={style.mean():.4f}")
    
    negative_style = get_negative_style_prompt('cpu')
    print(f"   ✓ Negative style created - shape: {negative_style.shape}")
    
    latent_prompt, pred_frames = get_reference_latent(
        device='cpu',
        max_frames=2048,
        edit=False,
        pred_segments=None,
        ref_song=None,
        vae_model=vae
    )
    print(f"   ✓ Reference latent created - shape: {latent_prompt.shape}")
    print(f"   ✓ Pred frames: {pred_frames}")
    
    return (lrc_emb, start_time_val, end_frame, song_duration, 
            style, negative_style, latent_prompt, pred_frames)


def validate_latent(latent, index):
    """Validate a single latent tensor"""
    print(f"   ✓ Latent {index} shape: {latent.shape}")
    print(f"   ✓ Latent {index} stats: min={latent.min():.6f}, "
          f"max={latent.max():.6f}, mean={latent.mean():.6f}, "
          f"std={latent.std():.6f}")
    
    # Check for problematic values
    has_nan = torch.isnan(latent).any()
    has_inf = torch.isinf(latent).any()
    is_all_zeros = (latent.abs() < 1e-8).all()
    
    print(f"   ✓ Latent {index} validation: NaN={has_nan}, "
          f"Inf={has_inf}, AllZeros={is_all_zeros}")
    
    if has_nan or has_inf:
        print(f"   ✗ ERROR: Latent {index} contains NaN or Inf values!")
        return False
        
    if is_all_zeros:
        print(f"   ✗ ERROR: Latent {index} is all zeros!")
        return False
    
    return True


def test_cfm_sampling(cfm, lrc_emb, start_time_val, end_frame, song_duration, style, negative_style, latent_prompt, pred_frames):
    """Test CFM sampling and validate latents"""
    print("\n[6/7] Testing CFM sampling...")
    print("   This will take several minutes on CPU...")
    
    with torch.inference_mode():
        latents, trajectory = cfm.sample(
            cond=latent_prompt,
            text=lrc_emb,
            duration=end_frame,
            style_prompt=style,
            negative_style_prompt=negative_style,
            start_time=start_time_val,
            latent_pred_segments=pred_frames,
            song_duration=song_duration,
            steps=32,
            cfg_strength=4.0,
            batch_infer_num=1
        )
        
        print("   ✓ CFM sampling completed")
        print(f"   ✓ Number of latent outputs: {len(latents)}")
        
        # Validate each latent - return latents if all valid, None otherwise
        return latents if all(validate_latent(latent, i) for i, latent in enumerate(latents)) else None


def decode_and_validate_latent(latent, vae, index):
    """Decode and validate a single latent"""
    print(f"   Testing decode for latent {index}...")
    
    # Prepare latent for VAE (transpose to [b d t])
    latent_for_vae = latent.to(torch.float32).transpose(1, 2)
    print(f"   ✓ Latent prepared for VAE - shape: {latent_for_vae.shape}")
    
    # Test VAE decode
    try:
        output = decode_audio(latent_for_vae, vae, chunked=True)
        print(f"   ✓ VAE decode successful - shape: {output.shape}")
        print(f"   ✓ Audio stats: min={output.min():.6f}, max={output.max():.6f}, mean={output.mean():.6f}, std={output.std():.6f}")
        
        # Check audio quality
        has_nan = torch.isnan(output).any()
        has_inf = torch.isinf(output).any()
        is_all_zeros = (output.abs() < 1e-8).all()
        max_abs = torch.max(torch.abs(output))
        
        print(f"   ✓ Audio validation: NaN={has_nan}, Inf={has_inf}, AllZeros={is_all_zeros}, MaxAbs={max_abs:.6f}")
        
        if has_nan or has_inf:
            print("   ✗ ERROR: Audio contains NaN or Inf values!")
            return False
            
        if is_all_zeros:
            print("   ✗ ERROR: Audio is all zeros!")
            return False
            
        if max_abs < 1e-6:
            print("   ✗ ERROR: Audio is too quiet (max amplitude < 1e-6)!")
            return False
        
        # Test normalization and save
        return save_normalized_audio(output, max_abs, index)
            
    except Exception as e:
        print(f"   ✗ ERROR during VAE decode: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vae_decoding(latents, vae):
    """Test VAE decoding and save outputs"""
    print("\n[7/7] Testing VAE decoding...")
    
    # Process each latent and check if all succeed
    return all(decode_and_validate_latent(latent, vae, i) 
               for i, latent in enumerate(latents))


def save_normalized_audio(output, max_abs, index):
    """Normalize and save audio output"""
    print("   Testing normalization...")
    if max_abs > 0:
        normalized = (output.div(max_abs).clamp(-1, 1)
                      .mul(32767).to(torch.int16))
        print("   ✓ Normalization successful")
        
        # Save test output
        output_dir = Path("./output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"debug_test_{index}.wav"
        
        # Rearrange if needed
        if normalized.dim() == 3:
            from einops import rearrange
            normalized = rearrange(normalized, "b d n -> d (b n)")
        
        torchaudio.save(str(output_path), normalized.cpu(), sample_rate=44100)
        
        file_size = output_path.stat().st_size
        print(f"   ✓ Audio saved: {output_path} ({file_size} bytes)")
        
        if file_size < 1000:
            print(f"   ⚠ WARNING: File size is very small ({file_size} bytes)")
        return True
    else:
        print("   ✗ ERROR: Cannot normalize - "
              "max amplitude is zero!")
        return False


def debug_generation_pipeline():
    """Debug each step of the generation pipeline"""
    print("="*80)
    print("DiffRhythm Generation Debug")
    print("="*80)
    
    try:
        # Load models and setup
        cfm, tokenizer, muq, vae = load_models_and_setup()
        
        # Prepare test inputs
        (lrc_emb, start_time_val, end_frame, song_duration, 
         style, negative_style, latent_prompt, pred_frames) = (
            prepare_test_inputs(tokenizer, muq, vae)
        )
        
        # Test CFM sampling
        latents = test_cfm_sampling(
            cfm, lrc_emb, start_time_val, end_frame, song_duration, 
            style, negative_style, latent_prompt, pred_frames
        )
        if latents is None:
            return False
        
        # Test VAE decoding
        if not test_vae_decoding(latents, vae):
            return False
        
        print("\n" + "="*80)
        print("✓ DEBUG COMPLETE - All steps successful!")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = debug_generation_pipeline()
    sys.exit(0 if success else 1)