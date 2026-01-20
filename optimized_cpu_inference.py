#!/usr/bin/env python3
"""
Optimized CPU inference with progress tracking and faster settings
"""
import os
import sys
import time
import torch
import torchaudio
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

# Global model cache
MODEL_CACHE = {}


def load_models_once():
    """Load models once and cache them"""
    if MODEL_CACHE:
        print("✓ Using cached models")
        return (
            MODEL_CACHE["cfm"],
            MODEL_CACHE["tokenizer"],
            MODEL_CACHE["muq"],
            MODEL_CACHE["vae"],
        )

    print("Loading models (this takes ~4 minutes on CPU)...")
    print("Progress will be shown for each step...")

    # Optimize CPU performance
    torch.set_num_threads(4)  # Limit CPU threads

    start_total = time.time()

    # Load models with progress tracking
    from infer.infer_utils import prepare_model

    cfm, tokenizer, muq, vae = prepare_model(max_frames=2048, device="cpu")

    # Cache models
    MODEL_CACHE["cfm"] = cfm
    MODEL_CACHE["tokenizer"] = tokenizer
    MODEL_CACHE["muq"] = muq
    MODEL_CACHE["vae"] = vae

    total_time = time.time() - start_total
    print(
        f"✓ All models loaded and cached in {total_time:.1f}s ({total_time/60:.1f} minutes)"
    )

    return cfm, tokenizer, muq, vae


def fast_inference(
    lyrics_text, style_prompt, output_path="./output/optimized_output.wav"
):
    """Run inference with optimized settings for CPU"""
    print("\n" + "=" * 60)
    print("OPTIMIZED CPU INFERENCE")
    print("=" * 60)

    # Load models (cached after first run)
    cfm, tokenizer, muq, vae = load_models_once()

    print("\nPreparing generation...")
    start_prep = time.time()

    # Import utilities
    from infer.infer_utils import (
        get_lrc_token,
        get_style_prompt,
        get_negative_style_prompt,
        get_reference_latent,
    )

    device = "cpu"
    max_frames = 2048
    audio_length = 95

    # Process inputs
    lrc_emb, start_time_val, end_frame, song_duration = get_lrc_token(
        max_frames, lyrics_text, tokenizer, audio_length, device
    )

    style = get_style_prompt(muq, prompt=style_prompt)
    negative_style = get_negative_style_prompt(device)

    latent_prompt, pred_frames = get_reference_latent(
        device, max_frames, False, None, None, vae
    )

    prep_time = time.time() - start_prep
    print(f"✓ Preparation completed in {prep_time:.1f}s")

    # Run optimized inference
    print(f"\nStarting generation...")
    print(f"⚠ This will take 8-15 minutes on CPU (be patient!)")
    print(f"Using optimized settings:")
    print(f"  - CFM steps: 16 (reduced from 32 for speed)")
    print(f"  - Chunked decoding: enabled")
    print(f"  - Single batch: batch_size=1")

    start_gen = time.time()

    with torch.inference_mode():
        # Optimized CFM sampling - fewer steps for speed
        print("  [1/3] CFM sampling (this is the slow part)...")
        latents, trajectory = cfm.sample(
            cond=latent_prompt,
            text=lrc_emb,
            duration=end_frame,
            style_prompt=style,
            negative_style_prompt=negative_style,
            start_time=start_time_val,
            latent_pred_segments=pred_frames,
            song_duration=song_duration,
            steps=16,  # Reduced from 32 for speed
            cfg_strength=4.0,
            batch_infer_num=1,
        )

        cfm_time = time.time() - start_gen
        print(
            f"  ✓ CFM sampling completed in {cfm_time:.1f}s ({cfm_time/60:.1f} minutes)"
        )

        # Process latents
        print("  [2/3] Processing latents...")
        start_decode = time.time()

        outputs = []
        for i, latent in enumerate(latents):
            # Prepare for VAE
            latent = latent.to(torch.float32).transpose(1, 2)

            # VAE decode with chunking
            from infer.infer_utils import decode_audio

            output = decode_audio(latent, vae, chunked=True)

            # Rearrange
            from einops import rearrange

            output = rearrange(output, "b d n -> d (b n)")

            # Safe normalization (using our fix)
            max_val = torch.max(torch.abs(output))
            print(f"    Audio max amplitude: {max_val:.6f}")

            if max_val < 1e-8:
                print(f"    ⚠ WARNING: Audio is silent!")
                output = torch.zeros_like(output, dtype=torch.int16)
            else:
                # Safe normalization
                output = (
                    (output * 0.95 / max_val).clamp(-1, 1).mul(32767).to(torch.int16)
                )
                print(f"    ✓ Audio normalized successfully")

            outputs.append(output.cpu())

        decode_time = time.time() - start_decode
        print(f"  ✓ Decoding completed in {decode_time:.1f}s")

        # Save output
        print("  [3/3] Saving audio...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        generated_song = outputs[0]
        torchaudio.save(output_path, generated_song, sample_rate=44100)

        total_gen_time = time.time() - start_gen

        # Verify output
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"  ✓ Audio saved: {output_path}")
            print(f"  ✓ File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")

            if file_size > 100000:  # > 100KB
                print(f"  ✓ File size looks good!")
                success = True
            else:
                print(f"  ⚠ File size is small - audio may be silent")
                success = False
        else:
            print(f"  ✗ Failed to save audio file")
            success = False

    print(f"\n" + "=" * 60)
    if success:
        print("✓ GENERATION SUCCESSFUL!")
    else:
        print("⚠ GENERATION COMPLETED WITH WARNINGS")
    print("=" * 60)
    print(
        f"Total generation time: {total_gen_time:.1f}s ({total_gen_time/60:.1f} minutes)"
    )
    print(f"Output: {output_path}")

    return success


def main():
    """Main function with multiple test options"""
    print("DiffRhythm Optimized CPU Inference")
    print("=" * 40)
    print("This version is optimized for CPU-only systems:")
    print("- Models are cached after first load")
    print("- Uses faster inference settings")
    print("- Provides progress feedback")
    print("")

    # Test lyrics
    test_lyrics = """[00:00.00]Hello world
[00:05.00]This is a test song
[00:10.00]Can you hear me now
[00:15.00]Testing the optimized version
[00:20.00]Hope this works better"""

    print("Test lyrics:")
    print(test_lyrics)
    print("")

    style_prompt = "upbeat pop song with clear vocals"
    print(f"Style prompt: {style_prompt}")
    print("")

    choice = input("Start optimized inference? (y/n): ").lower().strip()
    if choice != "y":
        print("Cancelled.")
        return

    try:
        success = fast_inference(test_lyrics, style_prompt)

        if success:
            print("\n🎵 SUCCESS! Check the output file and listen to it.")
            print(
                "If this works, you can now generate songs (just be patient with CPU timing)."
            )
        else:
            print("\n⚠ Generation completed but with issues.")
            print(
                "The file was created but may be silent - this helps us debug further."
            )

    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
