#!/usr/bin/env python3
"""
Test vocal generation with DiffRhythm - to be copied to DiffRhythm-LINUX directory
"""
import os
import sys
import torch
from pathlib import Path
import time

# Add current directory to path
sys.path.append(os.getcwd())

def generate_vocal_song():
    """Generate a song with vocals"""
    print("="*80)
    print("DiffRhythm Vocal Generation Test")
    print("="*80)
    
    try:
        # Import required modules
        print("\n[1/5] Importing modules and loading models...")
        from infer.infer_utils import prepare_model, get_lrc_token, get_style_prompt, get_negative_style_prompt, get_reference_latent
        from infer.infer import inference
        import torchaudio
        print("   ✓ Imports successful")
        
        # Load models using prepare_model
        print("   Loading models (this may take a few minutes)...")
        cfm, tokenizer, muq, vae = prepare_model(max_frames=2048, device='cpu')
        print("   ✓ All models loaded")
        
        # Load lyrics
        print("\n[2/5] Loading lyrics...")
        lyrics_path = "./output/test_vocals_wsl.lrc"
        with open(lyrics_path, 'r', encoding='utf-8') as f:
            lyrics = f.read()
        print(f"   ✓ Lyrics loaded ({len(lyrics.split())} words)")
        
        # Prepare generation
        print("\n[3/5] Preparing generation...")
        
        # Get tokens
        lrc_emb, start_time_val, end_frame, song_duration = get_lrc_token(
            max_frames=2048,
            text=lyrics,
            tokenizer=tokenizer,
            max_secs=95.0,
            device='cpu'
        )
        print("   ✓ Lyrics tokenized")
        
        # Get style
        style = get_style_prompt(muq, prompt="upbeat pop with singing vocals")
        print("   ✓ Style embedding created")
        
        # Get negative style
        negative_style = get_negative_style_prompt('cpu')
        print("   ✓ Negative style created")
        
        # Get reference
        latent_prompt, pred_frames = get_reference_latent(
            device='cpu',
            max_frames=2048,
            edit=False,
            pred_segments=None,
            ref_song=None,
            vae_model=vae
        )
        print("   ✓ Reference latent created")
        
        # Generate
        print("\n[4/5] Generating song...")
        print("   This will take 5-15 minutes on CPU...")
        print("   Style: Upbeat pop with singing vocals")
        print("   Duration: 95 seconds")
        
        start_gen = time.time()
        
        generated_songs = inference(
            cfm_model=cfm,
            vae_model=vae,
            cond=latent_prompt,
            text=lrc_emb,
            duration=end_frame,
            style_prompt=style,
            negative_style_prompt=negative_style,
            start_time=start_time_val,
            pred_frames=pred_frames,
            batch_infer_num=1,
            song_duration=song_duration,
            chunked=True
        )
        
        gen_time = time.time() - start_gen
        print(f"   ✓ Generation completed in {gen_time:.2f} seconds ({gen_time/60:.1f} minutes)")
        
        # Save
        print("\n[5/5] Saving output...")
        output_path = Path("./output/vocal_test_wsl_output.wav")
        torchaudio.save(str(output_path), generated_songs[0], sample_rate=44100)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        
        print("\n" + "="*80)
        print("✓ SUCCESS!")
        print("="*80)
        print(f"\nAudio file generated:")
        print(f"  Location: {output_path.absolute()}")
        print(f"  Size: {file_size_mb:.2f} MB")
        print(f"  Generation time: {gen_time:.2f} seconds ({gen_time/60:.1f} minutes)")
        
        print("\n" + "="*80)
        print("QUALITY ASSESSMENT")
        print("="*80)
        print("\nListen to the audio and evaluate:")
        print("  1. Vocal clarity and intelligibility")
        print("  2. Musical quality and coherence")
        print("  3. Rhythm and timing accuracy")
        print("  4. Overall production quality")
        print("  5. Lyrics alignment with music")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = generate_vocal_song()
    sys.exit(0 if success else 1)
