#!/usr/bin/env python3
"""
Minimal generation test - bypass slow g2p loading where possible
"""
import sys
import os
import time
import torch
import json

sys.path.append(os.getcwd())

print("="*80)
print("Minimal DiffRhythm Generation Test")
print("="*80)

try:
    # Step 1: Load models WITHOUT tokenizer first
    print("\n[1/6] Loading CFM model...")
    start = time.time()
    
    from model import DiT, CFM
    from huggingface_hub import hf_hub_download
    
    dit_ckpt_path = hf_hub_download(
        repo_id="ASLP-lab/DiffRhythm-1_2",
        filename="cfm_model.pt",
        cache_dir="./pretrained"
    )
    
    dit_config_path = "./config/diffrhythm-1b.json"
    with open(dit_config_path) as f:
        model_config = json.load(f)
    
    device = "cpu"
    max_frames = 2048
    
    cfm = CFM(
        transformer=DiT(**model_config["model"], max_frames=max_frames),
        num_channels=model_config["model"]["mel_dim"],
        max_frames=max_frames
    )
    cfm = cfm.to(device)
    
    # Load checkpoint
    from infer.infer_utils import load_checkpoint
    cfm = load_checkpoint(cfm, dit_ckpt_path, device=device, use_ema=False)
    
    print(f"  ✓ CFM loaded in {time.time()-start:.2f}s")
    
    # Step 2: Load MuQ
    print("\n[2/6] Loading MuQ model...")
    start = time.time()
    
    from muq import MuQMuLan
    muq = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir="./pretrained")
    muq = muq.to(device).eval()
    
    print(f"  ✓ MuQ loaded in {time.time()-start:.2f}s")
    
    # Step 3: Load VAE
    print("\n[3/6] Loading VAE model...")
    start = time.time()
    
    vae_ckpt_path = hf_hub_download(
        repo_id="ASLP-lab/DiffRhythm-vae",
        filename="vae_model.pt",
        cache_dir="./pretrained",
    )
    vae = torch.jit.load(vae_ckpt_path, map_location="cpu").to(device)
    
    print(f"  ✓ VAE loaded in {time.time()-start:.2f}s")
    
    # Step 4: Load tokenizer (slow part)
    print("\n[4/6] Loading tokenizer (this takes ~90s due to Chinese g2p)...")
    start = time.time()
    
    from infer.infer_utils import CNENTokenizer
    tokenizer = CNENTokenizer()
    
    print(f"  ✓ Tokenizer loaded in {time.time()-start:.2f}s")
    
    # Step 5: Prepare generation
    print("\n[5/6] Preparing generation...")
    start = time.time()
    
    from infer.infer_utils import get_lrc_token, get_style_prompt, get_negative_style_prompt, get_reference_latent
    
    # Load lyrics
    with open("./output/test_vocals_wsl.lrc", 'r', encoding='utf-8') as f:
        lyrics = f.read()
    
    # Get tokens
    lrc_emb, start_time_val, end_frame, song_duration = get_lrc_token(
        max_frames=max_frames,
        text=lyrics,
        tokenizer=tokenizer,
        max_secs=95.0,
        device=device
    )
    
    # Get style
    style = get_style_prompt(muq, prompt="upbeat pop with singing vocals")
    negative_style = get_negative_style_prompt(device)
    
    # Get reference
    latent_prompt, pred_frames = get_reference_latent(
        device=device,
        max_frames=max_frames,
        edit=False,
        pred_segments=None,
        ref_song=None,
        vae_model=vae
    )
    
    print(f"  ✓ Prepared in {time.time()-start:.2f}s")
    
    # Step 6: Generate
    print("\n[6/6] Generating song...")
    print("  This will take 5-15 minutes on CPU...")
    start = time.time()
    
    from infer.infer import inference
    import torchaudio
    from pathlib import Path
    
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
    
    gen_time = time.time() - start
    print(f"  ✓ Generated in {gen_time:.2f}s ({gen_time/60:.1f} minutes)")
    
    # Save
    output_path = Path("./output/vocal_test_minimal_output.wav")
    torchaudio.save(str(output_path), generated_songs[0], sample_rate=44100)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print("\n" + "="*80)
    print("✓ SUCCESS!")
    print("="*80)
    print(f"\nAudio file: {output_path.absolute()}")
    print(f"Size: {file_size_mb:.2f} MB")
    print(f"Generation time: {gen_time:.2f}s ({gen_time/60:.1f} min)")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
