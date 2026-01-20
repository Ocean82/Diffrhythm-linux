#!/usr/bin/env python3
"""
Direct checkpoint pipeline run without user input
"""
import os
import sys
import time
import torch
import pickle
import json
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

class CheckpointPipeline:
    """Pipeline that saves intermediate results to disk and can resume from any step"""
    
    def __init__(self, checkpoint_dir="./output/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = 'cpu'
        self.max_frames = 2048
        
    def save_checkpoint(self, step_name, data):
        """Save checkpoint data to disk"""
        checkpoint_file = self.checkpoint_dir / f"{step_name}.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"  ✓ Checkpoint saved: {checkpoint_file}")
    
    def load_checkpoint(self, step_name):
        """Load checkpoint data from disk"""
        checkpoint_file = self.checkpoint_dir / f"{step_name}.pkl"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            print(f"  ✓ Checkpoint loaded: {checkpoint_file}")
            return data
        return None
    
    def checkpoint_exists(self, step_name):
        """Check if checkpoint exists"""
        checkpoint_file = self.checkpoint_dir / f"{step_name}.pkl"
        return checkpoint_file.exists()
    
    def clear_memory(self):
        """Clear memory and cache"""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def step1_lyrics(self, lyrics_text, audio_length=95, force_rerun=False):
        """Step 1: Process lyrics with checkpointing"""
        print("\n" + "="*50)
        print("STEP 1: LYRICS PROCESSING")
        print("="*50)
        
        if not force_rerun and self.checkpoint_exists("step1_lyrics"):
            print("  Found existing checkpoint, loading...")
            return self.load_checkpoint("step1_lyrics")
        
        print("  Loading tokenizer...")
        from infer.infer_utils import CNENTokenizer, get_lrc_token
        
        tokenizer = CNENTokenizer()
        
        print("  Processing lyrics...")
        lrc_emb, start_time_val, end_frame, song_duration = get_lrc_token(
            self.max_frames, lyrics_text, tokenizer, audio_length, self.device
        )
        
        # Save checkpoint
        checkpoint_data = {
            'lrc_emb': lrc_emb,
            'start_time_val': start_time_val,
            'end_frame': end_frame,
            'song_duration': song_duration
        }
        self.save_checkpoint("step1_lyrics", checkpoint_data)
        
        # Clear memory
        del tokenizer
        self.clear_memory()
        print("  ✓ Tokenizer unloaded")
        
        return checkpoint_data
    
    def step2_style(self, style_prompt, force_rerun=False):
        """Step 2: Process style with checkpointing"""
        print("\n" + "="*50)
        print("STEP 2: STYLE PROCESSING")
        print("="*50)
        
        if not force_rerun and self.checkpoint_exists("step2_style"):
            print("  Found existing checkpoint, loading...")
            return self.load_checkpoint("step2_style")
        
        print("  Loading MuQ model...")
        from muq import MuQMuLan
        from infer.infer_utils import get_style_prompt, get_negative_style_prompt
        
        muq = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir="./pretrained")
        muq = muq.to(self.device).eval()
        
        print("  Processing style...")
        style = get_style_prompt(muq, prompt=style_prompt)
        negative_style = get_negative_style_prompt(self.device)
        
        # Save checkpoint
        checkpoint_data = {
            'style': style,
            'negative_style': negative_style
        }
        self.save_checkpoint("step2_style", checkpoint_data)
        
        # Clear memory
        del muq
        self.clear_memory()
        print("  ✓ MuQ unloaded")
        
        return checkpoint_data
    
    def step3_generation(self, lyrics_data, style_data, steps=16, force_rerun=False):
        """Step 3: Generate latents with checkpointing"""
        print("\n" + "="*50)
        print("STEP 3: LATENT GENERATION")
        print("="*50)
        print(f"Using {steps} CFM steps for speed")
        
        if not force_rerun and self.checkpoint_exists("step3_generation"):
            print("  Found existing checkpoint, loading...")
            return self.load_checkpoint("step3_generation")
        
        print("  Loading CFM model (this takes ~3 minutes)...")
        from model import DiT, CFM
        from huggingface_hub import hf_hub_download
        from infer.infer_utils import load_checkpoint, get_reference_latent
        import json
        
        # Load CFM
        dit_ckpt_path = hf_hub_download(
            repo_id="ASLP-lab/DiffRhythm-1_2",
            filename="cfm_model.pt",
            cache_dir="./pretrained"
        )
        
        dit_config_path = "./config/diffrhythm-1b.json"
        with open(dit_config_path) as f:
            model_config = json.load(f)
        
        dit_model = DiT(**model_config["model"], max_frames=self.max_frames)
        cfm = CFM(
            transformer=dit_model,
            num_channels=model_config["model"]["mel_dim"],
            max_frames=self.max_frames
        )
        cfm = cfm.to(self.device)
        cfm = load_checkpoint(cfm, dit_ckpt_path, device=self.device, use_ema=False)
        
        # Load VAE briefly for reference latent
        print("  Loading VAE for reference latent...")
        vae_ckpt_path = hf_hub_download(
            repo_id="ASLP-lab/DiffRhythm-vae",
            filename="vae_model.pt",
            cache_dir="./pretrained",
        )
        vae = torch.jit.load(vae_ckpt_path, map_location="cpu").to(self.device)
        
        latent_prompt, pred_frames = get_reference_latent(
            self.device, self.max_frames, False, None, None, vae
        )
        
        # Unload VAE immediately
        del vae
        self.clear_memory()
        print("  ✓ VAE unloaded")
        
        print("  Starting CFM sampling (8-15 minutes on CPU)...")
        start_time = time.time()
        
        with torch.inference_mode():
            latents, trajectory = cfm.sample(
                cond=latent_prompt,
                text=lyrics_data['lrc_emb'],
                duration=lyrics_data['end_frame'],
                style_prompt=style_data['style'],
                negative_style_prompt=style_data['negative_style'],
                start_time=lyrics_data['start_time_val'],
                latent_pred_segments=pred_frames,
                song_duration=lyrics_data['song_duration'],
                steps=steps,
                cfg_strength=4.0,
                batch_infer_num=1
            )
        
        gen_time = time.time() - start_time
        print(f"  ✓ Generation completed in {gen_time:.1f}s ({gen_time/60:.1f} minutes)")
        
        # Save checkpoint
        checkpoint_data = {
            'latents': latents,
            'generation_time': gen_time
        }
        self.save_checkpoint("step3_generation", checkpoint_data)
        
        # Clear memory (free ~2GB)
        del cfm, dit_model
        self.clear_memory()
        print("  ✓ CFM unloaded (freed ~2GB)")
        
        return checkpoint_data
    
    def step4_decode(self, generation_data, output_path="./output/checkpoint_output.wav", force_rerun=False):
        """Step 4: Decode audio with checkpointing"""
        print("\n" + "="*50)
        print("STEP 4: AUDIO DECODING")
        print("="*50)
        
        if not force_rerun and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 100000:  # If file exists and is reasonable size
                print(f"  Found existing output: {output_path} ({file_size:,} bytes)")
                return True, output_path
        
        print("  Loading VAE...")
        from huggingface_hub import hf_hub_download
        from infer.infer_utils import decode_audio
        from einops import rearrange
        import torchaudio
        
        vae_ckpt_path = hf_hub_download(
            repo_id="ASLP-lab/DiffRhythm-vae",
            filename="vae_model.pt",
            cache_dir="./pretrained",
        )
        vae = torch.jit.load(vae_ckpt_path, map_location="cpu").to(self.device)
        
        print("  Decoding latents to audio...")
        outputs = []
        for i, latent in enumerate(generation_data['latents']):
            print(f"    Processing latent {i+1}/{len(generation_data['latents'])}...")
            
            latent = latent.to(torch.float32).transpose(1, 2)
            output = decode_audio(latent, vae, chunked=True)
            output = rearrange(output, "b d n -> d (b n)")
            
            # Safe normalization
            max_val = torch.max(torch.abs(output))
            if max_val < 1e-8:
                print(f"      ⚠ Silent audio detected")
                output = torch.zeros_like(output, dtype=torch.int16)
            else:
                output = (output * 0.95 / max_val).clamp(-1, 1).mul(32767).to(torch.int16)
                print(f"      ✓ Audio normalized (max: {max_val:.6f})")
            
            outputs.append(output.cpu())
        
        # Save audio
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        generated_song = outputs[0]
        torchaudio.save(output_path, generated_song, sample_rate=44100)
        
        # Verify
        success = False
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"  ✓ Audio saved: {output_path} ({file_size:,} bytes)")
            success = file_size > 100000
        
        # Clear memory
        del vae
        self.clear_memory()
        print("  ✓ VAE unloaded")
        
        return success, output_path

def main():
    """Run checkpoint pipeline automatically"""
    print("DiffRhythm Checkpoint Pipeline - Auto Run")
    print("="*45)
    print("Running automatically with test data...")
    print("")
    
    # Test data
    lyrics = """[00:00.00]Checkpoint test song
[00:05.00]Saving progress as we go
[00:10.00]Never lose our work again
[00:15.00]Memory efficient approach
[00:20.00]Perfect for CPU systems"""
    
    style = "energetic rock with powerful vocals"
    
    print("Test lyrics:")
    print(lyrics)
    print(f"\nStyle: {style}")
    print("")
    
    # Create and run pipeline
    pipeline = CheckpointPipeline()
    
    total_start = time.time()
    
    try:
        # Step 1: Lyrics
        print("Starting Step 1...")
        lyrics_data = pipeline.step1_lyrics(lyrics, force_rerun=False)
        
        # Step 2: Style
        print("Starting Step 2...")
        style_data = pipeline.step2_style(style, force_rerun=False)
        
        # Step 3: Generation (the slow part)
        print("Starting Step 3 (this will take 8-15 minutes)...")
        generation_data = pipeline.step3_generation(lyrics_data, style_data, steps=16, force_rerun=False)
        
        # Step 4: Decode
        print("Starting Step 4...")
        success, final_path = pipeline.step4_decode(generation_data, "./output/checkpoint_output.wav", force_rerun=False)
        
        total_time = time.time() - total_start
        
        print("\n" + "="*50)
        if success:
            print("✓ CHECKPOINT PIPELINE COMPLETE!")
        else:
            print("⚠ PIPELINE COMPLETED WITH ISSUES")
        print("="*50)
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Output: {final_path}")
        print(f"Checkpoints: {pipeline.checkpoint_dir}")
        
        if success:
            print("\n🎵 SUCCESS! Generated audio file:")
            print(f"   {final_path}")
            print("   Try playing this file to hear the result!")
        else:
            print("\n⚠ Check the output file - it may be silent but the pipeline worked")
        
        return success
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)