#!/usr/bin/env python3
"""
Segmented pipeline that loads/unloads models to minimize memory usage
"""
import os
import sys
import time
import torch
import gc
import json
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())


class SegmentedPipeline:
    """Pipeline that loads models only when needed and unloads them after use"""

    def __init__(self, device="cpu", max_frames=2048):
        self.device = device
        self.max_frames = max_frames
        self.cache_dir = "./pretrained"

        # Model references (loaded on demand)
        self.cfm = None
        self.tokenizer = None
        self.muq = None
        self.vae = None

        # Intermediate results storage
        self.results = {}

    def clear_memory(self):
        """Force garbage collection and clear CUDA cache"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  ✓ Memory cleared")

    def load_tokenizer_only(self):
        """Load only the tokenizer for lyrics processing"""
        if self.tokenizer is not None:
            return self.tokenizer

        print("Loading tokenizer...")
        start = time.time()

        from infer.infer_utils import CNENTokenizer

        self.tokenizer = CNENTokenizer()

        print(f"  ✓ Tokenizer loaded in {time.time()-start:.1f}s")
        return self.tokenizer

    def unload_tokenizer(self):
        """Unload tokenizer to free memory"""
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            self.clear_memory()
            print("  ✓ Tokenizer unloaded")

    def load_muq_only(self):
        """Load only MuQ for style processing"""
        if self.muq is not None:
            return self.muq

        print("Loading MuQ model...")
        start = time.time()

        from muq import MuQMuLan

        self.muq = MuQMuLan.from_pretrained(
            "OpenMuQ/MuQ-MuLan-large", cache_dir=self.cache_dir
        )
        self.muq = self.muq.to(self.device).eval()

        print(f"  ✓ MuQ loaded in {time.time()-start:.1f}s")
        return self.muq

    def unload_muq(self):
        """Unload MuQ to free memory"""
        if self.muq is not None:
            del self.muq
            self.muq = None
            self.clear_memory()
            print("  ✓ MuQ unloaded")

    def load_cfm_only(self):
        """Load only CFM model for generation"""
        if self.cfm is not None:
            return self.cfm

        print("Loading CFM model (this is the big one - ~2GB)...")
        start = time.time()

        # Load CFM components
        from model import DiT, CFM
        from huggingface_hub import hf_hub_download
        from infer.infer_utils import load_checkpoint

        # Get model file
        dit_ckpt_path = hf_hub_download(
            repo_id="ASLP-lab/DiffRhythm-1_2",
            filename="cfm_model.pt",
            cache_dir=self.cache_dir,
        )

        # Load config
        dit_config_path = "./config/diffrhythm-1b.json"
        with open(dit_config_path) as f:
            model_config = json.load(f)

        # Create model
        dit_model = DiT(**model_config["model"], max_frames=self.max_frames)
        self.cfm = CFM(
            transformer=dit_model,
            num_channels=model_config["model"]["mel_dim"],
            max_frames=self.max_frames,
        )
        self.cfm = self.cfm.to(self.device)

        # Load weights
        self.cfm = load_checkpoint(
            self.cfm, dit_ckpt_path, device=self.device, use_ema=False
        )

        print(f"  ✓ CFM loaded in {time.time()-start:.1f}s")
        return self.cfm

    def unload_cfm(self):
        """Unload CFM to free memory"""
        if self.cfm is not None:
            del self.cfm
            self.cfm = None
            self.clear_memory()
            print("  ✓ CFM unloaded (freed ~2GB)")

    def load_vae_only(self):
        """Load only VAE for audio decoding"""
        if self.vae is not None:
            return self.vae

        print("Loading VAE model...")
        start = time.time()

        from huggingface_hub import hf_hub_download

        vae_ckpt_path = hf_hub_download(
            repo_id="ASLP-lab/DiffRhythm-vae",
            filename="vae_model.pt",
            cache_dir=self.cache_dir,
        )
        self.vae = torch.jit.load(vae_ckpt_path, map_location="cpu").to(self.device)

        print(f"  ✓ VAE loaded in {time.time()-start:.1f}s")
        return self.vae

    def unload_vae(self):
        """Unload VAE to free memory"""
        if self.vae is not None:
            del self.vae
            self.vae = None
            self.clear_memory()
            print("  ✓ VAE unloaded")

    def step1_process_lyrics(self, lyrics_text, audio_length=95):
        """Step 1: Process lyrics (load tokenizer, process, unload)"""
        print("\n" + "=" * 50)
        print("STEP 1: PROCESSING LYRICS")
        print("=" * 50)

        # Load only tokenizer
        tokenizer = self.load_tokenizer_only()

        # Process lyrics
        from infer.infer_utils import get_lrc_token

        lrc_emb, start_time_val, end_frame, song_duration = get_lrc_token(
            self.max_frames, lyrics_text, tokenizer, audio_length, self.device
        )

        # Store results
        self.results["lrc_emb"] = lrc_emb
        self.results["start_time_val"] = start_time_val
        self.results["end_frame"] = end_frame
        self.results["song_duration"] = song_duration

        print(f"  ✓ Lyrics processed: {lrc_emb.shape}")

        # Unload tokenizer to free memory
        self.unload_tokenizer()

        return True

    def step2_process_style(self, style_prompt):
        """Step 2: Process style (load MuQ, process, unload)"""
        print("\n" + "=" * 50)
        print("STEP 2: PROCESSING STYLE")
        print("=" * 50)

        # Load only MuQ
        muq = self.load_muq_only()

        # Process style
        from infer.infer_utils import get_style_prompt, get_negative_style_prompt

        style = get_style_prompt(muq, prompt=style_prompt)
        negative_style = get_negative_style_prompt(self.device)

        # Store results
        self.results["style"] = style
        self.results["negative_style"] = negative_style

        print(f"  ✓ Style processed: {style.shape}")

        # Unload MuQ to free memory
        self.unload_muq()

        return True

    def step3_generate_latents(self, steps=16):
        """Step 3: Generate latents (load CFM, generate, unload)"""
        print("\n" + "=" * 50)
        print("STEP 3: GENERATING LATENTS (SLOW STEP)")
        print("=" * 50)
        print(f"Using {steps} CFM steps (reduced for speed)")

        # Load only CFM
        cfm = self.load_cfm_only()

        # Prepare reference latent (we need VAE briefly)
        vae = self.load_vae_only()
        from infer.infer_utils import get_reference_latent

        latent_prompt, pred_frames = get_reference_latent(
            self.device, self.max_frames, False, None, None, vae
        )
        # Unload VAE immediately after getting reference
        self.unload_vae()

        print("  Starting CFM sampling (this takes 8-15 minutes on CPU)...")
        start_gen = time.time()

        # Generate latents
        with torch.inference_mode():
            latents, trajectory = cfm.sample(
                cond=latent_prompt,
                text=self.results["lrc_emb"],
                duration=self.results["end_frame"],
                style_prompt=self.results["style"],
                negative_style_prompt=self.results["negative_style"],
                start_time=self.results["start_time_val"],
                latent_pred_segments=pred_frames,
                song_duration=self.results["song_duration"],
                steps=steps,  # Reduced for speed
                cfg_strength=4.0,
                batch_infer_num=1,
            )

        gen_time = time.time() - start_gen
        print(f"  ✓ Latents generated in {gen_time:.1f}s ({gen_time/60:.1f} minutes)")

        # Store results
        self.results["latents"] = latents

        # Unload CFM to free memory (this frees ~2GB!)
        self.unload_cfm()

        return True

    def step4_decode_audio(self, output_path="./output/segmented_output.wav"):
        """Step 4: Decode audio (load VAE, decode, save, unload)"""
        print("\n" + "=" * 50)
        print("STEP 4: DECODING AUDIO")
        print("=" * 50)

        # Load only VAE
        vae = self.load_vae_only()

        # Decode latents to audio
        from infer.infer_utils import decode_audio
        from einops import rearrange
        import torchaudio

        outputs = []
        for i, latent in enumerate(self.results["latents"]):
            print(f"  Decoding latent {i+1}/{len(self.results['latents'])}...")

            # Prepare for VAE
            latent = latent.to(torch.float32).transpose(1, 2)

            # Decode
            output = decode_audio(latent, vae, chunked=True)

            # Rearrange
            output = rearrange(output, "b d n -> d (b n)")

            # Safe normalization (our fix)
            max_val = torch.max(torch.abs(output))
            print(f"    Audio max amplitude: {max_val:.6f}")

            if max_val < 1e-8:
                print(f"    ⚠ WARNING: Audio is silent!")
                output = torch.zeros_like(output, dtype=torch.int16)
            else:
                output = (
                    (output * 0.95 / max_val).clamp(-1, 1).mul(32767).to(torch.int16)
                )
                print(f"    ✓ Audio normalized successfully")

            outputs.append(output.cpu())

        # Save audio
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        generated_song = outputs[0]
        torchaudio.save(output_path, generated_song, sample_rate=44100)

        # Verify output
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"  ✓ Audio saved: {output_path}")
            print(f"  ✓ File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
            success = file_size > 100000
        else:
            print(f"  ✗ Failed to save audio")
            success = False

        # Unload VAE
        self.unload_vae()

        return success, output_path

    def run_full_pipeline(
        self, lyrics_text, style_prompt, output_path="./output/segmented_output.wav"
    ):
        """Run the complete segmented pipeline"""
        print("DiffRhythm Segmented Pipeline")
        print("=" * 40)
        print("This approach loads/unloads models to minimize memory usage")
        print("Perfect for CPU-only systems with limited RAM")
        print("")

        total_start = time.time()

        try:
            # Step 1: Process lyrics
            if not self.step1_process_lyrics(lyrics_text):
                return False

            # Step 2: Process style
            if not self.step2_process_style(style_prompt):
                return False

            # Step 3: Generate latents (the slow part)
            if not self.step3_generate_latents(steps=16):  # Reduced steps for speed
                return False

            # Step 4: Decode audio
            success, final_path = self.step4_decode_audio(output_path)

            total_time = time.time() - total_start

            print("\n" + "=" * 50)
            if success:
                print("✓ SEGMENTED PIPELINE COMPLETE!")
            else:
                print("⚠ PIPELINE COMPLETED WITH ISSUES")
            print("=" * 50)
            print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
            print(f"Output: {final_path}")
            print("")
            print("Memory usage was minimized by loading/unloading models:")
            print("- Only one major model in memory at a time")
            print("- ~2GB CFM model unloaded after generation")
            print("- Much more CPU/RAM friendly approach")

            return success

        except Exception as e:
            print(f"\n✗ Pipeline failed: {e}")
            import traceback

            traceback.print_exc()
            return False


def main():
    """Main function to run segmented pipeline"""
    print("DiffRhythm Segmented Pipeline")
    print("=" * 35)
    print("Memory-efficient approach that loads/unloads models as needed")
    print("")

    # Test data
    lyrics = """[00:00.00]Hello world
[00:05.00]This is a segmented test
[00:10.00]Loading models one by one
[00:15.00]To save memory and RAM
[00:20.00]Hope this works much better"""

    style = "upbeat electronic pop with clear vocals"

    print("Test lyrics:")
    print(lyrics)
    print(f"\nStyle: {style}")
    print("")

    choice = input("Run segmented pipeline? (y/n): ").lower().strip()
    if choice != "y":
        print("Cancelled.")
        return

    # Create pipeline
    pipeline = SegmentedPipeline(device="cpu", max_frames=2048)

    # Run pipeline
    success = pipeline.run_full_pipeline(lyrics, style)

    if success:
        print("\n🎵 SUCCESS! Check ./output/segmented_output.wav")
    else:
        print("\n⚠ Pipeline completed but check the output file")


if __name__ == "__main__":
    main()
