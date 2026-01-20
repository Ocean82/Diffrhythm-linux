#!/usr/bin/env python3
"""
LoRA Fine-Tuning Script for DiffRhythm

This script enables fine-tuning the DiffRhythm model using LoRA (Low-Rank Adaptation),
which requires significantly less GPU memory and training time than full fine-tuning.

Requirements:
    pip install peft accelerate transformers

Usage:
    python train/train_lora.py --config config/lora_config.json

Or with accelerate for multi-GPU:
    accelerate launch train/train_lora.py --config config/lora_config.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import CFM, DiT
from dataset.dataset import DiffusionDataset

# Check for PEFT
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("WARNING: PEFT not installed. Install with: pip install peft")


def create_lora_model(base_model, lora_config):
    """
    Apply LoRA to the DiT transformer

    Args:
        base_model: CFM model with DiT transformer
        lora_config: LoRA configuration dict

    Returns:
        Model with LoRA adapters applied
    """
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT library required. Install with: pip install peft")

    # Default target modules for LlamaDecoderLayer
    default_targets = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",      # FFN
    ]

    config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        target_modules=lora_config.get("target_modules", default_targets),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        bias=lora_config.get("bias", "none"),
        modules_to_save=lora_config.get("modules_to_save", None),
    )

    # Apply LoRA to transformer
    base_model.transformer = get_peft_model(base_model.transformer, config)

    # Print trainable parameters
    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"LoRA Configuration Applied")
    print(f"{'='*60}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %:          {100 * trainable_params / total_params:.2f}%")
    print(f"LoRA rank (r):        {config.r}")
    print(f"LoRA alpha:           {config.lora_alpha}")
    print(f"Target modules:       {config.target_modules}")
    print(f"{'='*60}\n")

    return base_model


def load_base_model(model_config_path, checkpoint_path, max_frames=2048, device="cuda"):
    """Load the base DiffRhythm model"""
    from huggingface_hub import hf_hub_download

    # Load model config
    with open(model_config_path) as f:
        model_config = json.load(f)

    # Create model
    cfm = CFM(
        transformer=DiT(**model_config["model"], max_frames=max_frames),
        num_channels=model_config["model"]["mel_dim"],
        max_frames=max_frames
    )

    # Load checkpoint
    if checkpoint_path:
        if checkpoint_path.startswith("ASLP-lab/"):
            # Download from HuggingFace (use configured cache directory)
            repo_id = checkpoint_path
            cache_dir = "D:\\_hugging-face"
            ckpt_path = hf_hub_download(repo_id=repo_id, filename="cfm_model.pt", cache_dir=cache_dir)
        else:
            ckpt_path = checkpoint_path

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        if "model_state_dict" in checkpoint:
            cfm.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            cfm.load_state_dict(checkpoint, strict=False)

        print(f"Loaded checkpoint from {ckpt_path}")

    return cfm.to(device)


def train_lora(
    model,
    train_dataloader,
    optimizer,
    scheduler,
    num_epochs,
    save_dir,
    device="cuda",
    grad_accumulation_steps=1,
    max_grad_norm=1.0,
    save_every=1000,
    log_every=100,
):
    """
    Training loop for LoRA fine-tuning

    Args:
        model: CFM model with LoRA adapters
        train_dataloader: Training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        save_dir: Directory to save checkpoints
        device: Training device
        grad_accumulation_steps: Gradient accumulation
        max_grad_norm: Gradient clipping
        save_every: Save checkpoint every N steps
        log_every: Log every N steps
    """
    os.makedirs(save_dir, exist_ok=True)

    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch"
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            text_inputs = batch["lrc"].to(device)
            mel_spec = batch["latent"].permute(0, 2, 1).to(device)
            mel_lengths = batch["latent_lengths"].to(device)
            style_prompt = batch["prompt"].to(device)
            start_time = batch["start_time"].to(device)

            # Forward pass
            loss, cond, pred = model(
                mel_spec,
                text=text_inputs,
                lens=mel_lengths,
                style_prompt=style_prompt,
                start_time=start_time,
            )

            # Backward pass with gradient accumulation
            loss = loss / grad_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % grad_accumulation_steps == 0:
                # Gradient clipping
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % log_every == 0:
                    lr = scheduler.get_last_lr()[0]
                    progress_bar.set_postfix(
                        loss=f"{loss.item() * grad_accumulation_steps:.4f}",
                        lr=f"{lr:.2e}",
                        step=global_step
                    )

                # Save checkpoint
                if global_step % save_every == 0:
                    save_lora_checkpoint(model, save_dir, global_step)

            epoch_loss += loss.item() * grad_accumulation_steps
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

        # Save end of epoch checkpoint
        save_lora_checkpoint(model, save_dir, f"epoch_{epoch + 1}")

    # Save final checkpoint
    save_lora_checkpoint(model, save_dir, "final")
    print(f"Training complete! LoRA adapters saved to {save_dir}")


def save_lora_checkpoint(model, save_dir, step):
    """Save only the LoRA adapter weights"""
    adapter_path = os.path.join(save_dir, f"lora_adapter_{step}")

    # Save LoRA adapter
    model.transformer.save_pretrained(adapter_path)

    print(f"Saved LoRA adapter to {adapter_path}")


def load_lora_checkpoint(model, adapter_path):
    """Load LoRA adapter weights"""
    from peft import PeftModel

    model.transformer = PeftModel.from_pretrained(
        model.transformer,
        adapter_path
    )
    print(f"Loaded LoRA adapter from {adapter_path}")
    return model


def merge_lora_weights(model):
    """Merge LoRA weights into base model for inference"""
    model.transformer = model.transformer.merge_and_unload()
    print("LoRA weights merged into base model")
    return model


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning for DiffRhythm")
    parser.add_argument("--model-config", default="config/diffrhythm-1b.json")
    parser.add_argument("--checkpoint", default="ASLP-lab/DiffRhythm-1_2")
    parser.add_argument("--data-path", default="dataset/train.scp")
    parser.add_argument("--output-dir", default="ckpts/lora")
    parser.add_argument("--max-frames", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)  # Reduced for memory
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=8)  # Reduced for efficiency
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--grad-accumulation", type=int, default=8)  # Increased to compensate for small batch
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--verify-only", action="store_true", help="Only verify setup, don't train")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        print("WARNING: Training on CPU will be very slow!")

    # Load base model
    print("Loading base model...")
    model = load_base_model(
        args.model_config,
        args.checkpoint,
        args.max_frames,
        device
    )

    # Apply LoRA
    print("Applying LoRA adapters...")
    lora_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    model = create_lora_model(model, lora_config)

    # Create dataset and dataloader
    print("Loading dataset...")
    dataset = DiffusionDataset(
        args.data_path,
        max_frames=args.max_frames,
        min_frames=512,
        sampling_rate=44100,
        downsample_rate=2048,
        precision="fp32" if device == "cpu" else "fp16"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=dataset.custom_collate_fn,
        pin_memory=True if device == "cuda" else False
    )

    print(f"Dataset size: {len(dataset)} samples")

    if args.verify_only:
        print("\n" + "="*60)
        print("VERIFICATION COMPLETE")
        print("="*60)
        print("OK Model loaded successfully")
        print("OK LoRA adapters applied")
        print("OK Dataset loaded")
        print(f"OK Ready to train with {len(dataset)} samples")
        print("\nTo start training, run without --verify-only")
        print(f"Estimated GPU memory needed: ~12-16 GB")
        print(f"Estimated training time: {args.epochs * len(dataset) * 2 // 60} minutes on GPU")
        return

    # Optimizer and scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    total_steps = len(dataloader) * args.epochs // args.grad_accumulation
    warmup_steps = max(total_steps // 10, 1)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=max(total_steps, 1),
        pct_start=0.1,
        anneal_strategy="cos"
    )

    # Train
    print("\nStarting LoRA training...")
    train_lora(
        model,
        dataloader,
        optimizer,
        scheduler,
        num_epochs=args.epochs,
        save_dir=args.output_dir,
        device=device,
        grad_accumulation_steps=args.grad_accumulation,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
