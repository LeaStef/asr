#!/usr/bin/env python3

import os
import sys
import torch
from pathlib import Path
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.config import ModelConfig, DataConfig, TrainingConfig, Config
from models.asr_model import create_model
from data.dataset import create_dataloaders
from training.trainer import Trainer
from training.utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Single GPU ASR Training (Debug NaN issues)")
    parser.add_argument("--output-dir", default="./outputs", help="Output directory")
    parser.add_argument("--dataset", default="gigaspeech", help="Dataset")
    parser.add_argument("--subset", default="xs", help="Dataset subset (ultra small for testing)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (quick test)")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (ultra conservative)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (ultra small)")
    parser.add_argument("--disable-mixed-precision", action="store_true", default=True, help="Disable mixed precision by default")
    
    args = parser.parse_args()
    
    print("🔧 Single GPU Training (Debug Mode)")
    print("=" * 50)
    print(f"GPUs available: {torch.cuda.device_count()}")
    print(f"Dataset: {args.dataset} ({args.subset})")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Mixed precision: {not args.disable_mixed_precision}")
    print(f"Resume from: {args.resume}")
    print("=" * 50)
    
    # Create config with conservative settings
    model_config = ModelConfig(
        input_size=80,
        hidden_size=512,
        memory_size=256,
        num_lmu_layers=4,
        theta=1000,
        dropout=0.1,
        use_fft_lmu=False,
        use_attention=True,
        num_attention_heads=8,
        use_downsampling=False,
        downsample_factor=2,
        vocab_size=32  # Will be updated after loading data
    )
    
    data_config = DataConfig(
        dataset=args.dataset,
        subset=args.subset,
        save_dir="./data/gigaspeech",
        sample_rate=16000,
        n_mels=80,
        max_seq_len=500,
        augment=True,
        num_workers=4  # Reduced for debugging
    )
    
    training_config = TrainingConfig(
        batch_size=args.batch_size,  # Conservative batch size
        lr=args.lr,  # Reduced learning rate
        max_epochs=args.epochs,
        patience=10,
        mixed_precision=not args.disable_mixed_precision,
        gradient_clip_norm=1.0,
        accumulate_grad_batches=1,
        warmup_steps=1000,
        weight_decay=0.01,
        log_interval=10  # More frequent logging
    )
    
    config = Config(
        model=model_config,
        data=data_config,
        training=training_config
    )
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, vocab = create_dataloaders(config)
    config.model.vocab_size = vocab['vocab_size']
    
    # Create model
    print("Creating model...")
    model = create_model(config.model)
    
    # Move model to GPU (single GPU only)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model on device: {next(model.parameters()).device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(model, config.training, config.data, device, log_dir=args.output_dir)
    
    # Resume if checkpoint provided
    if args.resume:
        print(f"Resuming from: {args.resume}")
        checkpoint = load_checkpoint(args.resume, model, trainer.optimizer)
        trainer.epoch = checkpoint.get('epoch', 0)
        print(f"Resuming from epoch {trainer.epoch}")
        
        # Debug checkpoint loading
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        if 'model_state_dict' in checkpoint:
            print("✅ Model state loaded")
        if 'optimizer_state_dict' in checkpoint:
            print("✅ Optimizer state loaded")
        
        # Check for NaN weights
        nan_weights = 0
        total_weights = 0
        for name, param in model.named_parameters():
            total_weights += param.numel()
            if torch.isnan(param).any():
                nan_weights += torch.isnan(param).sum().item()
                print(f"⚠️  NaN weights found in {name}: {torch.isnan(param).sum().item()}")
        
        print(f"Weight check: {nan_weights}/{total_weights} NaN weights")
    
    # Start training
    print("🚀 Starting single GPU training (debug mode)")
    trainer.fit(train_loader, val_loader, vocab)


if __name__ == "__main__":
    main()