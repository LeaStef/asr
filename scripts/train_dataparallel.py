#!/usr/bin/env python3

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.config import create_config_from_dict
from models.asr_model import create_model
from data.dataset import create_dataloaders
from training.trainer import Trainer
from training.utils import load_checkpoint
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description="DataParallel ASR Training (No NCCL)")
    parser.add_argument("--preset", default="rtx6000-2gpu", help="Training preset")
    parser.add_argument("--output-dir", default="./outputs", help="Output directory")
    parser.add_argument("--dataset", default="gigaspeech", help="Dataset")
    parser.add_argument("--subset", default="m", help="Dataset subset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--resume", help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    print("🚫 NCCL-FREE DataParallel Training")
    print("=" * 50)
    print(f"GPUs available: {torch.cuda.device_count()}")
    print(f"Using preset: {args.preset}")
    print(f"Dataset: {args.dataset} ({args.subset})")
    print(f"Resume from: {args.resume}")
    print("=" * 50)
    
    # Create config - using single GPU config but will use DataParallel
    config_dict = {
        'model': {
            'encoder': {
                'input_size': 80,
                'hidden_size': 512,
                'memory_size': 256,
                'num_lmu_layers': 4,
                'theta': 1000,
                'dropout': 0.1,
                'use_fft_lmu': False,
                'use_attention': True,
                'num_attention_heads': 8,
                'use_downsampling': False,
                'downsample_factor': 2
            },
            'decoder': {
                'vocab_size': 32
            }
        },
        'data': {
            'dataset': args.dataset,
            'subset': args.subset,
            'save_dir': "./data/gigaspeech",
            'sample_rate': 16000,
            'n_mels': 80,
            'max_seq_len': 500,
            'augment': True,
            'num_workers': 8  # Reduced for single process
        },
        'training': {
            'batch_size': 32,  # Per GPU batch size (will be effective 64 with 2 GPUs)
            'lr': 2e-3,  # Slightly higher LR for larger effective batch
            'max_epochs': args.epochs,
            'patience': 10,
            'mixed_precision': True,
            'gradient_clip_norm': 1.0,
            'accumulate_grad_batches': 1,
            'warmup_steps': 1000,
            'weight_decay': 0.01,
            'log_interval': 50
        }
    }
    
    config = create_config_from_dict(OmegaConf.create(config_dict))
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, vocab = create_dataloaders(config)
    config.model.decoder.vocab_size = vocab['vocab_size']
    
    # Create model
    print("Creating model...")
    model = create_model(config.model)
    
    # Wrap in DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"🔄 Wrapping model in DataParallel for {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        # Adjust batch size logging
        effective_batch_size = config.training.batch_size * torch.cuda.device_count()
        print(f"Effective batch size: {effective_batch_size}")
    
    model = model.cuda()
    
    # Create trainer
    trainer = Trainer(model, config.training, config.data, log_dir=args.output_dir)
    
    # Resume if checkpoint provided
    if args.resume:
        print(f"Resuming from: {args.resume}")
        checkpoint = load_checkpoint(args.resume, model, trainer.optimizer)
        trainer.epoch = checkpoint.get('epoch', 0)
        print(f"Resuming from epoch {trainer.epoch}")
    
    # Start training
    print("🚀 Starting DataParallel training (NO NCCL/distributed)")
    trainer.fit(train_loader, val_loader, vocab)


if __name__ == "__main__":
    main()