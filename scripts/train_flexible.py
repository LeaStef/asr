#!/usr/bin/env python3
"""
Flexible training script with argparse support for easy batch size experimentation.

Usage Examples:
    # Default configuration
    python scripts/train_flexible.py
    
    # Custom batch size
    python scripts/train_flexible.py --batch-size 64
    
    # Multi-GPU with custom batch size
    torchrun --nproc_per_node=2 scripts/train_flexible.py --batch-size 128
    
    # L40S GPU optimization
    python scripts/train_flexible.py --batch-size 64 --lr 4e-3 --preset l40s-1gpu
    torchrun --nproc_per_node=2 scripts/train_flexible.py --batch-size 128 --lr 8e-3 --preset l40s-2gpu
    
    # Full customization
    python scripts/train_flexible.py --batch-size 32 --lr 2e-3 --epochs 10 --mixed-precision
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.config import create_config_from_dict
from models.asr_model import create_model
from data.dataset import create_distributed_dataloaders
from training.distributed import DistributedTrainer
from omegaconf import OmegaConf


def setup_distributed():
    """Setup distributed training for torchrun or single GPU"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Running with torchrun
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Initialize process group
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        # Single GPU mode
        return 0, 1, 0


def get_preset_config(preset: str) -> dict:
    """Get preset configurations for different GPU setups"""
    presets = {
        'l40s-1gpu': {
            'batch_size': 64,
            'lr': 4e-3,
            'epochs': 35,
            'num_workers': 6,
            'mixed_precision': True,
        },
        'l40s-2gpu': {
            'batch_size': 128,
            'lr': 8e-3,
            'epochs': 25,
            'num_workers': 8,
            'mixed_precision': True,
        },
        'a100-1gpu': {
            'batch_size': 48,
            'lr': 3e-3,
            'epochs': 40,
            'num_workers': 6,
            'mixed_precision': True,
        },
        'a100-2gpu': {
            'batch_size': 96,
            'lr': 6e-3,
            'epochs': 30,
            'num_workers': 8,
            'mixed_precision': True,
        },
        'rtx6000-1gpu': {
            'batch_size': 64,  # Reduced for better performance
            'lr': 2e-3,        # Slightly reduced LR
            'epochs': 30,
            'num_workers': 12, # Increased workers
            'mixed_precision': True,
        },
        'rtx6000-2gpu': {
            'batch_size': 80,  # Reduced from 96 for better performance
            'lr': 3e-3,        # Reduced from 5e-3
            'epochs': 25,
            'num_workers': 16, # Increased from 12
            'mixed_precision': True,
        },
        'default': {
            'batch_size': 16,
            'lr': 1e-3,
            'epochs': 50,
            'num_workers': 4,
            'mixed_precision': True,
        }
    }
    
    return presets.get(preset, presets['default'])


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='PyTorch LMU-ASR Training with Flexible Batch Size',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Core training parameters
    parser.add_argument('--batch-size', '-b', type=int, default=None,
                        help='Total batch size across all GPUs (default: use config)')
    parser.add_argument('--lr', '--learning-rate', type=float, default=None,
                        help='Learning rate (default: use config)')
    parser.add_argument('--epochs', '-e', type=int, default=None,
                        help='Number of training epochs (default: use config)')
    
    # GPU optimization presets
    parser.add_argument('--preset', choices=['l40s-1gpu', 'l40s-2gpu', 'a100-1gpu', 'a100-2gpu', 'rtx6000-1gpu', 'rtx6000-2gpu', 'default'], default='default',
                        help='Use optimized preset for specific GPU configuration')
    
    # Training options
    parser.add_argument('--mixed-precision', action='store_true', default=None,
                        help='Enable mixed precision training')
    parser.add_argument('--no-mixed-precision', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--gradient-clip', type=float, default=None,
                        help='Gradient clipping norm (default: 1.0)')
    
    # Data loading
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of data loading workers per GPU')
    
    # Configuration
    parser.add_argument('--config', '-c', default='base_config',
                        help='Base configuration file name (default: base_config)')
    parser.add_argument('--dataset', choices=['gigaspeech', 'librispeech'], default=None,
                        help='Dataset to use')
    parser.add_argument('--subset', default=None,
                        help='Dataset subset (e.g., xs, s, m for GigaSpeech)')
    
    # Output and logging
    parser.add_argument('--output-dir', default='./outputs',
                        help='Output directory for logs and checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    
    # Debugging
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (single epoch, verbose logging)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Test configuration without training')
    
    return parser.parse_args()


def apply_args_to_config(config, args, rank, world_size):
    """Apply command line arguments to configuration"""
    
    # Apply preset if specified
    if args.preset:
        preset = get_preset_config(args.preset)
        print(f"📋 Applying preset: {args.preset}")
        for key, value in preset.items():
            if key == 'batch_size':
                config.training.batch_size = value
            elif key == 'lr':
                config.training.lr = value
            elif key == 'epochs':
                config.training.max_epochs = value
            elif key == 'num_workers':
                config.data.num_workers = value
            elif key == 'mixed_precision':
                config.training.mixed_precision = value
    
    # Apply individual arguments (override preset)
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
        
    if args.lr is not None:
        config.training.lr = args.lr
        
    if args.epochs is not None:
        config.training.max_epochs = args.epochs
        
    if args.mixed_precision is not None:
        config.training.mixed_precision = True
    elif args.no_mixed_precision:
        config.training.mixed_precision = False
        
    if args.gradient_clip is not None:
        config.training.gradient_clip_norm = args.gradient_clip
        
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
        
    if args.dataset is not None:
        config.data.dataset = args.dataset
        
    if args.subset is not None:
        config.data.subset = args.subset
    
    # Debug mode adjustments
    if args.debug:
        config.training.max_epochs = 1
        config.training.patience = 1
        print("🐛 Debug mode: Single epoch training")
    
    # Calculate per-GPU batch size
    per_gpu_batch_size = config.training.batch_size // world_size
    if config.training.batch_size % world_size != 0:
        print(f"⚠️  Warning: Batch size {config.training.batch_size} not evenly divisible by {world_size} GPUs")
        per_gpu_batch_size = config.training.batch_size // world_size
        config.training.batch_size = per_gpu_batch_size * world_size
        print(f"   Adjusted to: {config.training.batch_size} total ({per_gpu_batch_size} per GPU)")
    
    return config, per_gpu_batch_size


def print_training_info(config, args, rank, world_size, per_gpu_batch_size):
    """Print training configuration info"""
    if rank == 0:  # Only print from main process
        print("\n" + "="*60)
        print("🚀 PyTorch LMU-ASR Training Configuration")
        print("="*60)
        
        print(f"📊 Training Setup:")
        print(f"   GPUs: {world_size}")
        print(f"   Total batch size: {config.training.batch_size}")
        print(f"   Batch size per GPU: {per_gpu_batch_size}")
        print(f"   Learning rate: {config.training.lr}")
        print(f"   Max epochs: {config.training.max_epochs}")
        print(f"   Mixed precision: {config.training.mixed_precision}")
        
        print(f"\n📁 Data:")
        print(f"   Dataset: {config.data.dataset}")
        if hasattr(config.data, 'subset'):
            print(f"   Subset: {config.data.subset}")
        print(f"   Data workers: {config.data.num_workers}")
        
        if args.preset:
            print(f"\n🎯 Using preset: {args.preset}")
            
        # Memory estimation
        if world_size == 2 and per_gpu_batch_size >= 64:
            print(f"\n💾 Memory: Optimized for L40S GPUs (expected ~75% usage)")
        elif per_gpu_batch_size >= 32:
            print(f"\n💾 Memory: High batch size detected (monitor GPU memory)")
            
        print("="*60)


def main():
    """Main training function"""
    args = parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Load base configuration
    try:
        cfg = OmegaConf.load(f'configs/{args.config}.yaml')
    except FileNotFoundError:
        if rank == 0:
            print(f"❌ Configuration file not found: configs/{args.config}.yaml")
            print(f"Available configs: {list(Path('configs').glob('*.yaml'))}")
        return 1
    
    # Create config object
    config = create_config_from_dict(cfg)
    
    # Apply command line arguments
    config, per_gpu_batch_size = apply_args_to_config(config, args, rank, world_size)
    
    # Print configuration
    print_training_info(config, args, rank, world_size, per_gpu_batch_size)
    
    if args.dry_run:
        if rank == 0:
            print("✅ Dry run completed successfully!")
        return 0
    
    try:
        # Create model and move to GPU
        device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        model = create_model(config.model).to(device)
        
        # Wrap model with DDP if distributed
        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
            if rank == 0:
                print(f"✅ Model wrapped with DDP")
        
        # Create distributed data loaders
        train_loader, val_loader, vocab = create_distributed_dataloaders(
            config, rank, world_size
        )
        
        # Update vocab size in config
        config.model.vocab_size = vocab['vocab_size']
        
        # Create distributed trainer
        trainer = DistributedTrainer(
            model=model,
            config=config,
            rank=rank,
            world_size=world_size,
            log_dir=args.output_dir
        )
        
        if rank == 0:
            print(f"🎯 Starting training...")
        
        # Resume from checkpoint if specified
        if args.resume:
            trainer.resume_training(args.resume, train_loader, val_loader, vocab)
        else:
            trainer.fit(train_loader, val_loader, vocab)
        
        if rank == 0:
            print("🎉 Training completed successfully!")
            
    except KeyboardInterrupt:
        if rank == 0:
            print("\n⏹️  Training interrupted by user")
        return 1
    except Exception as e:
        if rank == 0:
            print(f"❌ Training failed: {str(e)}")
        raise
    finally:
        # Cleanup distributed training
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)