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
    python scripts/train_flexible.py --batch-size 64 --lr 4e-4 --preset l40s-1gpu
    torchrun --nproc_per_node=2 scripts/train_flexible.py --batch-size 128 --lr 8e-4 --preset l40s-2gpu
    
    # Full customization
    python scripts/train_flexible.py --batch-size 32 --lr 2e-4 --epochs 10 --mixed-precision
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist
import psutil
import time
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.config import create_config_from_dict
from models.asr_model import create_model
from data.dataset import create_distributed_dataloaders
from training.distributed import DistributedTrainer
from omegaconf import OmegaConf


def get_memory_info(device_id=None):
    """Get comprehensive memory information for GPU and system"""
    memory_info = {}
    
    # GPU Memory
    if torch.cuda.is_available():
        if device_id is not None:
            torch.cuda.set_device(device_id)
        
        # Current GPU memory
        gpu_memory = torch.cuda.memory_stats()
        memory_info['gpu'] = {
            'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
            'max_reserved_gb': torch.cuda.max_memory_reserved() / 1024**3,
            'total_gb': torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**3
        }
        
        # Calculate utilization
        memory_info['gpu']['utilization_percent'] = (memory_info['gpu']['allocated_gb'] / memory_info['gpu']['total_gb']) * 100
    
    # System Memory
    memory_info['system'] = {
        'available_gb': psutil.virtual_memory().available / 1024**3,
        'used_gb': psutil.virtual_memory().used / 1024**3,
        'total_gb': psutil.virtual_memory().total / 1024**3,
        'percent_used': psutil.virtual_memory().percent
    }
    
    return memory_info


def print_memory_usage(prefix="", rank=0, device_id=None):
    """Print formatted memory usage information"""
    if rank != 0:  # Only print from rank 0 in distributed training
        return
    
    try:
        mem_info = get_memory_info(device_id)
        
        print(f"\n{'='*50}")
        print(f"🔍 {prefix} Memory Usage")
        print(f"{'='*50}")
        
        # GPU Memory
        if 'gpu' in mem_info:
            gpu = mem_info['gpu']
            print(f"🖥️  GPU Memory:")
            print(f"   Allocated:  {gpu['allocated_gb']:.2f} GB ({gpu['utilization_percent']:.1f}% of {gpu['total_gb']:.1f} GB)")
            print(f"   Reserved:   {gpu['reserved_gb']:.2f} GB")
            print(f"   Peak Alloc: {gpu['max_allocated_gb']:.2f} GB")
            print(f"   Peak Rsrvd: {gpu['max_reserved_gb']:.2f} GB")
            
            # Memory warnings
            if gpu['utilization_percent'] > 90:
                print(f"   ⚠️  WARNING: GPU memory usage > 90%")
            elif gpu['utilization_percent'] > 75:
                print(f"   ⚠️  CAUTION: GPU memory usage > 75%")
        
        # System Memory
        sys_mem = mem_info['system']
        print(f"\n🖥️  System Memory:")
        print(f"   Used:      {sys_mem['used_gb']:.2f} GB ({sys_mem['percent_used']:.1f}%)")
        print(f"   Available: {sys_mem['available_gb']:.2f} GB")
        print(f"   Total:     {sys_mem['total_gb']:.2f} GB")
        
        if sys_mem['percent_used'] > 90:
            print(f"   ⚠️  WARNING: System memory usage > 90%")
        
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"❌ Error getting memory info: {e}")


def setup_memory_monitoring():
    """Setup memory monitoring and clear cache"""
    if torch.cuda.is_available():
        # Clear cache and reset peak memory stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("🧹 Cleared CUDA cache and reset memory stats")


def verify_model_consistency(model, rank, world_size):
    """Verify that model parameters are consistent across all ranks"""
    if world_size == 1:
        return True
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if rank == 0:
        print(f"🔍 Rank {rank}: Model has {param_count} total parameters ({trainable_params} trainable)")
    
    # Create tensors with parameter counts
    param_tensor = torch.tensor([param_count, trainable_params], dtype=torch.long).cuda()
    param_tensors = [torch.zeros_like(param_tensor) for _ in range(world_size)]
    
    # Gather parameter counts from all ranks
    dist.all_gather(param_tensors, param_tensor)
    
    if rank == 0:
        print(f"📊 Parameter counts across ranks:")
        for i, pt in enumerate(param_tensors):
            total, trainable = pt[0].item(), pt[1].item()
            print(f"   Rank {i}: {total} total ({trainable} trainable)")
            
        # Check consistency
        base_total, base_trainable = param_tensors[0][0].item(), param_tensors[0][1].item()
        for i, pt in enumerate(param_tensors[1:], 1):
            if pt[0].item() != base_total or pt[1].item() != base_trainable:
                print(f"❌ ERROR: Rank {i} parameter count mismatch!")
                print(f"   Expected: {base_total} total ({base_trainable} trainable)")
                print(f"   Got: {pt[0].item()} total ({pt[1].item()} trainable)")
                return False
        
        print(f"✅ Model parameter counts consistent across all ranks")
    
    return True


def setup_distributed():
    """Setup distributed training for torchrun or single GPU"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Running with torchrun
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Initialize process group with configurable backend
        backend = os.environ.get('TORCH_DISTRIBUTED_BACKEND', 'nccl')
        print(f"Initializing distributed training with backend: {backend}")
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)
        
        # Synchronize random seeds across all processes
        import random
        import numpy as np
        
        # Use rank 0's random state as the base seed
        if rank == 0:
            base_seed = random.randint(0, 2**32 - 1)
            seed_tensor = torch.tensor(base_seed, dtype=torch.long).cuda()
        else:
            seed_tensor = torch.tensor(0, dtype=torch.long).cuda()
        
        # Broadcast seed from rank 0 to all ranks
        dist.broadcast(seed_tensor, src=0)
        synchronized_seed = seed_tensor.item()
        
        # Set synchronized seeds for all random number generators
        torch.manual_seed(synchronized_seed)
        torch.cuda.manual_seed(synchronized_seed)
        torch.cuda.manual_seed_all(synchronized_seed)
        np.random.seed(synchronized_seed)
        random.seed(synchronized_seed)
        
        if rank == 0:
            print(f"🎲 Synchronized random seed across all ranks: {synchronized_seed}")
        
        return rank, world_size, local_rank
    else:
        # Single GPU mode
        import random
        import numpy as np
        base_seed = 42  # Fixed seed for reproducibility
        torch.manual_seed(base_seed)
        torch.cuda.manual_seed_all(base_seed)
        np.random.seed(base_seed)
        random.seed(base_seed)
        return 0, 1, 0


def get_preset_config(preset: str) -> dict:
    """Get preset configurations for different GPU setups"""
    presets = {
        'l40s-1gpu': {
            'batch_size': 64,
            'lr': 4e-4,  # Reduced for better CTC convergence
            'epochs': 35,
            'num_workers': 6,
            'mixed_precision': True,
        },
        'l40s-2gpu': {
            'batch_size': 128,
            'lr': 8e-4,  # Reduced for better CTC convergence
            'epochs': 25,
            'num_workers': 8,
            'mixed_precision': True,
        },
        'a100-1gpu': {
            'batch_size': 48,
            'lr': 3e-4,  # Reduced for better CTC convergence
            'epochs': 40,
            'num_workers': 6,
            'mixed_precision': True,
        },
        'a100-2gpu': {
            'batch_size': 96,
            'lr': 6e-4,  # Reduced for better CTC convergence
            'epochs': 30,
            'num_workers': 8,
            'mixed_precision': True,
        },
        'rtx6000-1gpu': {
            'batch_size': 24,  # Conservative batch size for stability
            'lr': 1e-4,        # Already optimal for CTC training
            'epochs': 30,
            'num_workers': 12,
            'mixed_precision': False,  # Disable to prevent numerical issues
        },
        'rtx6000-2gpu': {
            'batch_size': 32,  # Much more conservative batch size
            'lr': 1e-4,        # Already optimal for CTC training
            'epochs': 25,
            'num_workers': 16,
            'mixed_precision': False,  # Disable to prevent numerical issues
        },
        'default': {
            'batch_size': 16,
            'lr': 1e-4,        # Updated to optimal CTC learning rate
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
    parser.add_argument('--dataset', choices=['gigaspeech'], default=None,
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
    
    # Setup memory monitoring
    setup_memory_monitoring()
    print_memory_usage("Initial", rank, local_rank)
    
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
        # Create distributed data loaders FIRST to get vocabulary
        train_loader, val_loader, vocab = create_distributed_dataloaders(
            config, rank, world_size
        )
        
        # Update vocab size in config BEFORE creating model
        config.model.vocab_size = vocab['vocab_size']
        
        # Verify config consistency across ranks AFTER vocab_size is set
        if world_size > 1:
            vocab_size_tensor = torch.tensor(config.model.vocab_size, dtype=torch.long).cuda()
            vocab_sizes = [torch.zeros_like(vocab_size_tensor) for _ in range(world_size)]
            dist.all_gather(vocab_sizes, vocab_size_tensor)
            
            if rank == 0:
                print(f"🔧 Config verification after vocab loading:")
                for i, vs in enumerate(vocab_sizes):
                    print(f"   Rank {i}: vocab_size = {vs.item()}")
                
                base_vocab_size = vocab_sizes[0].item()
                for i, vs in enumerate(vocab_sizes[1:], 1):
                    if vs.item() != base_vocab_size:
                        print(f"❌ CONFIG MISMATCH: Rank {i} vocab_size {vs.item()} != {base_vocab_size}")
                        raise RuntimeError(f"Configuration mismatch detected across ranks")
                print(f"✅ Config consistent across all ranks: vocab_size={base_vocab_size}")
        
        # NOW create model with correct vocab_size
        device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        
        print(f"🔧 Rank {rank}: Creating model with correct vocab_size={config.model.vocab_size}")
        
        try:
            model = create_model(config.model)
            param_count_before_gpu = sum(p.numel() for p in model.parameters())
            print(f"✅ Rank {rank}: Model created successfully with {param_count_before_gpu} parameters")
            
            model = model.to(device)
            param_count_after_gpu = sum(p.numel() for p in model.parameters())
            
            print(f"✅ Rank {rank}: Model moved to {device}, still has {param_count_after_gpu} parameters")
                
        except Exception as e:
            print(f"❌ Rank {rank}: Model creation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        # Monitor memory after model creation
        print_memory_usage("After Model Creation", rank, local_rank)
        
        # Verify model consistency before DDP wrapping
        if world_size > 1:
            # Add barrier to ensure all ranks have created models
            dist.barrier()
            
            # Verify model parameter consistency
            if not verify_model_consistency(model, rank, world_size):
                raise RuntimeError(f"Model parameter mismatch detected across ranks")
            
            # Add another barrier before DDP wrapping
            dist.barrier()
            
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
            if rank == 0:
                print(f"✅ Model wrapped with DDP")
        
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