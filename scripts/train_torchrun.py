#!/usr/bin/env python3
"""
Torchrun-compatible distributed training script with batch size override support.

Usage:
    # Single GPU
    python scripts/train_torchrun.py
    
    # Multi-GPU with torchrun (recommended)
    torchrun --nproc_per_node=2 scripts/train_torchrun.py
    torchrun --nproc_per_node=4 scripts/train_torchrun.py
    
    # Custom batch size with Hydra overrides
    python scripts/train_torchrun.py training.batch_size=64
    torchrun --nproc_per_node=2 scripts/train_torchrun.py training.batch_size=128 training.lr=8e-3
    
    # L40S GPU optimization examples
    python scripts/train_torchrun.py --config-name=l40s_1gpu
    torchrun --nproc_per_node=2 scripts/train_torchrun.py --config-name=l40s_2gpu
    
    # Multi-node distributed training
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr=192.168.1.1 --master_port=29500 scripts/train_torchrun.py
"""

import os
import sys
import torch
import torch.distributed as dist
import hydra
from omegaconf import DictConfig
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.config import create_config_from_dict
from config.distributed_config import setup_distributed, cleanup_distributed, get_distributed_config
from models.asr_model import create_model
from data.dataset import create_distributed_dataloaders
from training.distributed import DistributedTrainer


def setup_torchrun_distributed():
    """Setup distributed training for torchrun"""
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


@hydra.main(config_path="../configs", config_name="multi_gpu", version_base=None)
def main(cfg: DictConfig):
    """Main training function"""
    
    # Setup distributed training
    rank, world_size, local_rank = setup_torchrun_distributed()
    
    # Print info only from main process
    if rank == 0:
        print(f"🚀 Training Configuration:")
        print(f"  World size: {world_size}")
        print(f"  Rank: {rank}")
        print(f"  Local rank: {local_rank}")
        print(f"  Using dataset: {cfg.data.dataset}")
        if cfg.data.dataset == "gigaspeech":
            print(f"  GigaSpeech subset: {cfg.data.subset}")
            print(f"  Data directory: {cfg.data.save_dir}")
        print(f"  Device: cuda:{local_rank}")
        
        # Calculate and display batch size info
        total_batch_size = cfg.training.batch_size
        per_gpu_batch_size = total_batch_size // world_size
        if total_batch_size % world_size != 0:
            print(f"  ⚠️  Warning: Batch size {total_batch_size} not evenly divisible by {world_size} GPUs")
        
        print(f"  📊 Batch Configuration:")
        print(f"    Total batch size: {total_batch_size}")
        print(f"    Per-GPU batch size: {per_gpu_batch_size}")
        print(f"    Learning rate: {cfg.training.lr}")
        print(f"    Mixed precision: {cfg.training.mixed_precision}")
        
        # Memory usage estimate for common GPU types
        estimated_memory_per_sample = 18  # MB per sample (from our analysis)
        estimated_memory_per_gpu = per_gpu_batch_size * estimated_memory_per_sample + 100  # +100MB overhead
        
        if estimated_memory_per_gpu < 8000:  # Less than 8GB
            memory_status = "✅ Conservative"
        elif estimated_memory_per_gpu < 16000:  # Less than 16GB  
            memory_status = "✅ Moderate"
        elif estimated_memory_per_gpu < 32000:  # Less than 32GB
            memory_status = "⚠️  High (monitor memory)"
        else:
            memory_status = "🔥 Very High (may need adjustment)"
            
        print(f"    Estimated memory per GPU: ~{estimated_memory_per_gpu/1000:.1f}GB {memory_status}")
    
    # Create config object
    config = create_config_from_dict(cfg)
    
    # Create model and move to GPU
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    model = create_model(config.model).to(device)
    
    # Wrap model with DDP if distributed
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=cfg.distributed.find_unused_parameters)
        if rank == 0:
            print(f"Model wrapped with DDP")
    
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
        log_dir='./outputs'
    )
    
    if rank == 0:
        print(f"Starting distributed training...")
    
    try:
        # Train model
        trainer.fit(train_loader, val_loader, vocab)
        
        if rank == 0:
            print("Training completed successfully!")
            
    except KeyboardInterrupt:
        if rank == 0:
            print("Training interrupted by user")
    except Exception as e:
        if rank == 0:
            print(f"Training failed with error: {str(e)}")
        raise
    finally:
        # Cleanup distributed training
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()