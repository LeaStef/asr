#!/usr/bin/env python3

import os
import sys
import torch
import torch.multiprocessing as mp
import hydra
from omegaconf import DictConfig
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.config import create_config_from_dict
from config.distributed_config import setup_distributed, cleanup_distributed
from models.asr_model import create_model
from data.dataset import create_distributed_dataloaders
from training.distributed import DistributedTrainer


def train_worker(rank: int, world_size: int, cfg: DictConfig):
    """Worker function for distributed training"""
    
    # Setup distributed training
    setup_distributed(rank, world_size)
    
    # Create config object
    config = create_config_from_dict(cfg)
    
    # Create model and move to GPU
    model = create_model(config.model).to(rank)
    
    # Wrap model with DDP
    from torch.nn.parallel import DistributedDataParallel as DDP
    model = DDP(model, device_ids=[rank])
    
    # Create distributed data loaders
    train_loader, val_loader, vocab = create_distributed_dataloaders(
        config, rank, world_size
    )
    
    # Update vocab size in config
    config.model.vocab_size = vocab['vocab_size']
    
    # Create distributed trainer
    trainer = DistributedTrainer(model, config.training, rank, world_size)
    
    # Train model
    trainer.fit(train_loader, val_loader, vocab)
    
    # Cleanup
    cleanup_distributed()


@hydra.main(config_path="../configs", config_name="multi_gpu", version_base=None)
def main(cfg: DictConfig):
    """Main function for distributed training"""
    
    # Get number of GPUs
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print("Distributed training requires at least 2 GPUs")
        print(f"Found {world_size} GPU(s)")
        sys.exit(1)
    
    print(f"Starting distributed training with {world_size} GPUs")
    
    # Use torch.multiprocessing to spawn processes
    mp.spawn(
        train_worker,
        args=(world_size, cfg),
        nprocs=world_size,
        join=True
    )
    
    print("Distributed training completed!")


if __name__ == "__main__":
    main()