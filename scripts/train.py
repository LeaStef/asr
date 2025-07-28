#!/usr/bin/env python3

import os
import sys
import torch
import hydra
from omegaconf import DictConfig
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.config import create_config_from_dict
from models.asr_model import create_model
from data.dataset import create_dataloaders
from training.trainer import Trainer


@hydra.main(config_path="../configs", config_name="single_gpu", version_base=None)
def train(cfg: DictConfig):
    """Single GPU training script"""
    
    # Create config object
    config = create_config_from_dict(cfg)
    
    # Print dataset info
    print(f"Using dataset: {config.data.dataset}")
    if config.data.dataset == "gigaspeech":
        print(f"GigaSpeech subset: {config.data.subset}")
        print(f"Data directory: {config.data.save_dir}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(config.model).to(device)
    
    # Create data loaders
    train_loader, val_loader, vocab = create_dataloaders(config)
    
    # Update vocab size in config
    config.model.vocab_size = vocab['vocab_size']
    
    # Create trainer
    trainer = Trainer(model, config.training, config.data, device)
    
    # Train model
    trainer.fit(train_loader, val_loader, vocab)
    
    print("Training completed!")


if __name__ == "__main__":
    train()