#!/usr/bin/env python3

import os
import sys
# Change to parent directory to ensure relative paths work correctly
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("src")

from config.config import create_config_from_dict
from data.dataset import create_distributed_dataloaders
from omegaconf import OmegaConf

def debug_validation_data():
    """Debug what validation data is actually being used"""
    
    print("=== VALIDATION DATA DEBUG ===")
    
    # Load configuration
    cfg = OmegaConf.load("configs/base_config.yaml")
    config = create_config_from_dict(cfg)
    
    # Create dataloaders
    train_loader, val_loader, vocab = create_distributed_dataloaders(config, rank=0, world_size=1)
    
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    print(f"Vocab size: {vocab['vocab_size']}")
    
    # Check first few validation samples
    val_iter = iter(val_loader)
    batch = next(val_iter)
    
    spectrograms, texts, input_lengths, target_lengths = batch
    
    print(f"\nFirst validation batch:")
    print(f"Batch size: {spectrograms.shape[0]}")
    
    # Check first 3 samples
    for i in range(min(3, spectrograms.shape[0])):
        target_len = target_lengths[i].item()
        target_indices = texts[i][:target_len].cpu().numpy()
        target_text = ''.join([vocab['idx_to_char'][idx] for idx in target_indices])
        
        print(f"\nValidation Sample {i+1}:")
        print(f"  Target indices: {target_indices}")
        print(f"  Target text: '{target_text}'")
        print(f"  Contains punctuation tags: {'<' in target_text}")

if __name__ == "__main__":
    debug_validation_data()