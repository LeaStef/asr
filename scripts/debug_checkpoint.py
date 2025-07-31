#!/usr/bin/env python3

import torch
import sys
from pathlib import Path

def debug_checkpoint(checkpoint_path):
    """Debug checkpoint contents"""
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print("\n=== Checkpoint Keys ===")
    for key in checkpoint.keys():
        print(f"- {key}")
    
    print(f"\n=== Model State Dict Info ===")
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Number of parameters: {len(state_dict)}")
        
        print("\nModel layers:")
        for key in sorted(state_dict.keys())[:10]:  # First 10 layers
            shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'scalar'
            print(f"  {key}: {shape}")
        
        if len(state_dict) > 10:
            print(f"  ... and {len(state_dict) - 10} more layers")
    
    print(f"\n=== Training Info ===") 
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'global_step' in checkpoint:
        print(f"Global step: {checkpoint['global_step']}")
    if 'best_val_loss' in checkpoint:
        print(f"Best validation loss: {checkpoint['best_val_loss']}")
    if 'train_loss' in checkpoint:
        print(f"Training loss: {checkpoint['train_loss']}")
    if 'val_loss' in checkpoint:
        print(f"Validation loss: {checkpoint['val_loss']}")
    
    print(f"\n=== Vocabulary Info ===")
    if 'vocab' in checkpoint:
        vocab = checkpoint['vocab']
        print(f"Vocab keys: {list(vocab.keys())}")
        if 'vocab_size' in vocab:
            print(f"Vocab size: {vocab['vocab_size']}")
        if 'idx_to_char' in vocab:
            print(f"First 10 characters: {list(vocab['idx_to_char'].items())[:10]}")
    else:
        print("No vocabulary found in checkpoint")
    
    print(f"\n=== Config Info ===")
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"Config type: {type(config)}")
        if hasattr(config, 'model'):
            print(f"Model config: {config.model}")
    else:
        print("No config found in checkpoint")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_checkpoint.py <checkpoint_path>")
        sys.exit(1)
    
    debug_checkpoint(sys.argv[1])