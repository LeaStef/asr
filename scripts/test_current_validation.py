#!/usr/bin/env python3

import os
import sys
import torch
# Change to parent directory to ensure relative paths work correctly
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("src")

from config.config import create_config_from_dict
from models.asr_model import create_model
from data.dataset import create_distributed_dataloaders
from training.utils import decode_predictions, decode_targets, compute_wer, compute_cer
from omegaconf import OmegaConf

def test_current_validation():
    """Test what the current model actually outputs during validation"""
    
    print("=== CURRENT VALIDATION TEST ===")
    
    # Load configuration
    cfg = OmegaConf.load("configs/base_config.yaml")
    config = create_config_from_dict(cfg)
    
    print(f"Model vocab size: {config.model.vocab_size}")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(config.model).to(device)
    model.eval()
    
    # Create dataloaders
    train_loader, val_loader, vocab = create_distributed_dataloaders(config, rank=0, world_size=1)
    
    print(f"Data vocab size: {vocab['vocab_size']}")
    print(f"Blank token ID: {vocab['blank_token_id']}")
    
    # Test with first validation batch
    val_iter = iter(val_loader)
    batch = next(val_iter)
    
    spectrograms, targets, input_lengths, target_lengths = batch
    spectrograms = spectrograms.to(device)
    targets = targets.to(device)
    input_lengths = input_lengths.to(device)
    target_lengths = target_lengths.to(device)
    
    print(f"\nBatch size: {spectrograms.shape[0]}")
    
    with torch.no_grad():
        # Forward pass
        log_probs, _ = model(spectrograms, input_lengths)
        
        # Decode predictions  
        predictions = model.decode(log_probs, input_lengths)
        
        # Convert to text
        pred_texts = decode_predictions(predictions, vocab)
        target_texts = decode_targets(targets, target_lengths, vocab)
        
        # Show first 3 examples
        print(f"\nActual current validation results:")
        for i in range(min(3, len(pred_texts))):
            print(f"\nExample {i+1}:")
            print(f"  Target:     '{target_texts[i]}'")
            print(f"  Prediction: '{pred_texts[i]}'")
            print(f"  Target has punctuation: {'<' in target_texts[i]}")
            print(f"  Prediction length: {len(pred_texts[i])}")
        
        # Compute metrics
        wer = compute_wer(pred_texts, target_texts)
        cer = compute_cer(pred_texts, target_texts)
        
        print(f"\nMetrics on this batch:")
        print(f"  WER: {wer:.4f}")
        print(f"  CER: {cer:.4f}")

if __name__ == "__main__":
    test_current_validation()