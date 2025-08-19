#!/usr/bin/env python3

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import jiwer

# Change to parent directory to ensure relative paths work correctly
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("src")

from config.config import create_config_from_dict
from models.asr_model import create_model
from data.dataset import create_dataloaders
from omegaconf import OmegaConf

def decode_predictions(predictions, vocab):
    """Convert prediction indices to text"""
    decoded = []
    for pred in predictions:
        chars = []
        for idx in pred:
            if idx < len(vocab['idx_to_char']):
                char = vocab['idx_to_char'][idx]
                if char != '<blank>':
                    chars.append(char)
        decoded.append(''.join(chars))
    return decoded

def decode_targets(targets, target_lengths, vocab):
    """Convert target indices to text"""
    decoded = []
    for i, length in enumerate(target_lengths):
        target_seq = targets[i][:length]
        chars = []
        for idx in target_seq:
            if idx.item() < len(vocab['idx_to_char']):
                char = vocab['idx_to_char'][idx.item()]
                if char != '<blank>':
                    chars.append(char)
        decoded.append(''.join(chars))
    return decoded

def compute_wer(predictions, targets):
    """Compute Word Error Rate"""
    if not predictions or not targets:
        return 1.0
    
    total_wer = 0
    valid_samples = 0
    
    for pred, target in zip(predictions, targets):
        if target.strip():  # Only compute WER for non-empty targets
            try:
                wer = jiwer.wer(target, pred)
                total_wer += wer
                valid_samples += 1
            except:
                total_wer += 1.0  # Count as complete error if jiwer fails
                valid_samples += 1
    
    return total_wer / valid_samples if valid_samples > 0 else 1.0

def compute_cer(predictions, targets):
    """Compute Character Error Rate"""
    if not predictions or not targets:
        return 1.0
        
    total_cer = 0
    valid_samples = 0
    
    for pred, target in zip(predictions, targets):
        if target.strip():  # Only compute CER for non-empty targets
            try:
                cer = jiwer.cer(target, pred)
                total_cer += cer
                valid_samples += 1
            except:
                total_cer += 1.0  # Count as complete error if jiwer fails
                valid_samples += 1
    
    return total_cer / valid_samples if valid_samples > 0 else 1.0

def validate_model(checkpoint_path):
    """Validate model with checkpoint using ORIGINAL model size"""
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load base configuration but use ORIGINAL model size
    cfg = OmegaConf.load("configs/base_config.yaml")
    
    # Override with original model architecture
    cfg.model.encoder.hidden_size = 512
    cfg.model.encoder.memory_size = 256
    cfg.model.encoder.num_lmu_layers = 4
    cfg.model.encoder.num_attention_heads = 8
    
    config = create_config_from_dict(cfg)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using ORIGINAL model size: 512 hidden, 256 memory, 4 layers")
    
    # Create model with original size
    model = create_model(config.model).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create data loaders
    _, val_loader, vocab = create_dataloaders(config)
    
    print(f"Vocabulary size: {vocab['vocab_size']}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Set model to evaluation mode
    model.eval()
    
    all_predictions = []
    all_targets = []
    total_loss = 0
    num_batches = 0
    
    print("Running validation...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            spectrograms, texts, input_lengths, target_lengths = batch
            
            spectrograms = spectrograms.to(device)
            texts = texts.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)
            
            # Forward pass
            log_probs, _ = model(spectrograms, input_lengths)
            
            # Compute loss
            loss = model.compute_loss(log_probs, texts, input_lengths, target_lengths)
            total_loss += loss.item()
            num_batches += 1
            
            # Decode predictions
            predictions = model.decode(log_probs, input_lengths)
            
            # Convert to text
            pred_texts = decode_predictions(predictions, vocab)
            target_texts = decode_targets(texts, target_lengths, vocab)
            
            all_predictions.extend(pred_texts)
            all_targets.extend(target_texts)
    
    # Calculate metrics
    avg_loss = total_loss / num_batches
    wer = compute_wer(all_predictions, all_targets)
    cer = compute_cer(all_predictions, all_targets)
    
    # Print results
    print(f"\nValidation Results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  WER:  {wer:.4f} ({wer*100:.2f}%)")
    print(f"  CER:  {cer:.4f} ({cer*100:.2f}%)")
    
    # Print example predictions
    print(f"\nExample Predictions:")
    for i in range(min(5, len(all_predictions))):
        print(f"\nSample {i+1}:")
        print(f"  Target:     '{all_targets[i]}'")
        print(f"  Prediction: '{all_predictions[i]}'")
    
    return avg_loss, wer, cer

if __name__ == "__main__":
    checkpoint_path = "./checkpoint_epoch_7.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please make sure you've copied the checkpoint file to this directory.")
        sys.exit(1)
    
    validate_model(checkpoint_path)