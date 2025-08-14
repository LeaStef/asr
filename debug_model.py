#!/usr/bin/env python3

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.append("src")

from config.config import create_config_from_dict
from models.asr_model import create_model
from data.dataset import create_dataloaders
from omegaconf import OmegaConf

def debug_model_outputs(checkpoint_path, num_samples=5):
    """Debug what the model is actually outputting"""
    
    print(f"=== MODEL DEBUG ANALYSIS ===")
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load configuration
    cfg = OmegaConf.load("configs/base_config.yaml")
    config = create_config_from_dict(cfg)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(config.model).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Get vocabulary
    _, val_loader, vocab = create_dataloaders(config)
    print(f"Vocabulary size: {vocab['vocab_size']}")
    print(f"Blank token ID: {vocab['blank_token_id']}")
    
    # Print vocabulary
    print(f"\nVocabulary mapping:")
    for idx, char in vocab['idx_to_char'].items():
        print(f"  {idx:2d}: '{char}'")
    
    model.eval()
    
    print(f"\n=== ANALYZING {num_samples} SAMPLES ===")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= num_samples:
                break
                
            spectrograms, texts, input_lengths, target_lengths = batch
            spectrograms = spectrograms.to(device)
            texts = texts.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)
            
            # Forward pass
            log_probs, memory_states = model(spectrograms, input_lengths)
            
            # Analyze first sample in batch
            sample_idx = 0
            seq_len = input_lengths[sample_idx].item()
            target_len = target_lengths[sample_idx].item()
            
            print(f"\n--- SAMPLE {batch_idx + 1} ---")
            print(f"Input length: {seq_len}")
            print(f"Target length: {target_len}")
            
            # Get raw log probabilities for this sample
            sample_log_probs = log_probs[sample_idx, :seq_len]  # (seq_len, vocab_size)
            sample_probs = torch.exp(sample_log_probs)  # Convert to probabilities
            
            # Get target text
            target_indices = texts[sample_idx, :target_len].cpu().numpy()
            target_text = ''.join([vocab['idx_to_char'][idx] for idx in target_indices])
            print(f"Target text: '{target_text}'")
            
            # Analyze model predictions
            print(f"\nModel output analysis:")
            print(f"  Log prob shape: {sample_log_probs.shape}")
            print(f"  Min log prob: {sample_log_probs.min().item():.4f}")
            print(f"  Max log prob: {sample_log_probs.max().item():.4f}")
            
            # Get most likely predictions at each time step
            predicted_tokens = torch.argmax(sample_log_probs, dim=-1).cpu().numpy()
            print(f"  Predicted tokens (first 20): {predicted_tokens[:20]}")
            
            # Count predictions
            unique_tokens, counts = np.unique(predicted_tokens, return_counts=True)
            print(f"  Token distribution:")
            for token, count in zip(unique_tokens, counts):
                char = vocab['idx_to_char'].get(token, f'UNK_{token}')
                percentage = (count / len(predicted_tokens)) * 100
                print(f"    {token:2d} ('{char}'): {count:4d} times ({percentage:5.1f}%)")
            
            # Check if model is just predicting blank
            blank_id = vocab['blank_token_id']
            blank_percentage = (predicted_tokens == blank_id).sum() / len(predicted_tokens) * 100
            print(f"  Blank token percentage: {blank_percentage:.1f}%")
            
            # Analyze probability distribution
            avg_probs = sample_probs.mean(dim=0).cpu().numpy()
            print(f"  Average probability per token:")
            top_5_indices = np.argsort(avg_probs)[-5:][::-1]
            for idx in top_5_indices:
                char = vocab['idx_to_char'].get(idx, f'UNK_{idx}')
                print(f"    {idx:2d} ('{char}'): {avg_probs[idx]:.4f}")
            
            # Try CTC decoding manually
            print(f"\nCTC decoding analysis:")
            decoded = []
            prev_token = None
            for token in predicted_tokens:
                if token != blank_id and token != prev_token:
                    char = vocab['idx_to_char'].get(token, f'UNK_{token}')
                    decoded.append(char)
                prev_token = token
            
            decoded_text = ''.join(decoded)
            print(f"  Manual CTC decode: '{decoded_text}'")
            
            # Compare with model's decode method
            model_decoded = model.decode(log_probs[sample_idx:sample_idx+1], input_lengths[sample_idx:sample_idx+1])
            print(f"  Model decode method: '{model_decoded[0] if model_decoded else ''}'")
            
            # Check for numerical issues
            has_nan = torch.isnan(sample_log_probs).any()
            has_inf = torch.isinf(sample_log_probs).any()
            print(f"  Has NaN: {has_nan}")
            print(f"  Has Inf: {has_inf}")
            
            if has_nan or has_inf:
                print("  ⚠️  NUMERICAL ISSUES DETECTED!")
            
            # Check if probabilities sum to 1 (they should after softmax)
            prob_sums = sample_probs.sum(dim=-1)
            print(f"  Prob sums (should be ~1.0): min={prob_sums.min():.4f}, max={prob_sums.max():.4f}")
            
    print(f"\n=== SUMMARY ===")
    print("Key things to check:")
    print("1. Are predictions mostly blank tokens? (indicates model not learning)")
    print("2. Are probabilities distributed or concentrated? (indicates confidence)")
    print("3. Do manual and model CTC decoding match? (indicates decode logic)")
    print("4. Any numerical issues? (NaN/Inf values)")
    print("5. Are target texts correctly formatted? (indicates data processing)")

if __name__ == "__main__":
    checkpoint_path = "./checkpoint_epoch_8.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    debug_model_outputs(checkpoint_path, num_samples=3)