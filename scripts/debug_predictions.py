#!/usr/bin/env python3

import os
import sys
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.config import ModelConfig, DataConfig, create_config_from_dict
from models.asr_model import create_model
from data.dataset import create_dataloaders
from training.utils import load_checkpoint, decode_predictions, decode_targets, compute_wer, compute_cer


def debug_model_predictions(model, test_loader, vocab, device, num_samples=3):
    """Debug what the model is actually predicting"""
    
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
                
            spectrograms, texts, input_lengths, label_lengths = batch
            
            spectrograms = spectrograms.to(device)
            texts = texts.to(device)
            
            print(f"\n=== Sample {i+1} ===")
            print(f"Input shape: {spectrograms.shape}")
            print(f"Input length: {input_lengths[0].item()}")
            
            # Forward pass
            log_probs, _ = model(spectrograms, input_lengths)
            print(f"Log probs shape: {log_probs.shape}")
            print(f"Log probs range: [{log_probs.min().item():.3f}, {log_probs.max().item():.3f}]")
            
            # Get raw predictions (argmax)
            raw_predictions = torch.argmax(log_probs, dim=-1)
            print(f"Raw prediction shape: {raw_predictions.shape}")
            
            # Look at first sample's predictions
            first_sample_pred = raw_predictions[0]  # Shape: [time_steps]
            print(f"First 20 predicted indices: {first_sample_pred[:20].tolist()}")
            
            # Count unique predictions
            unique_preds, counts = torch.unique(first_sample_pred, return_counts=True)
            print("Predicted token distribution:")
            for pred, count in zip(unique_preds.tolist(), counts.tolist()):
                char = vocab['idx_to_char'].get(pred, f'UNK({pred})')
                percentage = count / len(first_sample_pred) * 100
                print(f"  {pred} ('{char}'): {count} times ({percentage:.1f}%)")
            
            # Decode predictions using CTC
            predictions = model.decode(log_probs, input_lengths)
            pred_text = decode_predictions(predictions, vocab)[0]
            
            # Get target
            target_text = decode_targets(texts, label_lengths, vocab)[0]
            
            print(f"CTC decoded prediction: '{pred_text}'")
            print(f"Target: '{target_text[:100]}...'")
            
            # Check if prediction is just blanks
            if len(pred_text.strip()) == 0:
                print("⚠️  WARNING: Prediction is empty after CTC decoding!")
                print("   This suggests model is outputting mostly blank tokens")


def main():
    parser = argparse.ArgumentParser(description="Debug ASR model predictions")
    parser.add_argument("--checkpoint_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for debugging")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to debug")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    
    # Create default config
    from omegaconf import OmegaConf
    config_dict = {
        'model': {
            'encoder': {
                'input_size': 80,
                'hidden_size': 512,
                'memory_size': 256,
                'num_lmu_layers': 4,
                'theta': 1000,
                'dropout': 0.1,
                'use_fft_lmu': False,
                'use_attention': True,
                'num_attention_heads': 8,
                'use_downsampling': False,
                'downsample_factor': 2
            },
            'decoder': {
                'vocab_size': 32
            }
        },
        'data': {
            'dataset': "gigaspeech",
            'subset': "xs",
            'save_dir': "./data/gigaspeech",
            'sample_rate': 16000,
            'n_mels': 80,
            'max_seq_len': 500,
            'augment': False,
            'num_workers': 4
        },
        'training': {
            'batch_size': args.batch_size,
            'lr': 1e-3,
            'max_epochs': 50,
            'patience': 10,
            'mixed_precision': True,
            'gradient_clip_norm': 1.0,
            'accumulate_grad_batches': 1,
            'warmup_steps': 1000,
            'weight_decay': 0.01,
            'log_interval': 100
        }
    }
    config = create_config_from_dict(OmegaConf.create(config_dict))
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, vocab = create_dataloaders(config)
    config.model.vocab_size = vocab['vocab_size']
    
    # Create model
    print("Creating model...")
    model = create_model(config.model).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"\nDebugging {args.num_samples} samples...")
    
    # Debug predictions
    debug_model_predictions(model, val_loader, vocab, device, args.num_samples)


if __name__ == "__main__":
    main()