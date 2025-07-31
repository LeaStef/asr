#!/usr/bin/env python3

import os
import sys
import torch
import argparse
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.config import ModelConfig, DataConfig, create_config_from_dict
from models.asr_model import create_model
from data.dataset import create_dataloaders
from training.utils import load_checkpoint, decode_predictions, decode_targets, compute_wer, compute_cer


def evaluate_model(model, test_loader, vocab, device):
    """Evaluate model on test set"""
    
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            spectrograms, texts, input_lengths, label_lengths = batch
            
            spectrograms = spectrograms.to(device)
            texts = texts.to(device)
            
            # Forward pass
            log_probs, _ = model(spectrograms, input_lengths)
            
            # Decode predictions
            predictions = model.decode(log_probs, input_lengths)
            
            # Convert to text
            pred_texts = decode_predictions(predictions, vocab)
            target_texts = decode_targets(texts, label_lengths, vocab)
            
            all_predictions.extend(pred_texts)
            all_targets.extend(target_texts)
    
    # Calculate WER
    wer = compute_wer(all_predictions, all_targets)
    cer = compute_cer(all_predictions, all_targets)
    
    return wer, cer, all_predictions, all_targets


def main():
    parser = argparse.ArgumentParser(description="Evaluate ASR model")
    parser.add_argument("--checkpoint_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--dataset", default="gigaspeech", help="Dataset to use")
    parser.add_argument("--subset", default="xs", help="Dataset subset")
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"Checkpoint not found: {args.checkpoint_path}")
        sys.exit(1)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint to get saved config
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    
    # Create config from checkpoint or use defaults
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
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
                    'vocab_size': 32  # Will be updated from vocab
                }
            },
            'data': {
                'dataset': args.dataset,
                'subset': args.subset,
                'save_dir': "./data/gigaspeech",
                'sample_rate': 16000,
                'n_mels': 80,
                'max_seq_len': 500,
                'augment': False,  # No augmentation for evaluation
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
    
    # Update vocab size in config
    config.model.vocab_size = vocab['vocab_size']
    
    print(f"Using vocabulary with {vocab['vocab_size']} characters")
    print("First 10 vocabulary mappings:")
    for i in range(min(10, len(vocab['idx_to_char']))):
        print(f"  {i}: '{vocab['idx_to_char'][i]}'")
    print()
    
    # Create model
    print("Creating model...")
    model = create_model(config.model).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_wer, val_cer, val_predictions, val_targets = evaluate_model(
        model, val_loader, vocab, device
    )
    
    print(f"\nResults:")
    print(f"Validation WER: {val_wer:.4f} ({val_wer*100:.2f}%)")
    print(f"Validation CER: {val_cer:.4f} ({val_cer*100:.2f}%)")
    
    # Print some examples
    print(f"\nExample predictions (showing first 5):")
    print("="*80)
    for i in range(min(5, len(val_predictions))):
        print(f"Target:     {val_targets[i]}")
        print(f"Prediction: {val_predictions[i]}")
        print("-"*40)


if __name__ == "__main__":
    main()