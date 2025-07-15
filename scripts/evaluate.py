#!/usr/bin/env python3

import os
import sys
import torch
import hydra
from omegaconf import DictConfig
from pathlib import Path
import jiwer
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.config import create_config_from_dict
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


@hydra.main(config_path="../configs", config_name="single_gpu", version_base=None)
def evaluate(cfg: DictConfig):
    """Evaluation script"""
    
    # Create config object
    config = create_config_from_dict(cfg)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(config.model).to(device)
    
    # Load checkpoint
    checkpoint_path = cfg.get("checkpoint_path", None)
    if checkpoint_path is None:
        print("No checkpoint path provided. Please specify --checkpoint_path")
        sys.exit(1)
    
    checkpoint = load_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create data loaders
    train_loader, val_loader, vocab = create_dataloaders(config.data)
    
    # Update vocab size in config
    config.model.vocab_size = vocab['vocab_size']
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_wer, val_cer, val_predictions, val_targets = evaluate_model(
        model, val_loader, vocab, device
    )
    
    print(f"Validation WER: {val_wer:.4f}")
    print(f"Validation CER: {val_cer:.4f}")
    
    # Print some examples
    print("\nExample predictions:")
    for i in range(min(5, len(val_predictions))):
        print(f"Target:     {val_targets[i]}")
        print(f"Prediction: {val_predictions[i]}")
        print()


if __name__ == "__main__":
    evaluate()