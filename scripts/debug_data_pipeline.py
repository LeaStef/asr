#!/usr/bin/env python3

import os
import sys
import torch
from pathlib import Path

# Change to parent directory to ensure relative paths work correctly
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("src")

from config.config import create_config_from_dict
from data.dataset import create_distributed_dataloaders
from omegaconf import OmegaConf

def debug_data_pipeline():
    """Debug the data pipeline to identify character learning issues"""
    
    print("=== DATA PIPELINE DEBUG ===")
    
    # Load configuration
    cfg = OmegaConf.load("configs/base_config.yaml")
    config = create_config_from_dict(cfg)
    
    print(f"Dataset: {config.data.dataset}")
    print(f"Subset: {config.data.subset}")
    print(f"Data directory: {config.data.save_dir}")
    
    # Check if data directory exists
    data_dir = Path(config.data.save_dir)
    print(f"\nData directory exists: {data_dir.exists()}")
    
    if data_dir.exists():
        print("Contents of data directory:")
        for item in data_dir.iterdir():
            print(f"  {item.name}")
    
    # Check vocabulary file
    vocab_path = data_dir / "vocab.txt"
    print(f"\nVocab file exists: {vocab_path.exists()}")
    
    if vocab_path.exists():
        with open(vocab_path, 'r') as f:
            vocab_chars = [line.strip() for line in f]
        print(f"Vocabulary size: {len(vocab_chars)}")
        print(f"First 10 chars: {vocab_chars[:10]}")
        print(f"Last 10 chars: {vocab_chars[-10:]}")
        print(f"Has blank token: {'<blank>' in vocab_chars}")
    
    # Check manifest files
    train_manifest = data_dir / "train_manifest.json"
    val_manifest = data_dir / "dev_manifest.json"
    
    print(f"\nTrain manifest exists: {train_manifest.exists()}")
    print(f"Val manifest exists: {val_manifest.exists()}")
    
    try:
        # Create dataloaders
        print("\n=== CREATING DATALOADERS ===")
        train_loader, val_loader, vocab = create_distributed_dataloaders(config, rank=0, world_size=1)
        
        print(f"Train dataset size: {len(train_loader.dataset)}")
        print(f"Val dataset size: {len(val_loader.dataset)}")
        print(f"Vocab size: {vocab['vocab_size']}")
        print(f"Blank token ID: {vocab['blank_token_id']}")
        
        # Print vocabulary mapping
        print(f"\nVocabulary mapping:")
        for idx, char in vocab['idx_to_char'].items():
            print(f"  {idx:2d}: '{char}'")
        
        # Check a few samples
        print(f"\n=== SAMPLE ANALYSIS ===")
        train_iter = iter(train_loader)
        batch = next(train_iter)
        
        spectrograms, texts, input_lengths, target_lengths = batch
        
        print(f"Batch size: {spectrograms.shape[0]}")
        print(f"Spectrogram shape: {spectrograms.shape}")
        print(f"Text shape: {texts.shape}")
        print(f"Input lengths: {input_lengths}")
        print(f"Target lengths: {target_lengths}")
        
        # Analyze first sample
        sample_idx = 0
        sample_spec = spectrograms[sample_idx]
        sample_text = texts[sample_idx]
        input_len = input_lengths[sample_idx].item()
        target_len = target_lengths[sample_idx].item()
        
        print(f"\nSample {sample_idx}:")
        print(f"  Input length: {input_len}")
        print(f"  Target length: {target_len}")
        print(f"  Spectrogram shape: {sample_spec.shape}")
        print(f"  Spectrogram range: [{sample_spec.min():.3f}, {sample_spec.max():.3f}]")
        
        # Decode target text
        target_indices = sample_text[:target_len].cpu().numpy()
        target_text = ''.join([vocab['idx_to_char'][idx] for idx in target_indices])
        print(f"  Target indices: {target_indices}")
        print(f"  Target text: '{target_text}'")
        
        # Check for common issues
        print(f"\n=== ISSUE DETECTION ===")
        
        # Check if all targets are blank
        all_blank = all(idx == vocab['blank_token_id'] for idx in target_indices)
        print(f"All targets are blank: {all_blank}")
        
        # Check if spectrograms are all zeros or have proper variation
        spec_mean = sample_spec.mean().item()
        spec_std = sample_spec.std().item()
        print(f"Spectrogram stats: mean={spec_mean:.3f}, std={spec_std:.3f}")
        
        if spec_std < 0.1:
            print("⚠️  WARNING: Spectrogram has very low variation - potential audio preprocessing issue")
        
        # Check target text quality
        if len(target_text.strip()) == 0:
            print("⚠️  WARNING: Target text is empty")
        elif len(set(target_text)) < 3:
            print("⚠️  WARNING: Target text has very low character diversity")
        
        # Check vocab size vs config
        if vocab['vocab_size'] != config.model.decoder.vocab_size:
            print(f"⚠️  WARNING: Vocab size mismatch - data: {vocab['vocab_size']}, model: {config.model.decoder.vocab_size}")
        
        # Analyze multiple samples
        print(f"\n=== MULTIPLE SAMPLE ANALYSIS ===")
        sample_texts = []
        sample_lengths = []
        
        for i in range(min(5, spectrograms.shape[0])):
            target_len = target_lengths[i].item()
            target_indices = texts[i][:target_len].cpu().numpy()
            target_text = ''.join([vocab['idx_to_char'][idx] for idx in target_indices])
            sample_texts.append(target_text)
            sample_lengths.append(target_len)
            print(f"Sample {i}: len={target_len}, text='{target_text}'")
        
        avg_length = sum(sample_lengths) / len(sample_lengths)
        print(f"Average target length: {avg_length:.1f}")
        
        # Check character distribution
        all_chars = ''.join(sample_texts)
        char_counts = {}
        for char in all_chars:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        print(f"Character distribution in samples:")
        for char, count in sorted(char_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  '{char}': {count}")
            
    except Exception as e:
        print(f"❌ Error creating dataloaders: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_data_pipeline()