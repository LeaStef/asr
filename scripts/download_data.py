#!/usr/bin/env python3

import os
import argparse
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
import torchaudio
import pandas as pd
from tqdm import tqdm


def download_librispeech(subset: str = "clean-100", save_dir: str = "./data"):
    """Download LibriSpeech dataset and create manifest files"""
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading LibriSpeech {subset} dataset...")
    
    # Map subset names to actual dataset configs
    subset_map = {
        "clean-100": "clean",
        "clean-360": "clean", 
        "other-500": "other"
    }
    
    if subset not in subset_map:
        raise ValueError(f"Unknown subset: {subset}")
    
    config = subset_map[subset]
    
    # Load training data
    train_dataset = load_dataset("librispeech_asr", config, split="train.100")
    
    # Load validation data
    val_dataset = load_dataset("librispeech_asr", config, split="validation")
    
    # Load test data
    test_dataset = load_dataset("librispeech_asr", config, split="test")
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create manifest files
    def create_manifest(dataset, split_name):
        manifest_data = []
        
        print(f"Processing {split_name} split...")
        for i, sample in enumerate(tqdm(dataset)):
            audio_array = sample['audio']['array']
            sample_rate = sample['audio']['sampling_rate']
            text = sample['text']
            
            # Save audio file
            audio_dir = save_path / split_name / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            
            audio_path = audio_dir / f"{i:06d}.wav"
            torchaudio.save(str(audio_path), audio_array.unsqueeze(0), sample_rate)
            
            manifest_data.append({
                'audio_path': str(audio_path),
                'text': text.upper(),  # Convert to uppercase
                'duration': len(audio_array) / sample_rate
            })
        
        # Save manifest
        manifest_path = save_path / f"{split_name}_manifest.json"
        df = pd.DataFrame(manifest_data)
        df.to_json(manifest_path, orient='records', lines=True)
        
        print(f"Saved {len(manifest_data)} samples to {manifest_path}")
        return manifest_data
    
    # Create manifests for all splits
    train_manifest = create_manifest(train_dataset, "train")
    val_manifest = create_manifest(val_dataset, "val")
    test_manifest = create_manifest(test_dataset, "test")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total training duration: {sum(item['duration'] for item in train_manifest):.2f} hours")
    print(f"Total validation duration: {sum(item['duration'] for item in val_manifest):.2f} hours")
    print(f"Total test duration: {sum(item['duration'] for item in test_manifest):.2f} hours")
    
    # Create vocabulary
    vocab_chars = set()
    for manifest in [train_manifest, val_manifest, test_manifest]:
        for item in manifest:
            vocab_chars.update(item['text'])
    
    # Standard character vocabulary
    vocab = sorted(list(vocab_chars))
    vocab_dict = {char: i for i, char in enumerate(vocab)}
    vocab_dict['<blank>'] = len(vocab)  # CTC blank token
    
    # Save vocabulary
    vocab_path = save_path / "vocab.txt"
    with open(vocab_path, 'w') as f:
        for char in vocab:
            f.write(f"{char}\n")
        f.write("<blank>\n")
    
    print(f"Vocabulary size: {len(vocab_dict)}")
    print(f"Vocabulary saved to: {vocab_path}")
    
    print(f"\nDataset downloaded and processed successfully!")
    print(f"Data saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare LibriSpeech dataset")
    parser.add_argument("--subset", type=str, default="clean-100", 
                       choices=["clean-100", "clean-360", "other-500"],
                       help="LibriSpeech subset to download")
    parser.add_argument("--save_dir", type=str, default="./data",
                       help="Directory to save the dataset")
    
    args = parser.parse_args()
    
    download_librispeech(args.subset, args.save_dir)


if __name__ == "__main__":
    main()