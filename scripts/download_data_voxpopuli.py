#!/usr/bin/env python3

import os
import argparse
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
import torchaudio
import pandas as pd
from tqdm import tqdm

import torch

def download(subset: str = "clean-360", save_dir: str = '/home/ctnuser/pytorch-lmu-asr/data/LibriSpeech'):
    """Download LibriSpeech dataset and create manifest files"""
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # print(f"Downloading {subset} dataset...")
    
    # # Map subset names to actual dataset configs
    # subset_map = {
    #     "clean-100": "clean",
    #     "clean-360": "clean", 
    #     "other-500": "other"
    # }
    
    # if subset not in subset_map:
    #     raise ValueError(f"Unknown subset: {subset}")
    
    # config = subset_map[subset]
    
    # Load training data
    train_dataset = load_dataset("facebook/voxpopuli", "en_accented", data_dir="/home/ctnuser/pytorch-lmu-asr/data" )#['train']
    
    # Load validation data
    val_dataset = load_dataset("facebook/voxpopuli", "en_accented", data_dir="/home/ctnuser/pytorch-lmu-asr/data")
    
    # Load test data
    test_dataset = load_dataset("facebook/voxpopuli", "en_accented", data_dir="/home/ctnuser/pytorch-lmu-asr/data")
    
    print(f"Train samples: {len(train_dataset['test'])}")
    print(f"Validation samples: {len(val_dataset['test'])}")
    print(f"Test samples: {len(test_dataset['test'])}")
    
    # Create manifest files
    def create_manifest(dataset, split_name):
        manifest_data = []
        
        print(f"Processing {split_name} split...")
        for i, sample in enumerate(tqdm(dataset['test'])):
            # Get audio data
            audio_array = torch.tensor(sample['audio']['array'])
            sample_rate = sample['audio']['sampling_rate']
            
            # Get text directly from the sample
            text = sample['normalized_text']
            
            # Save audio file
            audio_dir = save_path / split_name / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            
            audio_path = audio_dir / f"{i:06d}.wav"
            torchaudio.save(str(audio_path), audio_array.unsqueeze(0), sample_rate)
            
            manifest_data.append({
                'audio_path': str(audio_path),
                'text': text.upper() if isinstance(text, str) else str(text).upper(),
                'duration': len(audio_array) / sample_rate
            })
        
        # Save manifest
        manifest_path = save_path / f"{split_name}_manifest.json"
        df = pd.DataFrame(manifest_data)
        df.to_json(manifest_path, orient='records', lines=True)
        
        print(f"Saved {len(manifest_data)} samples to {manifest_path}")
        return manifest_data
    
    for i, sample in enumerate(tqdm(train_dataset['test'])):
        print(sample.keys())
        break 
    
    # Create manifests for all splits
    train_manifest = create_manifest(train_dataset, "test")
    val_manifest = create_manifest(val_dataset, "test")
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
    parser = argparse.ArgumentParser(description="Download and prepare  dataset")
    parser.add_argument("--subset", type=str, default="", 
                       choices=["clean-100", "clean-360", "other-500"],
                       help="Subset to download")
    parser.add_argument("--save_dir", type=str, default="./data",
                       help="Directory to save the dataset")
    
    args = parser.parse_args()
    
    download(args.save_dir)


if __name__ == "__main__":
    main()