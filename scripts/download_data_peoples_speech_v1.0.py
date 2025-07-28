#!/usr/bin/env python3

import os
import argparse
from pathlib import Path
from datasets import load_dataset
import torchaudio
import pandas as pd
from tqdm import tqdm
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download(save_dir: str = '/home/ctnuser/pytorch-lmu-asr/data'):
    """Download People's Speech dataset and create manifest files"""
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading People's Speech dataset...")
    
    try:
        # Load dataset splits - note we don't specify config name
        train_dataset = load_dataset("MLCommons/peoples_speech_v1.0", split="train")
        val_dataset = load_dataset("MLCommons/peoples_speech_v1.0", split="validation")
        test_dataset = load_dataset("MLCommons/peoples_speech_v1.0", split="test")
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        return False
    
    def create_manifest(dataset, split_name):
        manifest_data = []
        
        logger.info(f"Processing {split_name} split...")
        for i, sample in enumerate(tqdm(dataset)):  # Remove ['test'] indexing
            try:
                # Get audio data
                audio_array = torch.tensor(sample['audio']['array'])
                sample_rate = sample['audio']['sampling_rate']
                
                # Get text
                text = sample['normalized_text']
                if isinstance(text, list):
                    text = " ".join(text)
                
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
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {str(e)}")
                continue
        
        # Save manifest
        manifest_path = save_path / f"{split_name}_manifest.json"
        df = pd.DataFrame(manifest_data)
        df.to_json(manifest_path, orient='records', lines=True)
        
        logger.info(f"Saved {len(manifest_data)} samples to {manifest_path}")
        return manifest_data

    # Create manifests for all splits with correct split names
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
    parser = argparse.ArgumentParser(description="Download and prepare  dataset")
    parser.add_argument("--subset", type=str, default="", 
                       choices=["clean-100", "clean-360", "other-500"],
                       help="Subset to download")
    parser.add_argument("--save_dir", type=str, default="/home/ctnuser/pytorch-lmu-asr/data",
                       help="Directory to save the dataset")
    
    args = parser.parse_args()
    
    download(args.save_dir)


if __name__ == "__main__":
    main()