#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GigaSpeechDownloader:
    """Handles downloading and processing of GigaSpeech dataset"""
    
    SUBSETS = {
        "xs": {
            "description": "10 hours subset",
            "splits": {
                "train": "train",
                "dev": "validation",
                "test": "test"
            }
        },
        "s": {
            "description": "250 hours subset",
            "splits": {
                "train": "train",
                "dev": "validation",
                "test": "test"
            }
        },
        "m": {
            "description": "1000 hours subset",
            "splits": {
                "train": "train",
                "dev": "validation",
                "test": "test"
            }
        }
    }
    
    def __init__(self, subset: str, save_dir: str):
        if subset not in self.SUBSETS:
            raise ValueError(f"Invalid subset: {subset}. Choose from {list(self.SUBSETS.keys())}")
            
        self.subset = subset
        self.save_dir = Path(save_dir)
        self.cache_dir = Path.home() / '.cache' / 'huggingface' / 'datasets'
        
        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def authenticate(self) -> None:
        """Authenticate with HuggingFace"""
        try:
            token = os.getenv('HF_TOKEN')
            if not token:
                logger.info("Please enter your HuggingFace token (from https://huggingface.co/settings/tokens):")
                token = input().strip()
            
            login(token)
            logger.info("✓ Successfully authenticated with HuggingFace")
            
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise
    
    def load_split(self, split_name: str) -> any:
        """Load a specific dataset split"""
        split = self.SUBSETS[self.subset]["splits"][split_name]
        logger.info(f"Loading {split_name} split ({split})...")
        
        try:
            dataset = load_dataset(
                "speechcolab/gigaspeech",
                name=self.subset,
                split=split,
                use_auth_token=True
            )
            logger.info(f"✓ Loaded {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load {split_name} split: {str(e)}")
            raise
    
    def create_manifest(self, dataset: any, split_name: str) -> List[Dict]:
        """Create manifest for a dataset split"""
        manifest_data = []
        audio_dir = self.save_dir / split_name / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing {split_name} split...")
        for idx, sample in enumerate(tqdm(dataset)):
            try:
                # Process audio
                audio = sample['audio']
                audio_array = torch.tensor(audio['array'])
                sample_rate = audio['sampling_rate']
                
                # Save audio file
                audio_path = audio_dir / f"{idx:06d}.wav"
                torchaudio.save(
                    str(audio_path),
                    audio_array.unsqueeze(0),
                    sample_rate
                )
                
                # Add to manifest
                manifest_data.append({
                    'audio_path': str(audio_path),
                    'text': sample['text'].upper(),
                    'duration': len(audio_array) / sample_rate
                })
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {str(e)}")
                continue
        
        # Save manifest
        manifest_path = self.save_dir / f"{split_name}_manifest.json"
        pd.DataFrame(manifest_data).to_json(
            manifest_path,
            orient='records',
            lines=True
        )
        
        logger.info(f"✓ Saved {len(manifest_data)} samples to {manifest_path}")
        return manifest_data
    
    def create_vocabulary(self, manifests: List[Dict]) -> Dict[str, int]:
        """Create and save vocabulary file"""
        # Collect characters from all splits
        vocab_chars = set()
        for manifest in manifests:
            for item in manifest:
                vocab_chars.update(item['text'])
        
        # Create vocabulary
        vocab = sorted(list(vocab_chars))
        vocab_dict = {char: i for i, char in enumerate(vocab)}
        vocab_dict['<blank>'] = len(vocab)
        
        # Save vocabulary
        vocab_path = self.save_dir / "vocab.txt"
        with open(vocab_path, 'w') as f:
            for char in vocab:
                f.write(f"{char}\n")
            f.write("<blank>\n")
        
        logger.info(f"✓ Saved vocabulary with {len(vocab_dict)} tokens to {vocab_path}")
        return vocab_dict
    
    def process(self) -> bool:
        """Main processing function"""
        try:
            logger.info(f"Processing GigaSpeech {self.subset} subset ({self.SUBSETS[self.subset]['description']})")
            
            # Authenticate
            self.authenticate()
            
            # Load all splits
            datasets = {}
            manifests = []
            
            for split in ["train", "dev", "test"]:
                # Load and process split
                datasets[split] = self.load_split(split)
                manifest = self.create_manifest(datasets[split], split)
                manifests.append(manifest)
            
            # Create vocabulary
            vocab = self.create_vocabulary(manifests)
            
            # Print statistics
            logger.info("\nDataset Statistics:")
            for split, manifest in zip(["train", "dev", "test"], manifests):
                duration = sum(item['duration'] for item in manifest)
                logger.info(f"{split.capitalize()}:")
                logger.info(f"  Duration: {duration/3600:.2f} hours")
                logger.info(f"  Samples: {len(manifest)}")
            
            logger.info(f"\n✓ Dataset downloaded and processed successfully!")
            logger.info(f"✓ Data saved to: {self.save_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Download and prepare GigaSpeech dataset")
    parser.add_argument(
        "--subset",
        type=str,
        default="xs",
        choices=["xs", "s", "m"],
        help="Subset to download (xs: 10h, s: 250h, m: 1000h)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./data/gigaspeech",
        help="Directory to save the dataset"
    )
    
    args = parser.parse_args()
    
    downloader = GigaSpeechDownloader(args.subset, args.save_dir)
    if not downloader.process():
        sys.exit(1)

if __name__ == "__main__":
    main()