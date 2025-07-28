import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import json
import os
from typing import Dict, List, Tuple, Optional
import random
from datasets import load_dataset
from pathlib import Path

from .preprocessing import AudioPreprocessor, TextPreprocessor, SpecAugment, collate_fn

class CustomManifestDataset(Dataset):
    """
    Generic ASR dataset using manifest files.
    Each line in the manifest is a JSON object with 'audio_path', 'text', and 'duration'.
    """

    def __init__(self,
                 manifest_path: str,
                 audio_processor: AudioPreprocessor,
                 text_processor: TextPreprocessor,
                 max_seq_len: int = 1000,
                 augment: bool = False,
                 spec_augment: Optional[SpecAugment] = None):
        self.manifest_path = manifest_path
        self.audio_processor = audio_processor
        self.text_processor = text_processor
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.spec_augment = spec_augment

        self.data = self._load_manifest()
        self.data = [item for item in self.data if item['duration'] * 100 <= max_seq_len]

        print(f"Loaded {len(self.data)} samples from {manifest_path}")

    def _load_manifest(self) -> List[Dict]:
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        with open(self.manifest_path, 'r') as f:
            return [json.loads(line.strip()) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        audio_path = item['audio_path']
        spectrogram = self.audio_processor.preprocess_audio(audio_path)

        if self.augment and self.spec_augment:
            spectrogram = self.spec_augment(spectrogram)

        text = item['text']
        text_indices = self.text_processor.text_to_indices(text)

        if spectrogram.shape[0] > self.max_seq_len:
            spectrogram = spectrogram[:self.max_seq_len]

        return {
            'spectrogram': spectrogram,
            'text': text,
            'text_indices': text_indices,
            'audio_path': audio_path
        }

def create_dataloaders(config, use_huggingface: bool = False) -> Tuple[DataLoader, DataLoader, Dict]:
    audio_processor = AudioPreprocessor(
        sample_rate=config.data.sample_rate,
        n_mels=config.data.n_mels,
        normalize=True
    )
    
    # Set vocab path based on dataset
    if config.data.dataset == "gigaspeech":
        vocab_path = Path(config.data.save_dir) / "vocab.txt"
    else:
        vocab_path = None
    
    text_processor = TextPreprocessor(vocab_path=vocab_path)

    spec_augment = None
    if config.data.augment:
        spec_augment = SpecAugment(
            freq_mask_param=config.data.n_mels // 4,
            time_mask_param=config.data.max_seq_len // 10,
            num_freq_masks=2,
            num_time_masks=2
        )

    manifest_root = Path(config.data.save_dir)
    train_manifest = manifest_root / "train_manifest.json"
    val_manifest = manifest_root / "dev_manifest.json"

    train_dataset = CustomManifestDataset(
        manifest_path=str(train_manifest),
        audio_processor=audio_processor,
        text_processor=text_processor,
        max_seq_len=config.data.max_seq_len,
        augment=config.data.augment,
        spec_augment=spec_augment
    )

    val_dataset = CustomManifestDataset(
        manifest_path=str(val_manifest),
        audio_processor=audio_processor,
        text_processor=text_processor,
        max_seq_len=config.data.max_seq_len,
        augment=False,
        spec_augment=None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False
    )

    vocab = {
        'char_to_idx': text_processor.char_to_idx,
        'idx_to_char': text_processor.idx_to_char,
        'vocab_size': text_processor.get_vocab_size(),
        'blank_token_id': text_processor.get_blank_token_id()
    }

    return train_loader, val_loader, vocab


def create_distributed_dataloaders(config, rank: int, world_size: int) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create distributed data loaders for multi-GPU training.
    
    Args:
        config: Configuration object
        rank: Process rank
        world_size: Total number of processes
        
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        vocab: Vocabulary dictionary
    """
    # Initialize processors
    audio_processor = AudioPreprocessor(
        sample_rate=config.data.sample_rate,
        n_mels=config.data.n_mels,
        normalize=True
    )
    
    # Set vocab path based on dataset
    if config.data.dataset == "gigaspeech":
        vocab_path = Path(config.data.save_dir) / "vocab.txt"
    else:
        vocab_path = None
    
    text_processor = TextPreprocessor(vocab_path=vocab_path)
    
    # Create SpecAugment if augmentation is enabled
    spec_augment = None
    if config.data.augment:
        spec_augment = SpecAugment(
            freq_mask_param=config.data.n_mels // 4,
            time_mask_param=config.data.max_seq_len // 10,
            num_freq_masks=2,
            num_time_masks=2
        )
    
    # Create datasets
    manifest_root = Path(config.data.save_dir)
    train_manifest = manifest_root / "train_manifest.json"
    val_manifest = manifest_root / "dev_manifest.json"

    train_dataset = CustomManifestDataset(
        manifest_path=str(train_manifest),
        audio_processor=audio_processor,
        text_processor=text_processor,
        max_seq_len=config.data.max_seq_len,
        augment=config.data.augment,
        spec_augment=spec_augment
    )
    
    val_dataset = CustomManifestDataset(
        manifest_path=str(val_manifest),
        audio_processor=audio_processor,
        text_processor=text_processor,
        max_seq_len=config.data.max_seq_len,
        augment=False,
        spec_augment=None
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        sampler=val_sampler,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False
    )
    
    # Create vocabulary dictionary
    vocab = {
        'char_to_idx': text_processor.char_to_idx,
        'idx_to_char': text_processor.idx_to_char,
        'vocab_size': text_processor.get_vocab_size(),
        'blank_token_id': text_processor.get_blank_token_id()
    }
    
    return train_loader, val_loader, vocab


def create_test_dataloader(config, use_huggingface: bool = True) -> Tuple[DataLoader, Dict]:
    """
    Create test data loader.
    
    Args:
        config: Configuration object
        use_huggingface: Whether to use HuggingFace datasets
        
    Returns:
        test_loader: Test data loader
        vocab: Vocabulary dictionary
    """
    # Initialize processors
    audio_processor = AudioPreprocessor(
        sample_rate=config.data.sample_rate,
        n_mels=config.data.n_mels,
        normalize=True
    )
    
    # Set vocab path based on dataset
    if config.data.dataset == "gigaspeech":
        vocab_path = Path(config.data.save_dir) / "vocab.txt"
    else:
        vocab_path = None
    
    text_processor = TextPreprocessor(vocab_path=vocab_path)
    
    # Create test dataset from manifest
    manifest_root = Path(config.data.save_dir)
    test_manifest = manifest_root / "test_manifest.json"
    
    test_dataset = CustomManifestDataset(
        manifest_path=str(test_manifest),
        audio_processor=audio_processor,
        text_processor=text_processor,
        max_seq_len=config.data.max_seq_len,
        augment=False,
        spec_augment=None
    )
    
    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False
    )
    
    # Create vocabulary dictionary
    vocab = {
        'char_to_idx': text_processor.char_to_idx,
        'idx_to_char': text_processor.idx_to_char,
        'vocab_size': text_processor.get_vocab_size(),
        'blank_token_id': text_processor.get_blank_token_id()
    }
    
    return test_loader, vocab
