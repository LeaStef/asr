import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import json
import os
from typing import Dict, List, Tuple, Optional
import random
from datasets import load_dataset

from .preprocessing import AudioPreprocessor, TextPreprocessor, SpecAugment, collate_fn


class LibriSpeechDataset(Dataset):
    """
    LibriSpeech dataset for ASR training.
    """
    
    def __init__(self,
                 manifest_path: str,
                 audio_processor: AudioPreprocessor,
                 text_processor: TextPreprocessor,
                 max_seq_len: int = 1000,
                 augment: bool = False,
                 spec_augment: Optional[SpecAugment] = None):
        """
        Initialize LibriSpeech dataset.
        
        Args:
            manifest_path: Path to manifest file
            audio_processor: Audio preprocessing pipeline
            text_processor: Text preprocessing pipeline
            max_seq_len: Maximum sequence length
            augment: Whether to apply data augmentation
            spec_augment: SpecAugment instance for augmentation
        """
        self.manifest_path = manifest_path
        self.audio_processor = audio_processor
        self.text_processor = text_processor
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.spec_augment = spec_augment
        
        # Load manifest data
        self.data = self._load_manifest()
        
        # Filter by sequence length
        self.data = [item for item in self.data if item['duration'] * 100 <= max_seq_len]
        
        print(f"Loaded {len(self.data)} samples from {manifest_path}")
    
    def _load_manifest(self) -> List[Dict]:
        """Load manifest file."""
        manifest_data = []
        
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        
        with open(self.manifest_path, 'r') as f:
            for line in f:
                manifest_data.append(json.loads(line.strip()))
        
        return manifest_data
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get dataset item.
        
        Args:
            idx: Sample index
            
        Returns:
            sample: Dictionary with spectrogram and text data
        """
        item = self.data[idx]
        
        # Load and process audio
        audio_path = item['audio_path']
        spectrogram = self.audio_processor.preprocess_audio(audio_path)
        
        # Apply augmentation if enabled
        if self.augment and self.spec_augment is not None:
            spectrogram = self.spec_augment(spectrogram)
        
        # Process text
        text = item['text']
        text_indices = self.text_processor.text_to_indices(text)
        
        # Truncate if too long
        if spectrogram.shape[0] > self.max_seq_len:
            spectrogram = spectrogram[:self.max_seq_len]
        
        return {
            'spectrogram': spectrogram,
            'text': text,
            'text_indices': text_indices,
            'audio_path': audio_path
        }


class HuggingFaceLibriSpeechDataset(Dataset):
    """
    LibriSpeech dataset using HuggingFace datasets.
    """
    
    def __init__(self,
                 split: str,
                 audio_processor: AudioPreprocessor,
                 text_processor: TextPreprocessor,
                 max_seq_len: int = 1000,
                 augment: bool = False,
                 spec_augment: Optional[SpecAugment] = None,
                 subset: str = "clean"):
        """
        Initialize HuggingFace LibriSpeech dataset.
        
        Args:
            split: Dataset split (train.100, validation, test)
            audio_processor: Audio preprocessing pipeline
            text_processor: Text preprocessing pipeline
            max_seq_len: Maximum sequence length
            augment: Whether to apply data augmentation
            spec_augment: SpecAugment instance for augmentation
            subset: LibriSpeech subset (clean, other)
        """
        self.split = split
        self.audio_processor = audio_processor
        self.text_processor = text_processor
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.spec_augment = spec_augment
        
        # Load dataset from HuggingFace
        self.dataset = load_dataset("librispeech_asr", subset, split=split)
        
        # Filter by duration (approximate)
        self.dataset = self.dataset.filter(lambda x: len(x['audio']['array']) / 16000 * 100 <= max_seq_len)
        
        print(f"Loaded {len(self.dataset)} samples from LibriSpeech {subset} {split}")
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get dataset item.
        
        Args:
            idx: Sample index
            
        Returns:
            sample: Dictionary with spectrogram and text data
        """
        item = self.dataset[idx]
        
        # Extract audio and text
        audio_array = torch.tensor(item['audio']['array']).float()
        text = item['text']
        
        # Process audio
        spectrogram = self.audio_processor.extract_mel_features(audio_array)
        
        # Apply augmentation if enabled
        if self.augment and self.spec_augment is not None:
            spectrogram = self.spec_augment(spectrogram)
        
        # Process text
        text_indices = self.text_processor.text_to_indices(text)
        
        # Truncate if too long
        if spectrogram.shape[0] > self.max_seq_len:
            spectrogram = spectrogram[:self.max_seq_len]
        
        return {
            'spectrogram': spectrogram,
            'text': text,
            'text_indices': text_indices,
            'audio_path': f"hf_librispeech_{idx}"
        }


def create_dataloaders(config, use_huggingface: bool = True) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create train and validation data loaders.
    
    Args:
        config: Configuration object
        use_huggingface: Whether to use HuggingFace datasets
        
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        vocab: Vocabulary dictionary
    """
    # Initialize processors
    audio_processor = AudioPreprocessor(
        sample_rate=config.sample_rate,
        n_mels=config.n_mels,
        normalize=True
    )
    
    text_processor = TextPreprocessor()
    
    # Create SpecAugment if augmentation is enabled
    spec_augment = None
    if config.augment:
        spec_augment = SpecAugment(
            freq_mask_param=config.n_mels // 4,
            time_mask_param=config.max_seq_len // 10,
            num_freq_masks=2,
            num_time_masks=2
        )
    
    if use_huggingface:
        # Create HuggingFace datasets
        train_dataset = HuggingFaceLibriSpeechDataset(
            split="train.100",
            audio_processor=audio_processor,
            text_processor=text_processor,
            max_seq_len=config.max_seq_len,
            augment=config.augment,
            spec_augment=spec_augment,
            subset="clean"
        )
        
        val_dataset = HuggingFaceLibriSpeechDataset(
            split="validation",
            audio_processor=audio_processor,
            text_processor=text_processor,
            max_seq_len=config.max_seq_len,
            augment=False,  # No augmentation for validation
            spec_augment=None,
            subset="clean"
        )
    else:
        # Create datasets from manifest files
        train_dataset = LibriSpeechDataset(
            manifest_path="./data/train_manifest.json",
            audio_processor=audio_processor,
            text_processor=text_processor,
            max_seq_len=config.max_seq_len,
            augment=config.augment,
            spec_augment=spec_augment
        )
        
        val_dataset = LibriSpeechDataset(
            manifest_path="./data/val_manifest.json",
            audio_processor=audio_processor,
            text_processor=text_processor,
            max_seq_len=config.max_seq_len,
            augment=False,
            spec_augment=None
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
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
        sample_rate=config.sample_rate,
        n_mels=config.n_mels,
        normalize=True
    )
    
    text_processor = TextPreprocessor()
    
    # Create SpecAugment if augmentation is enabled
    spec_augment = None
    if config.augment:
        spec_augment = SpecAugment(
            freq_mask_param=config.n_mels // 4,
            time_mask_param=config.max_seq_len // 10,
            num_freq_masks=2,
            num_time_masks=2
        )
    
    # Create datasets
    train_dataset = HuggingFaceLibriSpeechDataset(
        split="train.100",
        audio_processor=audio_processor,
        text_processor=text_processor,
        max_seq_len=config.max_seq_len,
        augment=config.augment,
        spec_augment=spec_augment,
        subset="clean"
    )
    
    val_dataset = HuggingFaceLibriSpeechDataset(
        split="validation",
        audio_processor=audio_processor,
        text_processor=text_processor,
        max_seq_len=config.max_seq_len,
        augment=False,
        spec_augment=None,
        subset="clean"
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
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        num_workers=config.num_workers,
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
        sample_rate=config.sample_rate,
        n_mels=config.n_mels,
        normalize=True
    )
    
    text_processor = TextPreprocessor()
    
    if use_huggingface:
        # Create HuggingFace test dataset
        test_dataset = HuggingFaceLibriSpeechDataset(
            split="test",
            audio_processor=audio_processor,
            text_processor=text_processor,
            max_seq_len=config.max_seq_len,
            augment=False,
            spec_augment=None,
            subset="clean"
        )
    else:
        # Create test dataset from manifest
        test_dataset = LibriSpeechDataset(
            manifest_path="./data/test_manifest.json",
            audio_processor=audio_processor,
            text_processor=text_processor,
            max_seq_len=config.max_seq_len,
            augment=False,
            spec_augment=None
        )
    
    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
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