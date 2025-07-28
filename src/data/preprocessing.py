import torch
import torchaudio
import librosa
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import random


class AudioPreprocessor:
    """
    Audio preprocessing utilities for ASR.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_mels: int = 80,
                 n_fft: int = 400,
                 hop_length: int = 160,
                 win_length: int = 400,
                 normalize: bool = True):
        """
        Initialize audio preprocessor.
        
        Args:
            sample_rate: Target sample rate
            n_mels: Number of mel filterbank channels
            n_fft: FFT size
            hop_length: Hop length for STFT
            win_length: Window length for STFT
            normalize: Whether to normalize features
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalize = normalize
        
        # Create mel spectrogram transform
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            power=2.0
        )
        
        # Amplitude to DB transform
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load audio file and resample if necessary.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            waveform: Audio waveform tensor
        """
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform.squeeze(0)  # Remove channel dimension
    
    def extract_mel_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract mel spectrogram features from waveform.
        
        Args:
            waveform: Audio waveform tensor
            
        Returns:
            mel_features: Mel spectrogram features
        """
        # Compute mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        
        # Convert to log scale
        log_mel = self.amplitude_to_db(mel_spec)
        
        # Transpose to (time, frequency) format
        log_mel = log_mel.transpose(0, 1)
        
        # Normalize if requested
        if self.normalize:
            log_mel = self._normalize_features(log_mel)
        
        return log_mel
    
    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Normalize features to zero mean and unit variance.
        
        Args:
            features: Input features
            
        Returns:
            normalized_features: Normalized features
        """
        mean = torch.mean(features, dim=0, keepdim=True)
        std = torch.std(features, dim=0, keepdim=True)
        
        # Avoid division by zero
        std = torch.where(std == 0, torch.ones_like(std), std)
        
        return (features - mean) / std
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        Complete audio preprocessing pipeline.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            features: Processed mel spectrogram features
        """
        waveform = self.load_audio(audio_path)
        features = self.extract_mel_features(waveform)
        return features


class TextPreprocessor:
    """
    Text preprocessing utilities for ASR.
    """
    
    def __init__(self, vocab_path: Optional[str] = None):
        """
        Initialize text preprocessor.
        
        Args:
            vocab_path: Path to vocabulary file (optional)
        """
        self.vocab_path = vocab_path
        self.char_to_idx = {}
        self.idx_to_char = {}
        
        if vocab_path:
            self.load_vocabulary(vocab_path)
        else:
            self.create_default_vocabulary()
    
    def create_default_vocabulary(self):
        """Create default character-level vocabulary."""
        # Standard English characters + space + apostrophe
        chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + [' ', "'"]
        
        # Add CTC blank token
        chars.append('<blank>')
        
        # Create mappings
        self.char_to_idx = {char: i for i, char in enumerate(chars)}
        self.idx_to_char = {i: char for i, char in enumerate(chars)}
        
        print(f"Created vocabulary with {len(chars)} characters")
    
    def load_vocabulary(self, vocab_path: str):
        """
        Load vocabulary from file.
        
        Args:
            vocab_path: Path to vocabulary file
        """
        with open(vocab_path, 'r') as f:
            chars = [line.strip() for line in f]
        
        self.char_to_idx = {char: i for i, char in enumerate(chars)}
        self.idx_to_char = {i: char for i, char in enumerate(chars)}
        
        print(f"Loaded vocabulary with {len(chars)} characters from {vocab_path}")
    
    def text_to_indices(self, text: str) -> List[int]:
        """
        Convert text to list of character indices.
        
        Args:
            text: Input text string
            
        Returns:
            indices: List of character indices
        """
        # Convert to uppercase and handle unknown characters
        text = text.upper()
        indices = []
        
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                # Skip unknown characters (silently for spaces to avoid spam)
                if char != ' ' or len(indices) == 0:  # Only warn for first space or non-space chars
                    pass  # Silent skip
                continue
        
        return indices
    
    def indices_to_text(self, indices: List[int]) -> str:
        """
        Convert list of indices to text.
        
        Args:
            indices: List of character indices
            
        Returns:
            text: Decoded text string
        """
        chars = []
        for idx in indices:
            if idx in self.idx_to_char:
                char = self.idx_to_char[idx]
                if char != '<blank>':  # Skip blank tokens
                    chars.append(char)
        
        return ''.join(chars)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.char_to_idx)
    
    def get_blank_token_id(self) -> int:
        """Get CTC blank token ID."""
        return self.char_to_idx.get('<blank>', len(self.char_to_idx) - 1)


class SpecAugment:
    """
    SpecAugment data augmentation for mel spectrograms.
    """
    
    def __init__(self,
                 freq_mask_param: int = 80,
                 time_mask_param: int = 100,
                 num_freq_masks: int = 2,
                 num_time_masks: int = 2,
                 mask_value: float = 0.0):
        """
        Initialize SpecAugment.
        
        Args:
            freq_mask_param: Maximum frequency mask size
            time_mask_param: Maximum time mask size
            num_freq_masks: Number of frequency masks
            num_time_masks: Number of time masks
            mask_value: Value to use for masking
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.mask_value = mask_value
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrogram.
        
        Args:
            spectrogram: Input spectrogram of shape (time, freq)
            
        Returns:
            augmented_spectrogram: Augmented spectrogram
        """
        spec = spectrogram.clone()
        time_steps, freq_bins = spec.shape
        
        # Apply frequency masking
        for _ in range(self.num_freq_masks):
            freq_mask_size = random.randint(0, min(self.freq_mask_param, freq_bins))
            freq_mask_start = random.randint(0, freq_bins - freq_mask_size)
            spec[:, freq_mask_start:freq_mask_start + freq_mask_size] = self.mask_value
        
        # Apply time masking
        for _ in range(self.num_time_masks):
            time_mask_size = random.randint(0, min(self.time_mask_param, time_steps))
            time_mask_start = random.randint(0, time_steps - time_mask_size)
            spec[time_mask_start:time_mask_start + time_mask_size, :] = self.mask_value
        
        return spec


def collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        spectrograms: Padded spectrograms tensor
        texts: Padded text tensor
        input_lengths: Input sequence lengths
        target_lengths: Target sequence lengths
    """
    # Sort batch by input length (descending)
    batch.sort(key=lambda x: x['spectrogram'].shape[0], reverse=True)
    
    # Extract components
    spectrograms = [item['spectrogram'] for item in batch]
    texts = [item['text_indices'] for item in batch]
    
    # Get lengths
    input_lengths = torch.tensor([spec.shape[0] for spec in spectrograms])
    target_lengths = torch.tensor([len(text) for text in texts])
    
    # Pad spectrograms
    max_input_len = max(spec.shape[0] for spec in spectrograms)
    n_mels = spectrograms[0].shape[1]
    
    padded_spectrograms = torch.zeros(len(batch), max_input_len, n_mels)
    for i, spec in enumerate(spectrograms):
        padded_spectrograms[i, :spec.shape[0], :] = spec
    
    # Pad text sequences
    max_target_len = max(len(text) for text in texts)
    padded_texts = torch.zeros(len(batch), max_target_len, dtype=torch.long)
    for i, text in enumerate(texts):
        padded_texts[i, :len(text)] = torch.tensor(text)
    
    return padded_spectrograms, padded_texts, input_lengths, target_lengths


def load_manifest(manifest_path: str) -> List[Dict]:
    """
    Load data manifest file.
    
    Args:
        manifest_path: Path to manifest file
        
    Returns:
        manifest_data: List of sample dictionaries
    """
    manifest_data = []
    with open(manifest_path, 'r') as f:
        for line in f:
            manifest_data.append(json.loads(line.strip()))
    
    return manifest_data


def compute_mean_std(data_loader, feature_dim: int = 80) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and standard deviation of features.
    
    Args:
        data_loader: DataLoader for computing statistics
        feature_dim: Feature dimension
        
    Returns:
        mean: Feature mean
        std: Feature standard deviation
    """
    mean = torch.zeros(feature_dim)
    std = torch.zeros(feature_dim)
    total_samples = 0
    
    for batch in data_loader:
        spectrograms, _, input_lengths, _ = batch
        
        for i, length in enumerate(input_lengths):
            spec = spectrograms[i, :length]
            mean += spec.sum(0)
            std += (spec ** 2).sum(0)
            total_samples += length
    
    mean /= total_samples
    std = torch.sqrt(std / total_samples - mean ** 2)
    
    return mean, std