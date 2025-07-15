import torch
import torch.nn as nn
import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
import jiwer
from pathlib import Path


class EarlyStopping:
    """
    Early stopping utility to monitor validation metrics.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for metrics that should decrease, 'max' for metrics that should increase
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        if mode == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')
    
    def __call__(self, score: float) -> bool:
        """
        Check if early stopping should be triggered.
        
        Args:
            score: Current validation score
            
        Returns:
            early_stop: Whether to stop training
        """
        if self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop


class LearningRateScheduler:
    """
    Learning rate scheduler with warmup and decay.
    """
    
    def __init__(self, optimizer, warmup_steps: int = 1000, max_lr: float = 1e-3, 
                 decay_steps: int = 10000, decay_rate: float = 0.96):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            max_lr: Maximum learning rate
            decay_steps: Steps between decay
            decay_rate: Decay rate
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.step_count = 0
    
    def step(self):
        """Update learning rate."""
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Warmup phase
            lr = self.max_lr * (self.step_count / self.warmup_steps)
        else:
            # Decay phase
            decay_factor = self.decay_rate ** ((self.step_count - self.warmup_steps) // self.decay_steps)
            lr = self.max_lr * decay_factor
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class MetricsTracker:
    """
    Utility to track training and validation metrics.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_wer': [],
            'val_cer': [],
            'learning_rate': []
        }
    
    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def get_latest(self, key: str) -> Optional[float]:
        """Get latest value for a metric."""
        if key in self.metrics and self.metrics[key]:
            return self.metrics[key][-1]
        return None
    
    def get_best(self, key: str, mode: str = 'min') -> Optional[float]:
        """Get best value for a metric."""
        if key in self.metrics and self.metrics[key]:
            if mode == 'min':
                return min(self.metrics[key])
            else:
                return max(self.metrics[key])
        return None
    
    def save(self, path: str):
        """Save metrics to file."""
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load(self, path: str):
        """Load metrics from file."""
        with open(path, 'r') as f:
            self.metrics = json.load(f)


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, metrics: Dict, 
                   checkpoint_path: str, is_best: bool = False):
    """
    Save training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        metrics: Training metrics
        checkpoint_path: Path to save checkpoint
        is_best: Whether this is the best checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model separately
    if is_best:
        best_path = checkpoint_path.replace('.pt', '_best.pt')
        torch.save(checkpoint, best_path)


def load_checkpoint(checkpoint_path: str, model: nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model
        optimizer: Optimizer (optional)
        
    Returns:
        checkpoint: Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def compute_wer(predictions: List[str], targets: List[str]) -> float:
    """
    Compute Word Error Rate (WER).
    
    Args:
        predictions: List of predicted text strings
        targets: List of target text strings
        
    Returns:
        wer: Word Error Rate
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    # Filter out empty strings
    filtered_preds = []
    filtered_targets = []
    
    for pred, target in zip(predictions, targets):
        if target.strip():  # Only include non-empty targets
            filtered_preds.append(pred)
            filtered_targets.append(target)
    
    if not filtered_targets:
        return 0.0
    
    return jiwer.wer(filtered_targets, filtered_preds)


def compute_cer(predictions: List[str], targets: List[str]) -> float:
    """
    Compute Character Error Rate (CER).
    
    Args:
        predictions: List of predicted text strings
        targets: List of target text strings
        
    Returns:
        cer: Character Error Rate
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    # Filter out empty strings
    filtered_preds = []
    filtered_targets = []
    
    for pred, target in zip(predictions, targets):
        if target.strip():  # Only include non-empty targets
            filtered_preds.append(pred)
            filtered_targets.append(target)
    
    if not filtered_targets:
        return 0.0
    
    return jiwer.cer(filtered_targets, filtered_preds)


def decode_predictions(predictions: List[List[int]], vocab: Dict) -> List[str]:
    """
    Decode predictions to text strings.
    
    Args:
        predictions: List of predicted token sequences
        vocab: Vocabulary dictionary
        
    Returns:
        decoded_texts: List of decoded text strings
    """
    idx_to_char = vocab['idx_to_char']
    decoded_texts = []
    
    for pred in predictions:
        chars = []
        for idx in pred:
            if idx in idx_to_char:
                char = idx_to_char[idx]
                if char != '<blank>':
                    chars.append(char)
        
        decoded_texts.append(''.join(chars))
    
    return decoded_texts


def decode_targets(targets: torch.Tensor, target_lengths: torch.Tensor, vocab: Dict) -> List[str]:
    """
    Decode target sequences to text strings.
    
    Args:
        targets: Target tensor
        target_lengths: Target lengths
        vocab: Vocabulary dictionary
        
    Returns:
        decoded_texts: List of decoded text strings
    """
    idx_to_char = vocab['idx_to_char']
    decoded_texts = []
    
    for i, length in enumerate(target_lengths):
        target_seq = targets[i, :length]
        chars = []
        
        for idx in target_seq:
            idx = idx.item()
            if idx in idx_to_char:
                char = idx_to_char[idx]
                if char != '<blank>':
                    chars.append(char)
        
        decoded_texts.append(''.join(chars))
    
    return decoded_texts


def calculate_model_size(model: nn.Module) -> Tuple[int, int]:
    """
    Calculate model size in parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def setup_logging(log_dir: str, rank: int = 0) -> None:
    """
    Set up logging directory.
    
    Args:
        log_dir: Directory for logs
        rank: Process rank (for distributed training)
    """
    if rank == 0:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (Path(log_dir) / 'checkpoints').mkdir(exist_ok=True)
        (Path(log_dir) / 'logs').mkdir(exist_ok=True)
        (Path(log_dir) / 'metrics').mkdir(exist_ok=True)


def log_model_summary(model: nn.Module, input_shape: Tuple[int, ...], 
                     device: torch.device) -> None:
    """
    Log model summary.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device
    """
    total_params, trainable_params = calculate_model_size(model)
    
    print(f"Model Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1e6:.2f} MB")
    
    # Test forward pass
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_shape).to(device)
        try:
            output = model(dummy_input)
            if isinstance(output, tuple):
                output = output[0]
            print(f"  Output shape: {output.shape}")
        except Exception as e:
            print(f"  Forward pass test failed: {e}")


def gradient_clipping(model: nn.Module, max_norm: float = 1.0) -> float:
    """
    Apply gradient clipping to model parameters.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
        
    Returns:
        total_norm: Total gradient norm before clipping
    """
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return total_norm.item()


def count_correct_predictions(predictions: List[List[int]], targets: List[List[int]]) -> Tuple[int, int]:
    """
    Count correct predictions at token level.
    
    Args:
        predictions: List of predicted token sequences
        targets: List of target token sequences
        
    Returns:
        correct: Number of correct tokens
        total: Total number of tokens
    """
    correct = 0
    total = 0
    
    for pred, target in zip(predictions, targets):
        min_len = min(len(pred), len(target))
        
        for i in range(min_len):
            if pred[i] == target[i]:
                correct += 1
            total += 1
        
        # Count remaining tokens as incorrect
        total += abs(len(pred) - len(target))
    
    return correct, total