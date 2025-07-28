import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import os
import time
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np

from .utils import (
    EarlyStopping, LearningRateScheduler, MetricsTracker, 
    save_checkpoint, load_checkpoint, compute_wer, compute_cer,
    decode_predictions, decode_targets, gradient_clipping,
    log_model_summary, setup_logging
)


class Trainer:
    """
    Single GPU trainer for LMU ASR model.
    """
    
    def __init__(self, model: nn.Module, training_config, data_config, device: torch.device, 
                 log_dir: str = './logs'):
        """
        Initialize trainer.
        
        Args:
            model: LMU ASR model
            config: Configuration object
            device: Training device
            log_dir: Directory for logs and checkpoints
        """
        self.model = model
        self.config = training_config
        self.device = device
        self.log_dir = log_dir
        
        # Setup logging
        setup_logging(log_dir)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize learning rate scheduler
        self.lr_scheduler = LearningRateScheduler(
            self.optimizer,
            warmup_steps=getattr(training_config, 'warmup_steps', 1000),
            max_lr=training_config.lr,
            decay_steps=getattr(training_config, 'decay_steps', 10000),
            decay_rate=getattr(training_config, 'decay_rate', 0.96)
        )
        
        # Initialize mixed precision scaler
        self.scaler = GradScaler() if training_config.mixed_precision else None
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=training_config.patience,
            min_delta=0.001,
            mode='min'
        )
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_wer = float('inf')
        
        # Log model summary
        log_model_summary(model, (data_config.max_seq_len, data_config.n_mels), device)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        if hasattr(self.config, 'optimizer') and self.config.optimizer == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                betas=(0.9, 0.999),
                weight_decay=getattr(self.config, 'weight_decay', 0.01)
            )
        else:
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.lr,
                betas=(0.9, 0.999),
                weight_decay=getattr(self.config, 'weight_decay', 0.0)
            )
    
    def train_epoch(self, train_loader, vocab: Dict) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            vocab: Vocabulary dictionary
            
        Returns:
            avg_loss: Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            spectrograms, targets, input_lengths, target_lengths = batch
            
            # Move to device
            spectrograms = spectrograms.to(self.device)
            targets = targets.to(self.device)
            input_lengths = input_lengths.to(self.device)
            target_lengths = target_lengths.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    log_probs, _ = self.model(spectrograms, input_lengths)
                    loss = self.model.compute_loss(log_probs, targets, input_lengths, target_lengths)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if hasattr(self.config, 'gradient_clip_norm'):
                    self.scaler.unscale_(self.optimizer)
                    gradient_clipping(self.model, self.config.gradient_clip_norm)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular forward pass
                log_probs, _ = self.model(spectrograms, input_lengths)
                loss = self.model.compute_loss(log_probs, targets, input_lengths, target_lengths)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if hasattr(self.config, 'gradient_clip_norm'):
                    gradient_clipping(self.model, self.config.gradient_clip_norm)
                
                # Optimizer step
                self.optimizer.step()
            
            # Update learning rate
            current_lr = self.lr_scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Avg Loss': f"{total_loss / num_batches:.4f}",
                'LR': f"{current_lr:.6f}"
            })
            
            # Log metrics
            if self.global_step % getattr(self.config, 'log_interval', 100) == 0:
                self.metrics_tracker.update(
                    train_loss=loss.item(),
                    learning_rate=current_lr
                )
        
        return total_loss / num_batches
    
    def validate(self, val_loader, vocab: Dict) -> Tuple[float, float, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            vocab: Vocabulary dictionary
            
        Returns:
            avg_loss: Average validation loss
            wer: Word Error Rate
            cer: Character Error Rate
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            
            for batch in progress_bar:
                spectrograms, targets, input_lengths, target_lengths = batch
                
                # Move to device
                spectrograms = spectrograms.to(self.device)
                targets = targets.to(self.device)
                input_lengths = input_lengths.to(self.device)
                target_lengths = target_lengths.to(self.device)
                
                # Forward pass
                log_probs, _ = self.model(spectrograms, input_lengths)
                loss = self.model.compute_loss(log_probs, targets, input_lengths, target_lengths)
                
                # Decode predictions
                predictions = self.model.decode(log_probs, input_lengths)
                
                # Convert to text
                pred_texts = decode_predictions(predictions, vocab)
                target_texts = decode_targets(targets, target_lengths, vocab)
                
                all_predictions.extend(pred_texts)
                all_targets.extend(target_texts)
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Avg Loss': f"{total_loss / num_batches:.4f}"
                })
        
        # Compute metrics
        avg_loss = total_loss / num_batches
        wer = compute_wer(all_predictions, all_targets)
        cer = compute_cer(all_predictions, all_targets)
        
        return avg_loss, wer, cer
    
    def fit(self, train_loader, val_loader, vocab: Dict):
        """
        Training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            vocab: Vocabulary dictionary
        """
        print(f"Starting training for {self.config.max_epochs} epochs...")
        print(f"Training on {len(train_loader.dataset)} samples")
        print(f"Validating on {len(val_loader.dataset)} samples")
        
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Training
            print(f"\nEpoch {epoch + 1}/{self.config.max_epochs}")
            train_loss = self.train_epoch(train_loader, vocab)
            
            # Validation
            val_loss, val_wer, val_cer = self.validate(val_loader, vocab)
            
            epoch_time = time.time() - start_time
            
            # Update metrics
            self.metrics_tracker.update(
                train_loss=train_loss,
                val_loss=val_loss,
                val_wer=val_wer,
                val_cer=val_cer
            )
            
            # Print epoch summary
            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val WER: {val_wer:.4f}")
            print(f"  Val CER: {val_cer:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            
            # Save checkpoint
            is_best = val_wer < self.best_val_wer
            if is_best:
                self.best_val_wer = val_wer
                print(f"  New best WER: {val_wer:.4f}")
            
            checkpoint_path = os.path.join(self.log_dir, 'checkpoints', f'checkpoint_epoch_{epoch + 1}.pt')
            save_checkpoint(
                self.model, self.optimizer, epoch + 1, val_loss,
                self.metrics_tracker.metrics, checkpoint_path, is_best
            )
            
            # Save metrics
            metrics_path = os.path.join(self.log_dir, 'metrics', 'training_metrics.json')
            self.metrics_tracker.save(metrics_path)
            
            # Early stopping
            if self.early_stopping(val_wer):
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Print validation examples
            if epoch % getattr(self.config, 'example_interval', 5) == 0:
                self._print_examples(val_loader, vocab, num_examples=3)
        
        print(f"\nTraining completed!")
        print(f"Best validation WER: {self.best_val_wer:.4f}")
    
    def _print_examples(self, val_loader, vocab: Dict, num_examples: int = 3):
        """Print validation examples."""
        self.model.eval()
        examples_printed = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if examples_printed >= num_examples:
                    break
                
                spectrograms, targets, input_lengths, target_lengths = batch
                
                # Move to device
                spectrograms = spectrograms.to(self.device)
                targets = targets.to(self.device)
                input_lengths = input_lengths.to(self.device)
                target_lengths = target_lengths.to(self.device)
                
                # Forward pass
                log_probs, _ = self.model(spectrograms, input_lengths)
                
                # Decode predictions
                predictions = self.model.decode(log_probs, input_lengths)
                
                # Convert to text
                pred_texts = decode_predictions(predictions, vocab)
                target_texts = decode_targets(targets, target_lengths, vocab)
                
                # Print examples
                for i in range(min(num_examples - examples_printed, len(pred_texts))):
                    print(f"\nExample {examples_printed + 1}:")
                    print(f"  Target:     '{target_texts[i]}'")
                    print(f"  Prediction: '{pred_texts[i]}'")
                    examples_printed += 1
                    
                    if examples_printed >= num_examples:
                        break
    
    def resume_training(self, checkpoint_path: str, train_loader, val_loader, vocab: Dict):
        """
        Resume training from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            train_loader: Training data loader
            val_loader: Validation data loader
            vocab: Vocabulary dictionary
        """
        print(f"Resuming training from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = load_checkpoint(checkpoint_path, self.model, self.optimizer)
        
        # Restore training state
        self.epoch = checkpoint['epoch']
        self.global_step = self.epoch * len(train_loader)
        
        # Load metrics if available
        if 'metrics' in checkpoint:
            self.metrics_tracker.metrics = checkpoint['metrics']
            self.best_val_wer = self.metrics_tracker.get_best('val_wer', mode='min')
        
        print(f"Resumed from epoch {self.epoch}")
        print(f"Current best WER: {self.best_val_wer:.4f}")
        
        # Continue training
        self.fit(train_loader, val_loader, vocab)
    
    def evaluate(self, test_loader, vocab: Dict) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            vocab: Vocabulary dictionary
            
        Returns:
            results: Evaluation results
        """
        print("Evaluating on test set...")
        
        test_loss, test_wer, test_cer = self.validate(test_loader, vocab)
        
        results = {
            'test_loss': test_loss,
            'test_wer': test_wer,
            'test_cer': test_cer
        }
        
        print(f"Test Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test WER: {test_wer:.4f}")
        print(f"  Test CER: {test_cer:.4f}")
        
        # Save results
        results_path = os.path.join(self.log_dir, 'test_results.json')
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results