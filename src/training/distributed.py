import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
import os
import time
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np

from config.distributed_config import (
    get_rank, get_world_size, is_main_process, reduce_tensor
)
from training.utils import (
    EarlyStopping, LearningRateScheduler, MetricsTracker, 
    save_checkpoint, load_checkpoint, compute_wer, compute_cer,
    decode_predictions, decode_targets, gradient_clipping,
    log_model_summary, setup_logging, get_model_stats
)


class DistributedTrainer:
    """
    Distributed trainer for LMU ASR model.
    """
    
    def __init__(self, model: nn.Module, config, rank: int = 0, world_size: int = 1, 
                 log_dir: str = './logs'):
        """
        Initialize distributed trainer.
        
        Args:
            model: LMU ASR model (should be wrapped with DDP)
            config: Configuration object (full config with training, data, model)
            rank: Process rank
            world_size: Total number of processes
            log_dir: Directory for logs and checkpoints
        """
        self.model = model
        self.config = config
        self.training_config = config.training if hasattr(config, 'training') else config
        self.data_config = config.data if hasattr(config, 'data') else None
        self.rank = rank
        self.world_size = world_size
        self.log_dir = log_dir
        self.device = torch.device(f'cuda:{rank}')
        
        # Setup logging (only on main process)
        if is_main_process():
            setup_logging(log_dir, rank)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize learning rate scheduler
        self.lr_scheduler = LearningRateScheduler(
            self.optimizer,
            warmup_steps=getattr(self.training_config, 'warmup_steps', 1000),
            max_lr=self.training_config.lr,
            decay_steps=getattr(self.training_config, 'decay_steps', 10000),
            decay_rate=getattr(self.training_config, 'decay_rate', 0.96)
        )
        
        # Initialize mixed precision scaler
        self.scaler = GradScaler('cuda') if self.training_config.mixed_precision else None
        
        # Initialize early stopping (only on main process)
        self.early_stopping = EarlyStopping(
            patience=self.training_config.patience,
            min_delta=0.001,
            mode='min'
        ) if is_main_process() else None
        
        # Initialize metrics tracker (only on main process)
        self.metrics_tracker = MetricsTracker() if is_main_process() else None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_wer = float('inf')
        
        # Log model summary (only on main process)
        if is_main_process():
            # Get the underlying model (handle both DDP and non-DDP cases)
            underlying_model = model.module if hasattr(model, 'module') else model
            if self.data_config:
                log_model_summary(underlying_model, (self.data_config.max_seq_len, self.data_config.n_mels), self.device)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        if hasattr(self.training_config, 'optimizer') and self.training_config.optimizer == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.training_config.lr,
                betas=(0.9, 0.999),
                weight_decay=getattr(self.training_config, 'weight_decay', 0.01)
            )
        else:
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.training_config.lr,
                betas=(0.9, 0.999),
                weight_decay=getattr(self.training_config, 'weight_decay', 0.0)
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
        
        # Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(self.epoch)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch + 1}", 
                          disable=not is_main_process())
        
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
                with autocast('cuda'):
                    log_probs, _ = self.model(spectrograms, input_lengths)
                    # Get underlying model for loss computation
                    underlying_model = self.model.module if hasattr(self.model, 'module') else self.model
                    loss = underlying_model.compute_loss(log_probs, targets, input_lengths, target_lengths)
                
                # Check for NaN loss
                if torch.isnan(loss).any():
                    if is_main_process():
                        print(f"WARNING: NaN loss detected at step {self.global_step}, skipping batch")
                    self.optimizer.zero_grad()  # Clear gradients before continuing
                    continue
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping and monitoring
                grad_norm = None
                skip_step = False
                
                if hasattr(self.training_config, 'gradient_clip_norm'):
                    self.scaler.unscale_(self.optimizer)
                    # Calculate gradient norm before clipping
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    grad_norm = total_norm ** (1. / 2)
                    
                    # Check for NaN gradients
                    if torch.isnan(torch.tensor(grad_norm)):
                        if is_main_process():
                            print(f"WARNING: NaN gradients detected at step {self.global_step}, skipping batch")
                        skip_step = True
                    else:
                        gradient_clipping(self.model, self.training_config.gradient_clip_norm)
                
                # Optimizer step
                if skip_step:
                    # Must call step and update even when skipping to reset scaler state
                    self.scaler.step(self.optimizer)  # This will be a no-op due to NaN gradients
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    continue
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                # Regular forward pass
                log_probs, _ = self.model(spectrograms, input_lengths)
                # Get underlying model for loss computation
                underlying_model = self.model.module if hasattr(self.model, 'module') else self.model
                loss = underlying_model.compute_loss(log_probs, targets, input_lengths, target_lengths)
                
                # Check for NaN loss
                if torch.isnan(loss).any():
                    if is_main_process():
                        print(f"WARNING: NaN loss detected at step {self.global_step}, skipping batch")
                    self.optimizer.zero_grad()
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping and monitoring
                grad_norm = None
                skip_step = False
                
                if hasattr(self.training_config, 'gradient_clip_norm'):
                    # Calculate gradient norm before clipping
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    grad_norm = total_norm ** (1. / 2)
                    
                    # Check for NaN gradients
                    if torch.isnan(torch.tensor(grad_norm)):
                        if is_main_process():
                            print(f"WARNING: NaN gradients detected at step {self.global_step}, skipping batch")
                        skip_step = True
                    else:
                        gradient_clipping(self.model, self.training_config.gradient_clip_norm)
                
                # Optimizer step
                if not skip_step:
                    self.optimizer.step()
            
            # Update learning rate
            current_lr = self.lr_scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar (only on main process)
            if is_main_process():
                avg_loss_display = total_loss / num_batches if num_batches > 0 else 0.0
                postfix = {
                    'Loss': f"{loss.item():.4f}",
                    'Avg Loss': f"{avg_loss_display:.4f}",
                    'LR': f"{current_lr:.6f}"
                }
                if grad_norm is not None:
                    postfix['Grad Norm'] = f"{grad_norm:.4f}"
                progress_bar.set_postfix(postfix)
            
            # Log metrics (only on main process)
            if is_main_process() and self.global_step % getattr(self.training_config, 'log_interval', 100) == 0:
                self.metrics_tracker.update(
                    train_loss=loss.item(),
                    learning_rate=current_lr
                )
        
        # Reduce loss across all processes
        if num_batches == 0:
            # Handle case where all batches were skipped due to NaN losses/gradients
            if is_main_process():
                print(f"WARNING: All batches in epoch {self.epoch + 1} were skipped due to NaN losses/gradients")
            avg_loss = 0.0
        else:
            avg_loss = total_loss / num_batches
        
        avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
        avg_loss_tensor = reduce_tensor(avg_loss_tensor, self.world_size)
        
        return avg_loss_tensor.item()
    
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
            progress_bar = tqdm(val_loader, desc="Validation", 
                              disable=not is_main_process())
            
            for batch in progress_bar:
                spectrograms, targets, input_lengths, target_lengths = batch
                
                # Move to device
                spectrograms = spectrograms.to(self.device)
                targets = targets.to(self.device)
                input_lengths = input_lengths.to(self.device)
                target_lengths = target_lengths.to(self.device)
                
                # Forward pass
                log_probs, _ = self.model(spectrograms, input_lengths)
                # Get underlying model for loss and decode
                underlying_model = self.model.module if hasattr(self.model, 'module') else self.model
                loss = underlying_model.compute_loss(log_probs, targets, input_lengths, target_lengths)
                
                # Decode predictions
                predictions = underlying_model.decode(log_probs, input_lengths)
                
                # Convert to text
                pred_texts = decode_predictions(predictions, vocab)
                target_texts = decode_targets(targets, target_lengths, vocab)
                
                all_predictions.extend(pred_texts)
                all_targets.extend(target_texts)
                
                total_loss += loss.item()
                num_batches += 1
                
                if is_main_process():
                    avg_loss_display = total_loss / num_batches if num_batches > 0 else 0.0
                    progress_bar.set_postfix({
                        'Loss': f"{loss.item():.4f}",
                        'Avg Loss': f"{avg_loss_display:.4f}"
                    })
        
        # Reduce loss across all processes
        if num_batches == 0:
            # Handle case where all validation batches were skipped
            if is_main_process():
                print(f"WARNING: All validation batches were skipped")
            avg_loss = 0.0
        else:
            avg_loss = total_loss / num_batches
        
        avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
        avg_loss_tensor = reduce_tensor(avg_loss_tensor, self.world_size)
        
        # Gather predictions and targets from all processes
        all_predictions = self._gather_predictions(all_predictions)
        all_targets = self._gather_predictions(all_targets)
        
        # Compute metrics (only on main process)
        if is_main_process():
            wer = compute_wer(all_predictions, all_targets)
            cer = compute_cer(all_predictions, all_targets)
        else:
            wer = 0.0
            cer = 0.0
        
        # Broadcast metrics to all processes
        wer_tensor = torch.tensor(wer, device=self.device)
        cer_tensor = torch.tensor(cer, device=self.device)
        
        if dist.is_initialized():
            dist.broadcast(wer_tensor, 0)
            dist.broadcast(cer_tensor, 0)
        
        return avg_loss_tensor.item(), wer_tensor.item(), cer_tensor.item()
    
    def _gather_predictions(self, predictions: List[str]) -> List[str]:
        """Gather predictions from all processes."""
        if not dist.is_initialized():
            return predictions
        
        # Convert to tensor format for gathering
        gathered_predictions = [None] * self.world_size
        
        if dist.is_initialized():
            dist.all_gather_object(gathered_predictions, predictions)
        
        # Flatten list of lists
        all_predictions = []
        for pred_list in gathered_predictions:
            all_predictions.extend(pred_list)
        
        return all_predictions
    
    def fit(self, train_loader, val_loader, vocab: Dict):
        """
        Training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            vocab: Vocabulary dictionary
        """
        if is_main_process():
            print(f"Starting distributed training for {self.training_config.max_epochs} epochs...")
            print(f"Training on {len(train_loader.dataset)} samples")
            print(f"Validating on {len(val_loader.dataset)} samples")
            print(f"Using {self.world_size} GPUs")
        
        for epoch in range(self.training_config.max_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Training
            if is_main_process():
                print(f"\nEpoch {epoch + 1}/{self.training_config.max_epochs}")
            
            train_loss = self.train_epoch(train_loader, vocab)
            
            # Validation
            val_loss, val_wer, val_cer = self.validate(val_loader, vocab)
            
            epoch_time = time.time() - start_time
            
            # Update metrics (only on main process)
            if is_main_process():
                self.metrics_tracker.update(
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_wer=val_wer,
                    val_cer=val_cer
                )
                
                # Get model statistics with input shape for torchinfo
                input_shape = None
                if self.data_config:
                    input_shape = (1, self.data_config.max_seq_len, self.data_config.n_mels)
                model_stats = get_model_stats(self.model, input_shape)
                
                # Print epoch summary
                print(f"Epoch {epoch + 1} Summary:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val WER: {val_wer:.4f}")
                print(f"  Val CER: {val_cer:.4f}")
                print(f"  Time: {epoch_time:.2f}s")
                print(f"  Model: {model_stats['total_params']:,} params ({model_stats['model_size_mb']:.1f}MB)")
                
                # Enhanced breakdown with attention params
                if model_stats['attention_params'] > 0:
                    print(f"    ├─ Encoder: {model_stats['encoder_params']:,} params ({model_stats['encoder_params']/model_stats['total_params']*100:.1f}%)")
                    print(f"    ├─ Attention: {model_stats['attention_params']:,} params ({model_stats['attention_params']/model_stats['total_params']*100:.1f}%)")
                    print(f"    └─ Decoder: {model_stats['decoder_params']:,} params ({model_stats['decoder_params']/model_stats['total_params']*100:.1f}%)")
                else:
                    print(f"    ├─ Encoder: {model_stats['encoder_params']:,} params ({model_stats['encoder_params']/model_stats['total_params']*100:.1f}%)")
                    print(f"    └─ Decoder: {model_stats['decoder_params']:,} params ({model_stats['decoder_params']/model_stats['total_params']*100:.1f}%)")
                
                # Add comprehensive torchinfo statistics if available
                if 'model_flops' in model_stats:
                    print(f"  FLOPs: {model_stats['model_flops']:,} ({model_stats['model_flops']/1e9:.2f}G)")
                
                if 'model_size_mb' in model_stats:
                    print(f"  Model Size: {model_stats['model_size_mb']:.1f} MB")
                
                if 'total_memory_estimate_mb' in model_stats:
                    print(f"  Est. Training Memory: {model_stats['total_memory_estimate_mb']:.1f} MB")
                
                # Print detailed torchinfo summary on first epoch
                if epoch == 0 and 'torchinfo_summary' in model_stats:
                    print(f"\n📋 Detailed Model Architecture (torchinfo):")
                    print(model_stats['torchinfo_summary'])
                    
                    # Print layer breakdown if available
                    if 'layer_breakdown' in model_stats:
                        print(f"\n📊 Top Parameter-Heavy Layers:")
                        layer_breakdown = model_stats['layer_breakdown']
                        # Sort by parameter count and show top 5
                        sorted_layers = sorted(
                            layer_breakdown.items(),
                            key=lambda x: x[1]['params'],
                            reverse=True
                        )[:5]
                        for layer_name, info in sorted_layers:
                            if info['params'] > 0:
                                print(f"  {layer_name}: {info['params']:,} params, {info['flops']:,} FLOPs")
                
                # Show fallback info if main torchinfo failed
                if epoch == 0 and 'torchinfo_error' in model_stats:
                    print(f"\n⚠️  torchinfo error: {model_stats['torchinfo_error']}")
                    if 'torchinfo_fallback' in model_stats:
                        print("📋 Basic Model Summary (fallback):")
                        print(model_stats['torchinfo_fallback'])
            
            # Save checkpoint (only on main process)
            if is_main_process():
                is_best = val_wer < self.best_val_wer
                if is_best:
                    self.best_val_wer = val_wer
                    print(f"  New best WER: {val_wer:.4f}")
                
                checkpoint_path = os.path.join(self.log_dir, 'checkpoints', f'checkpoint_epoch_{epoch + 1}.pt')
                # Get underlying model for checkpoint saving
                underlying_model = self.model.module if hasattr(self.model, 'module') else self.model
                save_checkpoint(
                    underlying_model, self.optimizer, epoch + 1, val_loss,
                    self.metrics_tracker.metrics, checkpoint_path, is_best
                )
                
                # Save metrics
                metrics_path = os.path.join(self.log_dir, 'metrics', 'training_metrics.json')
                self.metrics_tracker.save(metrics_path)
            
            # Early stopping (only on main process)
            if is_main_process() and self.early_stopping(val_wer):
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Synchronize all processes
            if dist.is_initialized():
                dist.barrier()
            
            # Print validation examples (only on main process)
            if is_main_process() and epoch % getattr(self.training_config, 'example_interval', 5) == 0:
                self._print_examples(val_loader, vocab, num_examples=3)
        
        if is_main_process():
            print(f"\nDistributed training completed!")
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
                underlying_model = self.model.module if hasattr(self.model, 'module') else self.model
                predictions = underlying_model.decode(log_probs, input_lengths)
                
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
        if is_main_process():
            print(f"Resuming distributed training from {checkpoint_path}")
        
        # Load checkpoint
        underlying_model = self.model.module if hasattr(self.model, 'module') else self.model
        checkpoint = load_checkpoint(checkpoint_path, underlying_model, self.optimizer)
        
        # Restore training state
        self.epoch = checkpoint['epoch']
        self.global_step = self.epoch * len(train_loader)
        
        # Load metrics if available (only on main process)
        if is_main_process() and 'metrics' in checkpoint:
            self.metrics_tracker.metrics = checkpoint['metrics']
            self.best_val_wer = self.metrics_tracker.get_best('val_wer', mode='min')
        
        if is_main_process():
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
        if is_main_process():
            print("Evaluating on test set...")
        
        test_loss, test_wer, test_cer = self.validate(test_loader, vocab)
        
        results = {
            'test_loss': test_loss,
            'test_wer': test_wer,
            'test_cer': test_cer
        }
        
        if is_main_process():
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