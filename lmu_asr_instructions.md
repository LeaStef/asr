# PyTorch LMU-based ASR System Setup Instructions

## Project Overview
Create a basic Automatic Speech Recognition (ASR) system using Legendre Memory Units (LMUs) with PyTorch. The system should be trainable on Linux with single or multi-GPU support using the GigaSpeech dataset, leveraging the pytorch-lmu implementation.

## Project Structure
Create the following directory structure:
```
pytorch_lmu_asr/
├── requirements.txt
├── environment.yml
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lmu_encoder.py
│   │   └── asr_model.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── distributed.py
│   │   └── utils.py
│   └── config/
│       ├── __init__.py
│       ├── config.py
│       └── distributed_config.py
├── scripts/
│   ├── download_data.py
│   ├── train.py
│   ├── train_distributed.py
│   └── evaluate.py
├── configs/
│   ├── base_config.yaml
│   ├── single_gpu.yaml
│   ├── multi_gpu.yaml
│   └── distributed.yaml
└── notebooks/
    └── explore_data.ipynb
```

## Dependencies (requirements.txt)
```
torch>=2.1.0
torchaudio>=2.1.0
torchvision>=0.16.0
librosa>=0.10.0
datasets>=2.14.0
numpy>=1.24.0
scipy>=1.10.0
jiwer>=3.0.0
tensorboard>=2.13.0
wandb>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
omegaconf>=2.3.0
hydra-core>=1.3.0
soundfile>=0.12.1
pandas>=2.0.0
scikit-learn>=1.3.0
accelerate>=0.24.0
deepspeed>=0.10.0
fairscale>=0.4.0
torch-optimizer>=0.3.0
pesq>=0.0.4
python-speech-features>=0.6
```

## Key Implementation Requirements

### 1. Install pytorch-lmu Library
Since pytorch-lmu isn't on PyPI, clone and install it:
```bash
git clone https://github.com/hrshtv/pytorch-lmu.git
cd pytorch-lmu
pip install -e .
cd ..
```

### 2. LMU Encoder Model (src/models/lmu_encoder.py)
Create an LMU-based encoder that:
- Takes mel-spectrogram features as input (80-dim)
- Uses LMU layers for temporal modeling
- Supports both standard LMU and FFT-based LMU for efficiency
- Includes downsampling for memory efficiency
- Has configurable memory size and hidden dimensions
- Supports distributed training

Key specifications:
- Input: Mel spectrograms (batch_size, seq_len, 80)
- LMU memory size: 256-512 (configurable)
- Hidden size: 512-1024 (configurable)
- Include dropout and layer norm
- Support for gradient checkpointing
- Output: Encoded features for CTC decoder

Example LMU usage:
```python
from lmu import LMU, LMUFFT

# Standard LMU (flexible sequence length)
lmu_layer = LMU(
    input_size=80,
    hidden_size=512,
    memory_size=256,
    theta=1000
)

# FFT-based LMU (faster for fixed sequence length)
lmu_fft_layer = LMUFFT(
    input_size=80,
    hidden_size=512,
    memory_size=256,
    seq_len=1000,  # Fixed sequence length
    theta=1000
)
```

### 3. Complete ASR Model (src/models/asr_model.py)
Build a full ASR model that combines:
- Mel-spectrogram preprocessing
- LMU encoder stack
- CTC decoder (linear layer to vocabulary)
- CTC loss computation
- Beam search decoding for inference
- Support for DistributedDataParallel

Model should support:
- Character-level tokenization initially
- Configurable vocabulary
- Mixed precision training (torch.cuda.amp)
- Gradient checkpointing for memory efficiency
- Multi-GPU training with minimal code changes

Example model structure:
```python
import torch
import torch.nn as nn
from lmu import LMU

class LMUASRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # LMU encoder stack
        self.lmu_layers = nn.ModuleList([
            LMU(
                input_size=config.input_size if i == 0 else config.hidden_size,
                hidden_size=config.hidden_size,
                memory_size=config.memory_size,
                theta=config.theta
            ) for i in range(config.num_lmu_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_size) 
            for _ in range(config.num_lmu_layers)
        ])
        
        self.dropout = nn.Dropout(config.dropout)
        
        # CTC output layer
        self.ctc_output = nn.Linear(config.hidden_size, config.vocab_size)
    
    def forward(self, x, lengths=None):
        # x: (batch_size, seq_len, features)
        memory_states = []
        
        for lmu, norm in zip(self.lmu_layers, self.layer_norms):
            x, (h, m) = lmu(x)
            x = norm(x)
            x = self.dropout(x)
            memory_states.append(m)
        
        # CTC logits
        logits = self.ctc_output(x)
        return logits, memory_states
```

### 4. Data Pipeline (src/data/)
Implement distributed-aware data loading for GigaSpeech:
- Download GigaSpeech subset
- Convert audio to mel spectrograms using librosa/torchaudio
- Character-level tokenization
- Efficient batching with padding and masking
- Data augmentation (SpecAugment-style)
- Support for DistributedSampler

Key preprocessing specs:
- Sample rate: 16kHz
- Mel features: 80 dimensions
- Window size: 25ms, hop: 10ms
- Normalization: per-utterance mean/std
- Sequence padding with collate function

Distributed data loading:
```python
import torch
from torch.utils.data import DataLoader, DistributedSampler

def create_dataloader(dataset, config, is_distributed=False):
    sampler = None
    if is_distributed:
        sampler = DistributedSampler(dataset)
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
```

### 5. Distributed Training Pipeline (src/training/)
Implement training logic with multiple distributed strategies:

#### trainer.py - Main Training Logic
- CTC loss optimization
- Learning rate scheduling (cosine annealing with warmup)
- Validation monitoring (WER calculation)
- Checkpointing and model saving
- Mixed precision training (torch.cuda.amp)
- Gradient accumulation and clipping

#### distributed.py - Distributed Training Setup
```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(rank, world_size, backend='nccl'):
    """Initialize distributed training"""
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

class DistributedTrainer:
    def __init__(self, model, config, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        
        # Wrap model for distributed training
        if world_size > 1:
            self.model = DDP(model, device_ids=[rank])
        else:
            self.model = model
    
    def train_epoch(self, dataloader, optimizer, scaler):
        # Training loop with distributed support
        pass
```

Training hyperparameters for multi-GPU:
- Batch size: 16-32 per GPU
- Learning rate: 1e-3 with linear scaling for multi-GPU
- Max epochs: 50
- Early stopping based on validation WER
- Gradient clipping: 1.0
- Mixed precision: True

### 6. Configuration Management (src/config/config.py)
Use Hydra/OmegaConf for configuration with distributed training options:

```yaml
# configs/base_config.yaml
model:
  encoder:
    input_size: 80
    hidden_size: 512
    memory_size: 256
    num_lmu_layers: 4
    theta: 1000
    dropout: 0.1
    use_fft_lmu: false  # Set to true for fixed sequence length
  decoder:
    vocab_size: 29  # 26 letters + space + apostrophe + blank
  
data:
  dataset: "gigaspeech"
  subset: "xs"
  sample_rate: 16000
  n_mels: 80
  max_seq_len: 1000
  augment: true
  num_workers: 4
  
training:
  batch_size: 16  # Per GPU
  lr: 1e-3
  max_epochs: 50
  patience: 10
  mixed_precision: true
  gradient_clip_norm: 1.0
  accumulate_grad_batches: 1
  
# configs/distributed.yaml
distributed:
  enabled: true
  backend: "nccl"
  find_unused_parameters: false
  sync_batchnorm: true
  
# configs/multi_gpu.yaml
defaults:
  - base_config
  - distributed

training:
  batch_size: 32  # Larger batch size for multi-GPU
  lr: 2e-3  # Scale learning rate
```

### 7. Distributed Training Scripts

#### scripts/train.py (Single GPU)
```python
import hydra
from omegaconf import DictConfig
import torch

@hydra.main(config_path="../configs", config_name="single_gpu")
def train(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model, data, training
    model = create_model(cfg).to(device)
    train_loader, val_loader = create_dataloaders(cfg)
    trainer = Trainer(model, cfg)
    
    # Train model
    trainer.fit(train_loader, val_loader)

if __name__ == "__main__":
    train()
```

#### scripts/train_distributed.py (Multi-GPU)
```python
import os
import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
import hydra
from omegaconf import DictConfig

def train_worker(rank, world_size, cfg):
    # Setup distributed training
    setup_distributed(rank, world_size)
    
    # Create model and wrap with DDP
    model = create_model(cfg).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    
    # Create distributed data loaders
    train_loader, val_loader = create_distributed_dataloaders(cfg, rank, world_size)
    
    # Train model
    trainer = DistributedTrainer(model, cfg, rank, world_size)
    trainer.fit(train_loader, val_loader)
    
    # Cleanup
    cleanup_distributed()

@hydra.main(config_path="../configs", config_name="multi_gpu")
def main(cfg: DictConfig):
    world_size = torch.cuda.device_count()
    mp.spawn(train_worker, args=(world_size, cfg), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
```

### 8. Launch Scripts for Easy Distributed Training

#### Using torchrun (Recommended)
```bash
# Single node, multiple GPUs
torchrun --nproc_per_node=4 scripts/train_distributed.py

# Multiple nodes (example)
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr="192.168.1.1" scripts/train_distributed.py
```

#### Using Accelerate (Easiest)
Create `scripts/train_accelerate.py`:
```python
from accelerate import Accelerator
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="base_config")
def train(cfg: DictConfig):
    # Initialize accelerator (handles distributed setup automatically)
    accelerator = Accelerator(mixed_precision='fp16' if cfg.training.mixed_precision else 'no')
    
    # Create model and data
    model = create_model(cfg)
    train_loader, val_loader = create_dataloaders(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)
    
    # Prepare for distributed training
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    # Training loop
    for epoch in range(cfg.training.max_epochs):
        train_epoch(model, train_loader, optimizer, accelerator)
        
        if accelerator.is_main_process:
            validate_and_save(model, val_loader, epoch)

if __name__ == "__main__":
    train()
```

Launch with:
```bash
accelerate launch scripts/train_accelerate.py
```

### 9. Evaluation Metrics and Monitoring
Implement distributed-aware evaluation:
- WER (Word Error Rate) calculation using jiwer
- Distributed validation (gather results across GPUs)
- TensorBoard logging with proper distributed handling
- Weights & Biases integration
- Model checkpointing with proper synchronization

### 10. Memory Optimization Strategies
Include these optimizations for efficient training:
- Gradient checkpointing for LMU layers
- Mixed precision training (torch.cuda.amp)
- Gradient accumulation for effective larger batch sizes
- Efficient data loading with proper num_workers
- Model sharding options for very large models

## Scripts to Create

### scripts/download_data.py
Download GigaSpeech and prepare data manifests with proper distributed access.

### scripts/train.py
Single GPU training script using Hydra for configuration management.

### scripts/train_distributed.py
Multi-GPU training script with DistributedDataParallel.

### scripts/train_accelerate.py
Training script using Hugging Face Accelerate (simplest distributed setup).

### scripts/evaluate.py
Evaluation script that loads trained model and computes WER on test set with distributed support.

## Development Notes

1. **LMU Variants**: 
   - Use standard LMU for variable sequence lengths
   - Use LMUFFT for fixed sequence lengths (faster)
   - Consider memory size vs accuracy trade-offs

2. **Distributed Strategy Selection**:
   - DataParallel: Simple, 2-4 GPUs, single machine
   - DistributedDataParallel: Recommended, scales well
   - Accelerate: Easiest to use, handles most complexity
   - DeepSpeed: For very large models

3. **Memory Management**: 
   - Use gradient checkpointing for large models
   - Monitor GPU memory usage across all devices
   - Implement gradient accumulation if needed

4. **Debugging**: 
   - Test single GPU setup first
   - Use NCCL_DEBUG=INFO for distributed debugging
   - Implement proper logging across ranks

5. **Checkpointing**: 
   - Save/load models with proper distributed handling
   - Only save from rank 0 to avoid conflicts
   - Include optimizer and scheduler states

## Expected Performance
Target WER on GigaSpeech test:
- Single GPU goal: <15% WER
- Multi-GPU goal: Same accuracy with faster training
- Scaling: Linear speedup up to 4-8 GPUs

## Implementation Priority
1. Basic LMU encoder + CTC model (single GPU)
2. Data pipeline for GigaSpeech
3. Training loop with CTC loss
4. Multi-GPU support with DistributedDataParallel
5. Evaluation with WER calculation
6. Hyperparameter tuning and optimization
7. Advanced distributed features (multi-node, DeepSpeed)

## Distributed Training Best Practices
- Always test single GPU setup first
- Use synchronous batch normalization for multi-GPU
- Scale learning rate with number of GPUs
- Monitor training across all ranks
- Use proper random seeding for reproducibility
- Implement graceful failure handling
- Profile communication vs computation time

## Testing Strategy
- Unit tests for model components
- Integration test with small dataset
- Multi-GPU functionality tests
- Memory profiling across devices
- Communication overhead analysis
- Gradient verification for distributed training

## Documentation
- Include docstrings with distributed training notes
- Add type hints throughout
- Create usage examples for different distributed setups
- Document scaling guidelines and performance expectations

Start with single GPU implementation, then gradually add distributed training capabilities. The infrastructure is designed to scale seamlessly from development to production workloads.