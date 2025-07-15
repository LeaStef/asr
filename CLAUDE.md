# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based Automatic Speech Recognition (ASR) system using Legendre Memory Units (LMUs). The system is designed for distributed training on Linux with LibriSpeech dataset, supporting both single and multi-GPU configurations.

## Essential Commands

### Environment Setup
```bash
# Create and activate conda environment
conda create -n pytorch-lmu-asr python=3.10 -y
conda activate pytorch-lmu-asr

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install project dependencies
pip install -r requirements.txt

# The pytorch-lmu library is located in ../pytorch-lmu/src/lmu.py
# Add it to Python path in scripts that need it:
# import sys
# sys.path.append('/Users/celiasmi/Documents/nengo/pytorch-lmu/src')
# from lmu import LMU, LMUFFT
```

### Verification and Testing
```bash
# Test environment setup
python pytorch-test-setup.py

# Download LibriSpeech dataset
python scripts/download_data.py --subset clean-100 --save_dir ./data
```

### Training Commands
```bash
# Single GPU training
python scripts/train.py

# Multi-GPU training with torch.multiprocessing
python scripts/train_distributed.py

# Multi-GPU training with torchrun (recommended)
torchrun --nproc_per_node=4 scripts/train_distributed.py

# Evaluation
python scripts/evaluate.py --checkpoint_path path/to/checkpoint.pt
```

### Configuration Management
All training configurations use Hydra with YAML files in `configs/`:
- `base_config.yaml` - Base configuration
- `single_gpu.yaml` - Single GPU setup
- `multi_gpu.yaml` - Multi-GPU setup
- `distributed.yaml` - Distributed training parameters

## Architecture Overview

### Core Components
1. **LMU Encoder** (`src/models/lmu_encoder.py`) - Stack of LMU layers for temporal modeling
2. **ASR Model** (`src/models/asr_model.py`) - Complete ASR system with CTC decoder
3. **Data Pipeline** (`src/data/`) - LibriSpeech dataset processing and mel-spectrogram conversion
4. **Training Logic** (`src/training/`) - Single GPU and distributed training implementations

### Configuration System
- **Dataclass-based config** in `src/config/config.py` with OmegaConf integration
- **Distributed setup** in `src/config/distributed_config.py` handles multi-GPU coordination
- **Hydra integration** for configuration management and command-line overrides

### Key Architecture Decisions
- **Features**: 80-dimensional mel-spectrograms (16kHz audio, 25ms window, 10ms hop)
- **Model**: LMU encoder (4 layers, 512 hidden units, 256 memory units) + CTC decoder
- **Vocabulary**: 29 characters (26 letters + space + apostrophe + CTC blank)
- **Training**: Mixed precision, gradient clipping, distributed data parallel

## Data Pipeline
- **Dataset**: LibriSpeech clean-100 subset via HuggingFace datasets
- **Preprocessing**: Audio → mel-spectrogram → character tokenization
- **Augmentation**: SpecAugment-style data augmentation
- **Distributed**: DistributedSampler for multi-GPU training

## Distributed Training Strategy
- **Primary**: DistributedDataParallel (DDP) with NCCL backend
- **Alternative**: Accelerate library for simplified distributed setup
- **Scaling**: Linear learning rate scaling with number of GPUs
- **Synchronization**: Synchronized batch normalization across GPUs

## Current Implementation Status
The repository contains a well-structured foundation but is missing core implementations:
- ✅ Configuration system, training scripts, project structure
- ❌ LMU encoder and ASR model implementations
- ❌ Data loading and preprocessing pipeline
- ❌ Training loop and distributed training logic
- ❌ Evaluation and checkpointing utilities

## Development Guidelines
- **LMU Variants**: Use standard LMU for variable sequences, LMUFFT for fixed-length sequences
- **Memory Management**: Implement gradient checkpointing for large models
- **Error Handling**: Include proper distributed training error handling and cleanup
- **Testing**: Always test single GPU setup before distributed training
- **Checkpointing**: Save only from rank 0 process in distributed training

## Expected Performance Targets
- **Single GPU**: <15% WER on LibriSpeech clean test
- **Multi-GPU**: Same accuracy with linear speedup up to 4-8 GPUs
- **Scaling**: Efficient scaling for production workloads