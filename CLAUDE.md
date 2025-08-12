# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based Automatic Speech Recognition (ASR) system using Legendre Memory Units (LMUs). The system is designed for distributed training on Linux with GigaSpeech dataset, supporting both single and multi-GPU configurations.

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

# Download GigaSpeech dataset (optional - HuggingFace datasets used by default)
python scripts/download_gigaspeech.py --subset xs --save_dir ./data

# Run the demo notebook for complete walkthrough
jupyter notebook notebooks/lmu_asr_demo.ipynb
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
1. **LMU Encoder** (`src/models/lmu_encoder.py`) - Stack of LMU layers with multi-head self-attention for enhanced temporal modeling
2. **ASR Model** (`src/models/asr_model.py`) - Complete ASR system with CTC decoder, beam search, and loss computation
3. **Data Pipeline** (`src/data/`) - GigaSpeech dataset processing, mel-spectrogram conversion, and SpecAugment
4. **Training Logic** (`src/training/`) - Single GPU and distributed training with mixed precision and early stopping
5. **Demo Notebook** (`notebooks/lmu_asr_demo.ipynb`) - Complete end-to-end workflow demonstration with attention analysis

### Configuration System
- **Dataclass-based config** in `src/config/config.py` with OmegaConf integration
- **Distributed setup** in `src/config/distributed_config.py` handles multi-GPU coordination
- **Hydra integration** for configuration management and command-line overrides

### Key Architecture Decisions
- **Features**: 80-dimensional mel-spectrograms (16kHz audio, 25ms window, 10ms hop)
- **Model**: Attention-enhanced LMU encoder (4 layers, 512 hidden units, 256 memory units) + CTC decoder
- **Attention**: Multi-head self-attention (8 heads) before each LMU layer with proper masking
- **Vocabulary**: 29 characters (26 letters + space + apostrophe + CTC blank)
- **Training**: Mixed precision, gradient clipping, distributed data parallel

## Data Pipeline
- **Dataset**: GigaSpeech subset via HuggingFace datasets
- **Preprocessing**: Audio → mel-spectrogram → character tokenization
- **Augmentation**: SpecAugment-style data augmentation
- **Distributed**: DistributedSampler for multi-GPU training

## Distributed Training Strategy
- **Primary**: DistributedDataParallel (DDP) with NCCL backend
- **Alternative**: Accelerate library for simplified distributed setup
- **Scaling**: Linear learning rate scaling with number of GPUs
- **Synchronization**: Synchronized batch normalization across GPUs

## Current Implementation Status
The repository contains a complete, fully functional LMU-based ASR system:
- ✅ Configuration system, training scripts, project structure
- ✅ LMU encoder and ASR model implementations
- ✅ Data loading and preprocessing pipeline
- ✅ Training loop and distributed training logic
- ✅ Evaluation and checkpointing utilities
- ✅ Complete demo notebook with end-to-end workflow

## Development Guidelines
- **LMU Variants**: Use standard LMU for variable sequences, LMUFFT for fixed-length sequences
- **Memory Management**: Implement gradient checkpointing for large models
- **Error Handling**: Include proper distributed training error handling and cleanup
- **Testing**: Always test single GPU setup before distributed training
- **Checkpointing**: Save only from rank 0 process in distributed training

## Expected Performance Targets
- **Single GPU**: Competitive WER on GigaSpeech test sets
- **Multi-GPU**: Same accuracy with linear speedup up to 4-8 GPUs
- **Scaling**: Efficient scaling for production workloads
- **Demo Model**: Reduced configuration for fast experimentation and learning

## Quick Start
For immediate hands-on experience, run the demo notebook:
```bash
cd notebooks
jupyter notebook lmu_asr_demo.ipynb
```

This notebook provides a complete walkthrough from setup to inference with a smaller model configuration suitable for learning and experimentation.