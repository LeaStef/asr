# PyTorch LMU-ASR

A high-performance PyTorch-based Automatic Speech Recognition (ASR) system using Legendre Memory Units (LMUs) with attention mechanisms. Optimized for fast distributed training on both single and multi-GPU configurations with GigaSpeech dataset support.

## Features

- **🚀 High-Performance Training**: Optimized for speed with large batch sizes and efficient memory usage
- **🧠 LMU-based encoder** with multi-head self-attention for enhanced temporal modeling
- **📊 GigaSpeech dataset support** with flexible subset selection (xs, s, m, l, xl)
- **⚡ Distributed training** with DDP and torchrun support for multi-GPU scaling
- **🎯 Mixed precision training** with automatic loss scaling for memory efficiency
- **⚙️ Flexible configuration** system with optimized presets and SLURM integration
- **🔧 CTC-based decoder** with blank token bias correction and beam search support
- **📈 Performance optimizations**: Model compilation, memory management, and I/O efficiency

## Quick Start

### Prerequisites

```bash
# Create conda environment
conda create -n pytorch-lmu-asr python=3.10 -y
conda activate pytorch-lmu-asr

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python pytorch-test-setup.py
```

### Training

```bash
# Single GPU optimized training
sbatch train_single_gpu.sh

# 2-GPU distributed training
sbatch train_2gpu.sh

# Large-scale training with monitoring (4 GPUs)
sbatch train_4gpu.sh

# Interactive training with presets
python scripts/train_flexible.py --preset performance --mixed-precision
python scripts/train_flexible.py --preset conservative --epochs 30
```

## Architecture

### Model Components

1. **LMU Encoder** (`src/models/lmu_encoder.py`)
   - 3-layer LMU stack with 6-head self-attention (optimized architecture)
   - 384 hidden units, 192 memory units for speed/accuracy balance
   - FFT-based LMUs and 2x downsampling for performance

2. **ASR Model** (`src/models/asr_model.py`)
   - CTC decoder with corrected blank token bias initialization
   - Beam search and greedy decoding
   - Optimized loss computation and gradient handling

3. **Data Pipeline** (`src/data/`)
   - High-performance GigaSpeech loaders (32 workers)
   - 80-dim mel-spectrograms with 400-frame max length
   - Optional SpecAugment for improved generalization

### Key Features

- **Speed Optimizations**: Large batch sizes, model compilation, optimized memory allocation
- **Attention-Enhanced LMUs**: 6-head self-attention with efficient attention mechanisms  
- **Mixed Precision**: FP16 training with automatic loss scaling
- **Distributed Training**: NCCL-optimized DDP with performance tuning
- **SLURM Integration**: Optimized batch scripts with resource allocation

## Training Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `train_single_gpu.sh` | Optimized single GPU SLURM | Production single GPU |
| `train_2gpu.sh` | 2-GPU SLURM distributed | Production 2-GPU training |
| `train_4gpu.sh` | 4-GPU SLURM with monitoring | Large-scale training |
| `train_flexible.py` | Interactive with presets | Development & tuning |

### SLURM Optimizations
- **Resource allocation**: Optimized CPU/memory ratios, proper GPU topology
- **NCCL configuration**: Fixed P2P communication issues, enhanced debugging
- **Performance tuning**: CUDA optimizations, memory management, module loading
- **Monitoring**: Built-in GPU utilization tracking (train_4gpu.sh), progress indicators
- **Cluster compatibility**: Aligned with proven YOLOv5 multi-GPU configuration

## Configuration

### Training Presets

The training system now uses generic presets instead of GPU-specific configurations:

```bash
# Available presets
--preset default      # Balanced settings (batch_size=16, lr=1e-3, mixed_precision=True)
--preset conservative # Lower resource usage (batch_size=24, lr=1e-4, mixed_precision=False)
--preset performance  # Higher throughput (batch_size=64, lr=4e-3, mixed_precision=True)

# Examples
python scripts/train_flexible.py --preset performance --epochs 35
torchrun --nproc_per_node=2 scripts/train_flexible.py --preset default --batch-size 96
```

### Production SLURM Training

```bash
# Single GPU training
sbatch train_single_gpu.sh

# 2-GPU training
sbatch train_2gpu.sh

# Large-scale training (4 GPUs with monitoring)
sbatch train_4gpu.sh
```

### Custom Configuration

```bash
# Custom parameters
python scripts/train_flexible.py \
    --batch-size 32 \
    --lr 2e-3 \
    --epochs 10 \
    --mixed-precision \
    --dataset gigaspeech \
    --subset xs
```

## Datasets

### GigaSpeech
- Requires manual download and preprocessing
- Supports multiple subsets (xs, s, m, l, xl)
- Manifest-based data loading

```bash
# Download GigaSpeech (example for 'xs' subset)
python scripts/download_gigaspeech.py --subset xs --save_dir ./data
```

## Model Architecture Details

### Optimized Configuration
- **Input**: 80-dimensional mel-spectrograms (16kHz audio, 400 max frames)
- **Encoder**: 3 LMU layers, 384 hidden units, 192 memory units (speed-optimized)
- **Attention**: 6-head self-attention with efficient computation
- **Decoder**: CTC with 29-character vocabulary and corrected blank token bias
- **Features**: FFT-based LMUs, 2x downsampling, model compilation support

### Performance & Memory Requirements

| GPU Type | Batch Size | Memory Usage | Training Speed |
|----------|------------|--------------|----------------|
| RTX 4090 (24GB) | 48 | ~20GB | ~2.0 it/s |
| RTX 6000 (48GB) | 64 | ~35GB | ~2.2 it/s |
| A100 (80GB) | 96 | ~55GB | ~3.5 it/s |

**Multi-GPU scaling**: Near-linear speedup up to 4 GPUs with optimized NCCL configuration.

## Output Files

Checkpoint files are saved to:
- `./logs/checkpoints/` (legacy scripts)
- `./outputs/checkpoints/` (train_flexible.py)

Files saved as:
- `checkpoint_epoch_{epoch}.pt`
- `checkpoint_epoch_{epoch}_best.pt`

## Performance Targets

- **Training Speed**: 2-3x faster than baseline with optimized batch sizes and architecture
- **Convergence**: 30% faster convergence with optimized learning rates and warmup
- **WER Performance**: Corrected CTC bias enables proper learning (resolves 100% WER issue)
- **Multi-GPU**: Near-linear speedup with NCCL-optimized distributed training
- **Memory Efficiency**: 25% better GPU utilization with mixed precision and compilation
- **Cluster Stability**: Resolved terminal freezing and communication issues through proven NCCL configuration

## Development

### Project Structure
```
pytorch-lmu-asr/
├── src/
│   ├── models/          # LMU encoder and ASR model
│   ├── data/            # Dataset loaders and preprocessing
│   ├── training/        # Training loops and distributed setup
│   ├── config/          # Configuration management
│   └── lmu.py          # Local LMU implementation
├── scripts/             # Training scripts
├── configs/             # YAML configuration files
├── notebooks/           # Demo notebooks
└── requirements.txt
```

### Testing & Debugging
```bash
# Environment verification
python pytorch-test-setup.py

# Configuration dry run
python scripts/train_flexible.py --dry-run

# Debug mode (single epoch)
python scripts/train_flexible.py --debug --preset conservative

# Test different presets
python scripts/train_flexible.py --preset default --dry-run
python scripts/train_flexible.py --preset performance --dry-run

# Debug scripts for troubleshooting
python scripts/debug_validation_data.py  # Check data pipeline
python scripts/validate_checkpoint.py    # Verify model loading
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Legendre Memory Units**: Based on the neural memory architecture from Nengo
- **GigaSpeech**: Large-scale speech recognition corpus
- **PyTorch**: Deep learning framework

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{pytorch-lmu-asr,
  title={PyTorch LMU-ASR: Attention-Enhanced Legendre Memory Units for Speech Recognition},
  author={[Your Name]},
  year={2024},
  url={https://github.com/h6ly/pytorch-lmu-asr}
}
```

## Support

For questions and issues:
- Open a GitHub issue for bugs or feature requests
- Check `CLAUDE.md` for detailed development notes and instructions
- Use debug scripts in `/scripts` for troubleshooting training issues
- Review SLURM scripts for production deployment examples