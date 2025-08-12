# PyTorch LMU-ASR

A PyTorch-based Automatic Speech Recognition (ASR) system using Legendre Memory Units (LMUs) with attention mechanisms. Designed for distributed training on both single and multi-GPU configurations.

## Features

- **LMU-based encoder** with multi-head self-attention for enhanced temporal modeling
- **GigaSpeech dataset support** with flexible subset selection
- **Distributed training** with DDP and torchrun support
- **Mixed precision training** for memory efficiency
- **Flexible configuration** system with Hydra and argparse
- **GPU-optimized presets** for L40S and A100 GPUs
- **CTC-based decoder** with beam search support

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

### Basic Training

```bash
# Single GPU training
python scripts/train.py

# Multi-GPU training with torchrun
torchrun --nproc_per_node=2 scripts/train_torchrun.py

# Flexible training with custom batch size
python scripts/train_flexible.py --batch-size 64 --preset l40s-1gpu
```

## Architecture

### Model Components

1. **LMU Encoder** (`src/models/lmu_encoder.py`)
   - Stack of LMU layers with multi-head self-attention
   - Configurable memory size and temporal modeling
   - Support for both standard and FFT-based LMUs

2. **ASR Model** (`src/models/asr_model.py`)
   - Complete ASR system with CTC decoder
   - Beam search inference
   - Loss computation and metrics

3. **Data Pipeline** (`src/data/`)
   - GigaSpeech dataset loaders
   - Mel-spectrogram feature extraction
   - SpecAugment data augmentation

### Key Features

- **Attention-Enhanced LMUs**: Multi-head self-attention before each LMU layer
- **Mixed Precision**: FP16/FP32 training for memory optimization
- **Distributed Training**: DDP with NCCL backend
- **Flexible Configuration**: Hydra + argparse support

## Training Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `train.py` | Single GPU with Hydra | Basic development |
| `train_torchrun.py` | Modern distributed training | Production multi-GPU |
| `train_flexible.py` | Argparse with presets | Experimentation |
| `train_distributed.py` | Legacy multiprocessing | Compatibility |

## Configuration

### GPU Presets

```bash
# L40S GPU (48GB)
python scripts/train_flexible.py --preset l40s-1gpu  # batch_size=64
torchrun --nproc_per_node=2 scripts/train_flexible.py --preset l40s-2gpu  # batch_size=128

# A100 GPU (80GB)
python scripts/train_flexible.py --preset a100-1gpu  # batch_size=48
torchrun --nproc_per_node=2 scripts/train_flexible.py --preset a100-2gpu  # batch_size=96
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

### Default Configuration
- **Input**: 80-dimensional mel-spectrograms (16kHz audio)
- **Encoder**: 4 LMU layers, 512 hidden units, 256 memory units
- **Attention**: 8-head self-attention with proper masking
- **Decoder**: CTC with 32-character vocabulary
- **Window**: 25ms window, 10ms hop length

### Memory Requirements

| GPU Type | Recommended Batch Size | Memory Usage |
|----------|----------------------|--------------|
| RTX 4090 (24GB) | 32-48 | ~18GB |
| L40S (48GB) | 64-128 | ~35GB |
| A100 (80GB) | 96-192 | ~60GB |

## Output Files

Checkpoint files are saved to:
- `./logs/checkpoints/` (legacy scripts)
- `./outputs/checkpoints/` (train_flexible.py)

Files saved as:
- `checkpoint_epoch_{epoch}.pt`
- `checkpoint_epoch_{epoch}_best.pt`

## Performance Targets

- **Single GPU**: Competitive WER on GigaSpeech test sets
- **Multi-GPU**: Linear speedup up to 4-8 GPUs
- **GigaSpeech**: Competitive results on various subsets

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

### Testing
```bash
# Environment verification
python pytorch-test-setup.py

# Configuration dry run
python scripts/train_flexible.py --dry-run

# Debug mode (single epoch)
python scripts/train_flexible.py --debug
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
- Open a GitHub issue
- Check the demo notebook for examples
- Review CLAUDE.md for detailed development notes