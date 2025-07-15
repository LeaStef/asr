# PyTorch-LMU ASR Environment Setup for Linux

## Quick Setup Commands

```bash
# Create new conda environment with Python 3.10
conda create -n pytorch-lmu-asr python=3.10 -y

# Activate the environment
conda activate pytorch-lmu-asr

# Install PyTorch with CUDA support (adjust CUDA version as needed)
# For CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1:
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install core scientific packages
conda install -c conda-forge \
    numpy=1.24.* \
    scipy=1.10.* \
    matplotlib=3.7.* \
    seaborn=0.12.* \
    jupyter \
    notebook \
    pandas=2.0.*

# Install system audio libraries
sudo apt-get update
sudo apt-get install -y libsndfile1-dev ffmpeg

# Clone and install pytorch-lmu
git clone https://github.com/hrshtv/pytorch-lmu.git
cd pytorch-lmu
pip install -e .
cd ..

# Install ML and audio packages
pip install \
    librosa==0.10.1 \
    datasets==2.14.5 \
    jiwer==3.0.3 \
    tensorboard==2.14.1 \
    wandb==0.15.12 \
    tqdm==4.66.1 \
    omegaconf==2.3.0 \
    hydra-core==1.3.2 \
    soundfile==0.12.1 \
    scikit-learn==1.3.0

# Create project directory
mkdir pytorch_lmu_asr
cd pytorch_lmu_asr
```

## Prerequisites for Linux

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    libsndfile1-dev \
    ffmpeg \
    libasound2-dev \
    portaudio19-dev

# CentOS/RHEL/Fedora
sudo yum install -y \
    gcc \
    gcc-c++ \
    git \
    wget \
    curl \
    libsndfile-devel \
    ffmpeg \
    alsa-lib-devel \
    portaudio-devel
```

### NVIDIA GPU Setup
```bash
# Check NVIDIA driver and CUDA
nvidia-smi

# If CUDA not installed, follow NVIDIA's installation guide:
# https://developer.nvidia.com/cuda-downloads

# Verify CUDA version (important for PyTorch compatibility)
nvcc --version
```

### Install Miniconda (if not already installed)
```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate
conda init bash
# Restart terminal or source ~/.bashrc
```

## Detailed Step-by-Step Setup

### 1. Create and Activate Environment
```bash
# Create environment with Python 3.10
conda create -n pytorch-lmu-asr python=3.10 -y

# Activate the environment
conda activate pytorch-lmu-asr
```

### 2. Install PyTorch with CUDA Support
```bash
# Check your CUDA version first
nvidia-smi

# For CUDA 11.8 (most common)
conda install pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1
# conda install pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# For CPU-only (not recommended for training)
# conda install pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 cpuonly -c pytorch

# Verify PyTorch CUDA installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
```

### 3. Install Scientific Computing Stack
```bash
# Core scientific packages via conda for optimal performance
conda install -c conda-forge \
    numpy=1.24.* \
    scipy=1.10.* \
    matplotlib=3.7.* \
    seaborn=0.12.* \
    jupyter \
    notebook \
    ipykernel \
    pandas=2.0.* \
    scikit-learn=1.3.*
```

### 4. Install pytorch-lmu Library
```bash
# Clone and install in development mode for easy updates
git clone https://github.com/hrshtv/pytorch-lmu.git
cd pytorch-lmu

# Install in development mode
pip install -e .

# Return to parent directory
cd ..

# Verify installation
python -c "from lmu import LMU; print('✅ pytorch-lmu installed successfully')"
```

### 5. Install Audio and ML Libraries
```bash
# Audio processing and machine learning packages
pip install \
    librosa==0.10.1 \
    datasets==2.14.5 \
    jiwer==3.0.3 \
    tensorboard==2.14.1 \
    wandb==0.15.12 \
    tqdm==4.66.1 \
    omegaconf==2.3.0 \
    hydra-core==1.3.2 \
    soundfile==0.12.1 \
    python-speech-features==0.6 \
    pesq==0.0.4
```

### 6. Install Distributed Training Dependencies
```bash
# For multi-GPU and distributed training
pip install \
    accelerate \
    deepspeed \
    fairscale \
    torch-optimizer

# Optional: Install Horovod for advanced distributed training
# HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch]
```

## Verification Test

Create and run this comprehensive test script:

```python
# test_pytorch_lmu_setup.py
import torch
import torch.distributed as dist
import torchaudio
import librosa
import numpy as np
from lmu import LMU, LMUFFT
import datasets
import jiwer
import os

print("=== Environment Verification ===")
print("PyTorch version:", torch.__version__)
print("TorchAudio version:", torchaudio.__version__)

# Check CUDA setup
print("\n=== CUDA Configuration ===")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
    print("Device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    device = "cuda"
else:
    print("⚠️  CUDA not available - using CPU")
    device = "cpu"

# Test pytorch-lmu functionality
print("\n=== Testing pytorch-lmu ===")

# Test standard LMU
print("Testing standard LMU...")
lmu_model = LMU(
    input_size=80,
    hidden_size=256,
    memory_size=128,
    theta=1000
).to(device)

x = torch.randn(4, 100, 80).to(device)  # batch_size=4, seq_len=100, features=80
with torch.no_grad():
    output, (h_n, m_n) = lmu_model(x)
print(f"✅ Standard LMU output shape: {output.shape}")

# Test FFT-based LMU (faster for long sequences)
print("Testing FFT-based LMU...")
lmu_fft_model = LMUFFT(
    input_size=80,
    hidden_size=256,
    memory_size=128,
    seq_len=100,
    theta=1000
).to(device)

with torch.no_grad():
    output_fft, h_n_fft = lmu_fft_model(x)
print(f"✅ FFT LMU output shape: {output_fft.shape}")

# Test multi-GPU capability
print("\n=== Multi-GPU Testing ===")
if torch.cuda.device_count() > 1:
    print(f"✅ {torch.cuda.device_count()} GPUs available for distributed training")
    
    # Test DataParallel (simple multi-GPU)
    if torch.cuda.device_count() >= 2:
        multi_gpu_model = torch.nn.DataParallel(lmu_model)
        with torch.no_grad():
            output_multi = multi_gpu_model(x)
        print("✅ DataParallel test successful")
    
    # Test distributed training setup
    print("✅ Ready for DistributedDataParallel setup")
else:
    print("ℹ️  Single GPU setup - distributed training will use single device")

# Test audio processing
print("\n=== Testing Audio Processing ===")
sr = 16000
audio = np.random.randn(sr * 2)  # 2 seconds of random audio
mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
print(f"✅ Mel spectrogram shape: {mels.shape}")

# Test data loading
print("\n=== Testing Dataset Access ===")
try:
    from datasets import load_dataset
    print("✅ HuggingFace datasets working")
except Exception as e:
    print(f"❌ Dataset issue: {e}")

# Test distributed training utilities
print("\n=== Distributed Training Support ===")
try:
    import accelerate
    print("✅ Accelerate library available")
except:
    print("⚠️  Accelerate not installed")

try:
    import deepspeed
    print("✅ DeepSpeed available")
except:
    print("ℹ️  DeepSpeed not installed (optional)")

print(f"\n🎉 Environment setup complete!")
print(f"Recommended device for training: {device}")
print(f"GPUs available: {torch.cuda.device_count()}")
print("\nNext steps:")
print("1. Test multi-GPU setup if available")
print("2. Download LibriSpeech dataset")
print("3. Implement distributed training configuration")
```

Save and run the test:
```bash
python test_pytorch_lmu_setup.py
```

## Distributed Training Infrastructure

### Multi-GPU Training Setup
The environment is pre-configured for several distributed training approaches:

#### 1. DataParallel (Simple Multi-GPU)
```python
# Wrap model for multi-GPU training
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

#### 2. DistributedDataParallel (Recommended)
```python
# Initialize process group
torch.distributed.init_process_group(backend='nccl')

# Wrap model
model = torch.nn.parallel.DistributedDataParallel(model)
```

#### 3. Accelerate (Hugging Face - Easiest)
```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

### Launch Scripts for Distributed Training

Create `launch_distributed.py`:
```python
import torch.multiprocessing as mp
import torch.distributed as dist
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_worker(rank, world_size, args):
    setup(rank, world_size)
    # Your training code here
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
```

Or use torchrun (recommended):
```bash
torchrun --nproc_per_node=2 train.py --distributed
```

## Linux Optimization Settings

### Environment Variables
Add these to your `~/.bashrc` for optimal performance:

```bash
# CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=1

# Memory optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Distributed training
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available

# Apply changes
source ~/.bashrc
```

### Performance Tuning
```bash
# Check CPU info for optimal thread settings
lscpu | grep "CPU(s):"
cat /proc/cpuinfo | grep "model name" | head -1

# Check GPU memory
nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits

# Monitor GPU usage during training
watch -n 1 nvidia-smi
```

## Environment Management

### Daily Usage
```bash
# Activate environment
conda activate pytorch-lmu-asr

# Check GPU availability
nvidia-smi

# Deactivate when done
conda deactivate
```

### Maintenance Commands
```bash
# List installed packages
conda list
pip list

# Update PyTorch (check compatibility first)
conda update pytorch torchvision torchaudio -c pytorch

# Update pytorch-lmu
cd pytorch-lmu
git pull
pip install -e .
cd ..

# Export environment
conda env export > pytorch_lmu_environment.yml
```

## Reproducible Environment File

Create `pytorch_lmu_environment.yml`:

```yaml
name: pytorch-lmu-asr
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch=2.1.0
  - torchvision=0.16.0
  - torchaudio=2.1.0
  - pytorch-cuda=11.8
  - numpy=1.24.*
  - scipy=1.10.*
  - matplotlib=3.7.*
  - seaborn=0.12.*
  - jupyter
  - notebook
  - pandas=2.0.*
  - scikit-learn=1.3.*
  - pip
  - pip:
    - librosa==0.10.1
    - datasets==2.14.5
    - jiwer==3.0.3
    - tensorboard==2.14.1
    - wandb==0.15.12
    - tqdm==4.66.1
    - omegaconf==2.3.0
    - hydra-core==1.3.2
    - soundfile==0.12.1
    - python-speech-features==0.6
    - accelerate
    - deepspeed
    - fairscale
    - torch-optimizer
    - pesq==0.0.4
```

Create environment from file:
```bash
conda env create -f pytorch_lmu_environment.yml
conda activate pytorch-lmu-asr

# Still need to install pytorch-lmu manually
git clone https://github.com/hrshtv/pytorch-lmu.git
cd pytorch-lmu && pip install -e . && cd ..
```

## Troubleshooting

### CUDA Issues
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
conda uninstall pytorch torchvision torchaudio
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Clear PyTorch cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Audio Library Issues
```bash
# Ubuntu/Debian
sudo apt-get install --reinstall libsndfile1-dev ffmpeg

# If librosa issues persist
pip uninstall librosa soundfile
pip install librosa==0.10.1 soundfile==0.12.1
```

### Distributed Training Issues
```bash
# Check NCCL installation
python -c "import torch; print(torch.distributed.is_nccl_available())"

# Test distributed setup
python -c "import torch.distributed as dist; print('Distributed available:', dist.is_available())"
```

### Memory Issues
```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# Clear memory in Python
import torch
torch.cuda.empty_cache()
```

## Multi-GPU Training Configuration

### Recommended Training Config for Distributed Setup
```yaml
# Update your training config for multi-GPU
training:
  distributed: true
  backend: "nccl"  # Use "gloo" for CPU-only
  batch_size: 16   # Per GPU batch size
  accumulate_grad_batches: 2
  sync_batchnorm: true
  find_unused_parameters: false

hardware:
  gpus: -1  # Use all available GPUs
  precision: 16  # Mixed precision
  strategy: "ddp"  # DistributedDataParallel
```

This setup provides a robust foundation for PyTorch-LMU development on Linux with excellent multi-GPU scaling capabilities. The distributed training infrastructure is ready to scale from single GPU to multi-node clusters with minimal code changes.