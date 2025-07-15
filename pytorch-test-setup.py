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