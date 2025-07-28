#!/bin/bash

# Setup script for creating virtual environment and installing dependencies
# Run this on the cluster before submitting SLURM jobs

echo "=============================="
echo "Setting up PyTorch LMU-ASR environment"
echo "=============================="

# Navigate to project directory
cd ~/asr

echo "Working directory: $(pwd)"

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

echo "Virtual environment activated: $VIRTUAL_ENV"
echo "Python version: $(python --version)"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support first
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo "Installing project dependencies..."
pip install -r requirements.txt

# Verify installation
echo "=============================="
echo "Verifying installation..."
echo "=============================="

echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Test import of local modules
echo "Testing local module imports..."
python -c "
import sys
sys.path.append('src')
from lmu import LMU, LMUFFT
print('✅ LMU modules imported successfully')
"

python pytorch-test-setup.py

echo "=============================="
echo "Setup completed successfully!"
echo "=============================="
echo "To activate this environment in future sessions:"
echo "source ~/asr/venv/bin/activate"
echo ""
echo "You can now submit SLURM jobs using:"
echo "sbatch train_single.sh"
echo "sbatch train_array.sh" 
echo "sbatch train_multi_gpu.sh"