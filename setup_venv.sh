#!/bin/bash

# Setup script for creating virtual environment and installing dependencies
# Run this on the cluster before submitting SLURM jobs

# Navigate to project directory
cd ~/asr

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt

# Verify installation
python -c "
import sys
sys.path.append('src')
from lmu import LMU, LMUFFT
"

python pytorch-test-setup.py