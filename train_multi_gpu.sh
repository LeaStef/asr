#!/bin/bash

#SBATCH --job-name=lmu-asr-multi-gpu
# Set resource requirements: Queues are limited to seven day allocations
# Time format: HH:MM:SS
#SBATCH --time=150:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --partition=CELIASMI

# Email notifications (update with your watid)
#SBATCH --mail-user=h6ly@uwaterloo.ca
#SBATCH --mail-type=ALL

# Output files
#SBATCH -o JOB%j.out
#SBATCH -e JOB%j-err.out

echo "==== SLURM Multi-GPU Job Information ===="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "=============================="

# Set up environment
log_dir=$HOME/asr
mkdir -p $log_dir
cd $log_dir

echo "Working directory: $(pwd)"
echo "Activating virtual environment..."

# Load up your virtual environment  
# Set up environment on watgpu.cs or in interactive session (use `source` keyword)
# Assuming venv is in asr/venv/ or asr/bin/activate
if [ -f "$log_dir/venv/bin/activate" ]; then
    source $log_dir/venv/bin/activate
elif [ -f "$log_dir/bin/activate" ]; then
    source $log_dir/bin/activate
else
    echo "Warning: Virtual environment not found. Please check the path."
fi

echo "Virtual environment activated: $VIRTUAL_ENV"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Already in project directory ($HOME/asr)
echo "Project directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# NCCL optimization for RTX 6000 GPUs (no NVLink)
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_TREE_THRESHOLD=0
export OMP_NUM_THREADS=1

echo "=============================="
echo "NCCL Configuration for RTX 6000:"
echo "  NCCL_DEBUG: $NCCL_DEBUG"
echo "  NCCL_TIMEOUT: $NCCL_TIMEOUT"
echo "  NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
echo "  NCCL_P2P_DISABLE: $NCCL_P2P_DISABLE"
echo "=============================="
echo "Starting distributed training..."
echo "=============================="

# Multi-GPU training with torchrun
# Estimated training time: ~3 days for GigaSpeech 'm' subset with RTX 6000
echo "  Checkpoints saved every epoch to ./outputs/checkpoints/"

torchrun --nproc_per_node=2 scripts/train_flexible.py \
    --preset rtx6000-2gpu \
    --output-dir ./outputs \
    --dataset gigaspeech \
    --subset xs

# For larger dataset training, use 's' subset instead:
# torchrun --nproc_per_node=2 scripts/train_flexible.py \
#     --preset rtx6000-2gpu \
#     --output-dir ./outputs \
#     --dataset gigaspeech \
#     --subset s

# Alternative: Use the torchrun-specific script
# torchrun --nproc_per_node=2 scripts/train_torchrun.py

echo "=============================="
echo "Training completed at: $(date)"
echo "=============================="