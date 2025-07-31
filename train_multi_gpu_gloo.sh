#!/bin/bash

#SBATCH --job-name=lmu-asr-multi-gpu-gloo
# Set resource requirements
#SBATCH --time=168:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --partition=CELIASMI

# Email notifications
#SBATCH --mail-user=h6ly@uwaterloo.ca
#SBATCH --mail-type=ALL

# Output files
#SBATCH -o JOB%j.out
#SBATCH -e JOB%j-err.out

echo "==== SLURM Multi-GPU Job Information (GLOO Backend) ===="
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

# Load virtual environment
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

# Create logs directory
mkdir -p logs

# Use Gloo backend instead of NCCL (more stable for RTX 6000)
export TORCH_DISTRIBUTED_BACKEND=gloo
export OMP_NUM_THREADS=8

echo "=============================="
echo "Distributed Configuration (GLOO Backend):"
echo "  Backend: GLOO (CPU-based, more stable)"
echo "  GPUs: 2x RTX 6000 Ada Generation"
echo "  No NCCL communication issues!"
echo "=============================="
echo "Starting distributed training with GLOO backend..."
echo "=============================="

# Multi-GPU training with Gloo backend
torchrun --nproc_per_node=2 scripts/train_flexible.py \
    --preset rtx6000-2gpu \
    --output-dir ./outputs \
    --dataset gigaspeech \
    --subset m \
    --epochs 20 \
    --resume /u4/h6ly/asr/outputs/checkpoints/checkpoint_epoch_8.pt \
    --distributed-backend gloo

echo "=============================="
echo "Training completed at: $(date)"
echo "=============================="