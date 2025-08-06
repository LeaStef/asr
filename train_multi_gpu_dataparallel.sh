#!/bin/bash

#SBATCH --job-name=lmu-asr-dataparallel
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

echo "==== DataParallel Training (No Distributed, No NCCL) ===="
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "=============================="

# Set up environment
log_dir=$HOME/asr
mkdir -p $log_dir
cd $log_dir

# Load virtual environment
if [ -f "$log_dir/venv/bin/activate" ]; then
    source $log_dir/venv/bin/activate
elif [ -f "$log_dir/bin/activate" ]; then
    source $log_dir/bin/activate
fi

echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# No distributed settings needed - DataParallel handles it
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "=============================="
echo "DataParallel Configuration (NCCL-FREE):"
echo "  Method: torch.nn.DataParallel"
echo "  Communication: Direct GPU-to-GPU (no NCCL/Gloo)"
echo "  GPUs: 2x RTX 6000 Ada Generation"
echo "  Process: Single process, multi-GPU"
echo "  ZERO distributed complexity!"
echo "=============================="

# Create logs directory
mkdir -p logs

echo "Starting DataParallel training (no NCCL)..."

# Single-process multi-GPU training (NO DISTRIBUTED)
python scripts/train_flexible.py \
    --preset rtx6000-2gpu \
    --output-dir ./outputs \
    --dataset gigaspeech \
    --subset m \
    --epochs 20 \
    --resume /u4/h6ly/asr/outputs/checkpoints/checkpoint_epoch_8.pt

echo "=============================="
echo "Training completed at: $(date)"
echo "=============================="