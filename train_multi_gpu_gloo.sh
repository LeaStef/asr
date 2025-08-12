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

# Create logs directory
mkdir -p logs

# Use Gloo backend instead of NCCL (more stable for RTX 6000)
export TORCH_DISTRIBUTED_BACKEND=gloo
export OMP_NUM_THREADS=8

# Multi-GPU training with Gloo backend

# Multi-GPU training with Gloo backend
torchrun --nproc_per_node=2 scripts/train_flexible.py \
    --preset rtx6000-2gpu \
    --output-dir ./outputs \
    --dataset gigaspeech \
    --subset m \
    --epochs 20 \
    --resume /u4/h6ly/asr/outputs/checkpoints/checkpoint_epoch_8.pt

