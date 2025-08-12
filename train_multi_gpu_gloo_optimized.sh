#!/bin/bash

#SBATCH --job-name=lmu-asr-gloo-optimized
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


# Gloo backend optimizations (NO NCCL!)
export TORCH_DISTRIBUTED_BACKEND=gloo
export GLOO_SOCKET_IFNAME=lo  # Use loopback interface
export GLOO_DEVICE_TRANSPORT=TCP  # Force TCP transport
export OMP_NUM_THREADS=8  # Optimize CPU threading for Gloo
export MKL_NUM_THREADS=8

# Memory and process optimizations
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128


# Create logs directory
mkdir -p logs


# Multi-GPU training with Gloo (NO NCCL)
torchrun --nproc_per_node=2 scripts/train_flexible.py \
    --preset rtx6000-2gpu \
    --output-dir ./outputs \
    --dataset gigaspeech \
    --subset m \
    --epochs 20 \
    --resume /u4/h6ly/asr/outputs/checkpoints/checkpoint_epoch_8.pt

