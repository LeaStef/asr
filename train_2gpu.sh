#!/bin/bash

#SBATCH --job-name=lmu-asr-2-gpu
#SBATCH --time=168:00:00
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --partition=CELIASMI

# Email notifications (update with your watid)
#SBATCH --mail-user=h6ly@uwaterloo.ca
#SBATCH --mail-type=ALL

# Output files
#SBATCH -o JOB%j.out
#SBATCH -e JOB%j-err.out


# Basic environment setup
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

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

# Optimized NCCL configuration for performance
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=lo

# Reduce NCCL timeout from 600s (10min) to 120s (2min) for faster feedback
export NCCL_TIMEOUT=120


# Check GPU topology
nvidia-smi topo -m

# Multi-GPU training with torchrun
# Optimized for GigaSpeech 'm' subset (~1000 hours, ~200k samples)

# Optimized multi-GPU training with performance improvements
torchrun --nproc_per_node=2 scripts/train_flexible.py \
--preset default \
--output-dir ./outputs_optimized_mgpu \
--dataset gigaspeech \
--subset m \
--epochs 50 \
--lr 2.5e-4 \
--batch-size 32 \
--mixed-precision \
--num-workers 16

# For faster testing, use smaller subsets:
# torchrun --nproc_per_node=2 scripts/train_flexible.py \
#     --preset rtx6000-2gpu \
#     --output-dir ./outputs \
#     --dataset gigaspeech \
#     --subset xs \
#     --epochs 5

# For even larger dataset training, use 'l' subset:
# torchrun --nproc_per_node=2 scripts/train_flexible.py \
#     --preset rtx6000-2gpu \
#     --output-dir ./outputs \
#     --dataset gigaspeech \
#     --subset l \
#     --epochs 15

# Alternative: Use the torchrun-specific script
# torchrun --nproc_per_node=2 scripts/train_torchrun.py

# FALLBACK: If distributed training keeps failing due to NCCL issues:
# python scripts/train_flexible.py \
#     --preset rtx6000-1gpu \
#     --output-dir ./outputs \
#     --dataset gigaspeech \
#     --subset m \
#     --epochs 20 \
#     --resume /u4/h6ly/asr/outputs/checkpoints/checkpoint_epoch_8.pt

