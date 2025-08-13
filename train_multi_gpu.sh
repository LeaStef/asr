#!/bin/bash

#SBATCH --job-name=lmu-asr-multi-gpu
#SBATCH --time=168:00:00
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


# Set up environment
log_dir=$HOME/asr
mkdir -p $log_dir
cd $log_dir


# Load up your virtual environment  
# Set up environment on watgpu.cs or in interactive session (use `source` keyword)
# Assuming venv is in asr/venv/ or asr/bin/activate
if [ -f "$log_dir/venv/bin/activate" ]; then
    source $log_dir/venv/bin/activate
elif [ -f "$log_dir/bin/activate" ]; then
    source $log_dir/bin/activate
fi


# Already in project directory ($HOME/asr)

# Create logs directory if it doesn't exist
mkdir -p logs

# NCCL configuration with full debugging for RTX 6000
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

# Check GPU topology
nvidia-smi topo -m

# Multi-GPU training with torchrun
# Optimized for GigaSpeech 'm' subset (~1000 hours, ~200k samples)

torchrun --nproc_per_node=2 scripts/train_flexible.py \
    --preset rtx6000-2gpu \
    --output-dir ./outputs \
    --dataset gigaspeech \
    --subset m \
    --epochs 20 \

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

