#!/bin/bash

#SBATCH --job-name=lmu-asr-multi-gpu
#SBATCH --time=168:00:00
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --partition=CELIASMI
#SBATCH --hint=nomultithread

# Email notifications (update with your watid)
#SBATCH --mail-user=h6ly@uwaterloo.ca
#SBATCH --mail-type=ALL

# Output files
#SBATCH -o JOB%j.out
#SBATCH -e JOB%j-err.out


# Performance optimizations for multi-GPU (restored from working version)
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_COMPILE_MODE=reduce-overhead

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

# Load required modules (match working YOLOv5 setup)
module load cuda/11.8
module load python/3.9

# NCCL configuration (match working YOLOv5 setup)
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

# Multi-GPU training (simplified like working YOLOv5)
torchrun --nproc_per_node=2 --master_port=29501 scripts/train_flexible.py \
--preset default \
--output-dir ./outputs_optimized_mgpu \
--dataset gigaspeech \
--subset m \
--epochs 50 \
--lr 2.5e-3 \
--batch-size 96 \
--gradient-clip 1.0 \
--mixed-precision \
--num-workers 32

# For faster testing, use smaller subsets:
# torchrun --nproc_per_node=2 scripts/train_flexible.py \
#     --preset conservative \
#     --output-dir ./outputs \
#     --dataset gigaspeech \
#     --subset xs \
#     --epochs 5

# For even larger dataset training, use 'l' subset:
# torchrun --nproc_per_node=2 scripts/train_flexible.py \
#     --preset performance \
#     --output-dir ./outputs \
#     --dataset gigaspeech \
#     --subset l \
#     --epochs 15

# Alternative: Use the torchrun-specific script
# torchrun --nproc_per_node=2 scripts/train_torchrun.py

# FALLBACK: If distributed training keeps failing due to NCCL issues:
# python scripts/train_flexible.py \
#     --preset conservative \
#     --output-dir ./outputs \
#     --dataset gigaspeech \
#     --subset m \
#     --epochs 20 \
#     --resume /u4/h6ly/asr/outputs/checkpoints/checkpoint_epoch_8.pt
