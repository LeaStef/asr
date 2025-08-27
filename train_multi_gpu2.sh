#!/bin/bash

#SBATCH --job-name=lmu-asr-multi-gpu-optimized
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


# Performance optimizations for multi-GPU
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
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

# Optimized NCCL configuration for performance
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=lo
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Check GPU topology and initial memory
nvidia-smi topo -m
nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu --format=csv

# Multi-GPU training with torchrun
# Optimized for GigaSpeech 'm' subset (~1000 hours, ~200k samples)

# Monitor memory during training (run in background)
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu --format=csv -l 30 > gpu_memory_log.csv &
NVIDIA_SMI_PID=$!

# Monitor system memory
top -b -d 30 -o %MEM | head -20 > system_memory_log.txt &
TOP_PID=$!

echo "Started memory monitoring - GPU log: gpu_memory_log.csv, System log: system_memory_log.txt"

# Optimized multi-GPU training with performance improvements
torchrun --nproc_per_node=2 --master_port=29501 --nnodes=1 --rdzv_backend=c10d scripts/train_flexible.py \
--preset default \
--output-dir ./outputs_optimized_mgpu \
--dataset gigaspeech \
--subset m \
--epochs 50 \
--lr 2.5e-4 \
--batch-size 96 \
--mixed-precision \
--num-workers 32

# Stop memory monitoring
kill $NVIDIA_SMI_PID 2>/dev/null
kill $TOP_PID 2>/dev/null
echo "Training completed - memory logs saved"