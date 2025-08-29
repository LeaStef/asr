#!/bin/bash

#SBATCH --job-name=lmu-asr-4gpu-optimized
#SBATCH --time=168:00:00
#SBATCH --mem=512GB
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --partition=CELIASMI
#SBATCH --hint=nomultithread

# Email notifications
#SBATCH --mail-user=h6ly@uwaterloo.ca
#SBATCH --mail-type=ALL

# Output files
#SBATCH -o JOB%j.out
#SBATCH -e JOB%j-err.out

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_COMPILE_MODE=max-autotune
export CUDA_MODULE_LOADING=LAZY

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

nvidia-smi
nvidia-smi topo -m
nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu --format=csv

# Monitor memory during training (run in background)
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu --format=csv -l 30 > gpu_memory_4gpu_log.csv &
NVIDIA_SMI_PID=$!

# Monitor system memory
top -b -d 30 -o %MEM | head -20 > system_memory_4gpu_log.txt &
TOP_PID=$!

echo "Started memory monitoring - GPU log: gpu_memory_4gpu_log.csv, System log: system_memory_4gpu_log.txt"

torchrun --nproc_per_node=4 --master_port=29502 scripts/train_flexible.py \
    --batch-size 320 \
    --lr 5e-4 \
    --epochs 35 \
    --mixed-precision \
    --num-workers 64 \
    --gradient-clip 5.0 \
    --output-dir ./outputs_4gpu \
    --dataset gigaspeech \
    --subset m

# Stop memory monitoring
kill $NVIDIA_SMI_PID 2>/dev/null
kill $TOP_PID 2>/dev/null
echo "Training completed - memory logs saved"

# Alternative configurations for different scenarios:

# For xs subset (quick testing):
# torchrun --nproc_per_node=4 --master_port=29502 scripts/train_flexible.py \
#     --batch-size 256 \
#     --lr 5e-4 \
#     --epochs 15 \
#     --mixed-precision \
#     --num-workers 48 \
#     --gradient-clip 5.0 \
#     --output-dir ./outputs_4gpu_xs \
#     --dataset gigaspeech \
#     --subset xs

# For l subset (large dataset):
# torchrun --nproc_per_node=4 --master_port=29502 scripts/train_flexible.py \
#     --batch-size 512 \
#     --lr 8e-4 \
#     --epochs 25 \
#     --mixed-precision \
#     --num-workers 80 \
#     --gradient-clip 3.0 \
#     --output-dir ./outputs_4gpu_large \
#     --dataset gigaspeech \
#     --subset l