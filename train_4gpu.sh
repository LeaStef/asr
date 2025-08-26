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
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
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

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=lo
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export NCCL_TREE_THRESHOLD=0
export NCCL_MIN_NCHANNELS=8
export NCCL_MAX_NCHANNELS=16
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_ALGO=Tree,Ring
export NCCL_PROTO=Simple

nvidia-smi
nvidia-smi topo -m

torchrun --nproc_per_node=4 --master_port=29502 --nnodes=1 --rdzv_backend=c10d scripts/train_flexible.py \
    --batch-size 320 \
    --lr 5e-4 \
    --epochs 35 \
    --mixed-precision \
    --num-workers 64 \
    --gradient-clip 5.0 \
    --output-dir ./outputs_4gpu \
    --dataset gigaspeech \
    --subset m

# Alternative configurations for different scenarios:

# For xs subset (quick testing on RTX Ada 6000):
# torchrun --nproc_per_node=4 --master_port=29502 scripts/train_flexible.py \
#     --batch-size 256 \
#     --lr 5e-4 \
#     --epochs 15 \
#     --mixed-precision \
#     --num-workers 48 \
#     --gradient-clip 5.0 \
#     --output-dir ./outputs_rtx_ada_xs \
#     --dataset gigaspeech \
#     --subset xs

# For l subset (large dataset on RTX Ada 6000):
# torchrun --nproc_per_node=4 --master_port=29502 scripts/train_flexible.py \
#     --batch-size 512 \
#     --lr 8e-4 \
#     --epochs 25 \
#     --mixed-precision \
#     --num-workers 80 \
#     --gradient-clip 3.0 \
#     --output-dir ./outputs_rtx_ada_large \
#     --dataset gigaspeech \
#     --subset l