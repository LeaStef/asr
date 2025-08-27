#!/bin/bash

#SBATCH --job-name=lmu-asr-3gpu-optimized
#SBATCH --time=168:00:00
#SBATCH --mem=384GB
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:3
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

log_dir=$HOME/asr
mkdir -p $log_dir
cd $log_dir

if [ -f "$log_dir/venv/bin/activate" ]; then
    source $log_dir/venv/bin/activate
elif [ -f "$log_dir/bin/activate" ]; then
    source $log_dir/bin/activate
fi

mkdir -p logs

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=lo
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export NCCL_TREE_THRESHOLD=0
export NCCL_MIN_NCHANNELS=6
export NCCL_MAX_NCHANNELS=12
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_ALGO=Tree,Ring
export NCCL_PROTO=Simple

nvidia-smi
nvidia-smi topo -m

torchrun --nproc_per_node=3 --master_port=29503 --nnodes=1 --rdzv_backend=c10d scripts/train_flexible.py \
    --batch-size 240 \
    --lr 4e-4 \
    --epochs 40 \
    --mixed-precision \
    --num-workers 36 \
    --gradient-clip 5.0 \
    --output-dir ./outputs_rtx_ada_3gpu \
    --dataset gigaspeech \
    --subset m

# Alternative configurations for different scenarios:

# For xs subset (quick testing on RTX Ada 6000):
# torchrun --nproc_per_node=3 --master_port=29503 scripts/train_flexible.py \
#     --batch-size 192 \
#     --lr 4e-4 \
#     --epochs 20 \
#     --mixed-precision \
#     --num-workers 24 \
#     --gradient-clip 5.0 \
#     --output-dir ./outputs_rtx_ada_3gpu_xs \
#     --dataset gigaspeech \
#     --subset xs

# For l subset (large dataset on RTX Ada 6000):
# torchrun --nproc_per_node=3 --master_port=29503 scripts/train_flexible.py \
#     --batch-size 384 \
#     --lr 6e-4 \
#     --epochs 30 \
#     --mixed-precision \
#     --num-workers 48 \
#     --gradient-clip 3.0 \
#     --output-dir ./outputs_rtx_ada_3gpu_large \
#     --dataset gigaspeech \
#     --subset l

# Conservative option with reduced workers for shared cluster:
# torchrun --nproc_per_node=3 --master_port=29503 scripts/train_flexible.py \
#     --batch-size 192 \
#     --lr 3e-4 \
#     --epochs 45 \
#     --mixed-precision \
#     --num-workers 18 \
#     --gradient-clip 5.0 \
#     --output-dir ./outputs_rtx_ada_3gpu_conservative \
#     --dataset gigaspeech \
#     --subset m
