#!/bin/bash

#SBATCH --job-name=lmu-asr-single-gpu-optimized
#SBATCH --time=72:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=CELIASMI
#SBATCH --exclusive
#SBATCH --hint=nomultithread

# Email notifications
#SBATCH --mail-user=h6ly@uwaterloo.ca
#SBATCH --mail-type=ALL

# Output files
#SBATCH -o JOB%j.out
#SBATCH -e JOB%j-err.out


# Performance optimizations
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

# Create logs directory with better I/O
mkdir -p logs

# GPU optimization
nvidia-smi
export CUDA_VISIBLE_DEVICES=0


# Optimized training with performance improvements
python -u scripts/train_flexible.py \
    --preset default \
    --output-dir ./outputs_optimized \
    --dataset gigaspeech \
    --subset m \
    --epochs 50 \
    --lr 1e-4 \
    --batch-size 48 \
    --mixed-precision \
    --num-workers 16

