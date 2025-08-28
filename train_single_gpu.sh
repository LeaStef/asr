#!/bin/bash

#SBATCH --job-name=lmu-asr-single-gpu
#SBATCH --time=168:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --partition=CELIASMI
#SBATCH --hint=nomultithread

# Email notifications
#SBATCH --mail-user=h6ly@uwaterloo.ca
#SBATCH --mail-type=ALL

# Output files
#SBATCH -o JOB%j.out
#SBATCH -e JOB%j-err.out

# Performance optimizations for single GPU
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

# Check GPU status
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu --format=csv

# Monitor memory during training (run in background)
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu --format=csv -l 30 > gpu_memory_single_log.csv &
NVIDIA_SMI_PID=$!

echo "Started memory monitoring - GPU log: gpu_memory_single_log.csv"

# Single GPU training - increased batch size to utilize full GPU
python scripts/train_flexible.py \
    --batch-size 128 \
    --lr 2e-4 \
    --epochs 50 \
    --mixed-precision \
    --num-workers 16 \
    --output-dir ./outputs_single_gpu \
    --dataset gigaspeech \
    --subset m

# Stop memory monitoring
kill $NVIDIA_SMI_PID 2>/dev/null
echo "Training completed - memory logs saved"