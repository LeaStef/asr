#!/bin/bash

#SBATCH --job-name=lmu-asr-single-gpu-debug
#SBATCH --time=168:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=CELIASMI

# Email notifications
#SBATCH --mail-user=h6ly@uwaterloo.ca
#SBATCH --mail-type=ALL

# Output files
#SBATCH -o JOB%j.out
#SBATCH -e JOB%j-err.out

echo "==== Single GPU Training (Debug NaN Issues) ===="
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "=============================="

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

echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Create logs directory
mkdir -p logs

echo "=============================="
echo "Single GPU Debug Configuration:"
echo "  Method: Standard PyTorch single GPU"
echo "  Batch size: 16 (reduced from 32)"
echo "  Learning rate: 1e-3 (reduced from 2e-3)"
echo "  Mixed precision: Enabled (can disable with --disable-mixed-precision)"
echo "  Focus: Debug NaN losses from DataParallel"
echo "=============================="

echo "Starting single GPU training..."

# Conservative single GPU training
python scripts/train_single_gpu.py \
    --output-dir ./outputs \
    --dataset gigaspeech \
    --subset m \
    --epochs 20 \
    --batch-size 16 \
    --lr 1e-3 \
    --resume /u4/h6ly/asr/outputs/checkpoints/checkpoint_epoch_8.pt

echo "=============================="
echo "Training completed at: $(date)"
echo "=============================="