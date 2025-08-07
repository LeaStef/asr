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
echo "Ultra-Conservative Fresh Training:"
echo "  Method: Fresh training from scratch (no checkpoints)"
echo "  Batch size: 8 (ultra small)"
echo "  Learning rate: 5e-4 (ultra conservative)"
echo "  Mixed precision: DISABLED by default"
echo "  Dataset: GigaSpeech XS (smallest subset)"
echo "  Epochs: 5 (quick validation)"
echo "=============================="

echo "Starting single GPU training..."

# Ultra-conservative fresh training from scratch (no resume)
python scripts/train_single_gpu.py \
    --output-dir ./outputs_scratch \
    --dataset gigaspeech \
    --subset xs \
    --epochs 5 \
    --batch-size 8 \
    --lr 5e-4

echo "=============================="
echo "Training completed at: $(date)"
echo "=============================="