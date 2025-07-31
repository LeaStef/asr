#!/bin/bash

#SBATCH --job-name=lmu-asr-single-large-batch
#SBATCH --time=200:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=CELIASMI

# Email notifications
#SBATCH --mail-user=h6ly@uwaterloo.ca
#SBATCH --mail-type=ALL

# Output files
#SBATCH -o JOB%j.out
#SBATCH -e JOB%j-err.out

echo "==== Single GPU with Large Effective Batch Size ===="
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

echo "=============================="
echo "Large Batch Training Configuration:"
echo "  Strategy: Gradient accumulation to simulate 2-GPU batch size"
echo "  Effective batch size: 128 (64 per GPU equivalent)"
echo "  Accumulation steps: 4 (64 * 4 = 256 total)"
echo "  Memory efficient training"
echo "=============================="

# Single GPU with gradient accumulation (simulates multi-GPU batch size)
python scripts/train_flexible.py \
    --preset rtx6000-1gpu-large-batch \
    --output-dir ./outputs \
    --dataset gigaspeech \
    --subset m \
    --epochs 20 \
    --resume /u4/h6ly/asr/outputs/checkpoints/checkpoint_epoch_8.pt

echo "=============================="
echo "Training completed at: $(date)"
echo "=============================="