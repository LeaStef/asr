#!/bin/bash

#SBATCH --job-name=lmu-asr-single-gpu
# Set resource requirements: Queues are limited to seven day allocations  
# Time format: HH:MM:SS
#SBATCH --time=168:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=CELIASMI

# Email notifications (update with your watid)
#SBATCH --mail-user=h6ly@uwaterloo.ca
#SBATCH --mail-type=ALL

# Output files
#SBATCH -o JOB%j.out
#SBATCH -e JOB%j-err.out


# Set up environment
log_dir=$HOME/asr
mkdir -p $log_dir
cd $log_dir


# Load up your virtual environment  
# Set up environment on watgpu.cs or in interactive session (use `source` keyword)
# Assuming venv is in asr/venv/ or asr/bin/activate
if [ -f "$log_dir/venv/bin/activate" ]; then
    source $log_dir/venv/bin/activate
elif [ -f "$log_dir/bin/activate" ]; then
    source $log_dir/bin/activate
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# RTX 6000 Single GPU training with optimizations

python scripts/train_flexible.py \
    --preset rtx6000-1gpu \
    --output-dir ./outputs \
    --dataset gigaspeech \
    --subset m \
    --epochs 20 \
    --resume /u4/h6ly/asr/outputs/checkpoints/checkpoint_epoch_8.pt

# For larger dataset training, use 's' or 'm' subset instead:
# python scripts/train_flexible.py \
#     --preset rtx6000-1gpu \
#     --output-dir ./outputs \
#     --dataset gigaspeech \
#     --subset s



# Optional: Copy important files to a backup location
# cp -r outputs/ $SCRATCH/lmu-asr-results-$SLURM_JOB_ID/