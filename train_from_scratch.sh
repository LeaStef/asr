#!/bin/bash

#SBATCH --job-name=lmu-asr-from-scratch
#SBATCH --time=168:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=CELIASMI

# Email notifications
#SBATCH --mail-user=h6ly@uwaterloo.ca
#SBATCH --mail-type=ALL

# Output files
#SBATCH -o JOB%j.out
#SBATCH -e JOB%j-err.out


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


# Create fresh logs directory
rm -rf logs_scratch
mkdir -p logs_scratch
rm -rf outputs_scratch
mkdir -p outputs_scratch


# Ultra-conservative fresh training
python scripts/train_single_gpu.py \
    --output-dir ./outputs_scratch \
    --dataset gigaspeech \
    --subset xs \
    --epochs 5 \
    --batch-size 8 \
    --lr 5e-4 \
    --disable-mixed-precision

