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


# Ultra-conservative training with minimal model
python scripts/train_flexible.py \
    --preset default \
    --output-dir ./outputs_sweetspot \
    --dataset gigaspeech \
    --subset xs \
    --epochs 10 \
    --lr 5e-5 \
    --batch-size 8 \
    --gradient-clip 0.5 \
    --no-mixed-precision

