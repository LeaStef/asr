#!/bin/bash

#SBATCH --job-name=lmu-asr-single-gpu
# Set resource requirements: Queues are limited to seven day allocations  
# Time format: HH:MM:SS
#SBATCH --time=150:00:00
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

echo "==== SLURM Job Information ===="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "=============================="

# Set up environment
log_dir=$HOME/asr
cd $log_dir

echo "Working directory: $(pwd)"
echo "Activating conda environment..."

# Load up your virtual environment  
# Set up environment on watgpu.cs or in interactive session (use `source` keyword)
source asr/bin/activate

echo "Virtual environment activated: $VIRTUAL_ENV"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=============================="
echo "Starting single GPU training..."
echo "=============================="

# RTX 6000 Single GPU training
echo "Training configuration:"
echo "  GPU: RTX 6000 Ada Generation (48GB)"
echo "  Dataset: GigaSpeech subset 'm' (~10k hours)"
echo "  Preset: rtx6000-1gpu (96 batch size, 3e-3 LR, 30 epochs)"
echo "  Estimated time: ~6 days"
echo "  Checkpoints saved every epoch to ./outputs/checkpoints/"

python scripts/train_flexible.py \
    --preset rtx6000-1gpu \
    --output-dir ./outputs \
    --dataset gigaspeech \
    --subset m


echo "=============================="
echo "Training completed at: $(date)"
echo "=============================="

# Optional: Copy important files to a backup location
# cp -r outputs/ $SCRATCH/lmu-asr-results-$SLURM_JOB_ID/