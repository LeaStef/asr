#!/bin/bash

#SBATCH --job-name=lmu-asr-sweep
# Set resource requirements: Queues are limited to seven day allocations
# Time format: HH:MM:SS  
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --array=0-9

# Email notifications (update with your watid)
#SBATCH --mail-user=h6ly@uwaterloo.ca
#SBATCH --mail-type=ALL

# Output files
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err

# Set up environment
log_dir=$HOME/asr
cd $log_dir

# Activate pip virtual environment
source asr/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Define parameter arrays for hyperparameter sweep
batch_sizes=(32 64 128)
learning_rates=(1e-3 2e-3 4e-3 8e-3)
presets=("default" "l40s-1gpu")

# Calculate parameters based on array task ID
num_batch_sizes=${#batch_sizes[@]}
num_learning_rates=${#learning_rates[@]}
num_presets=${#presets[@]}

# Map SLURM_ARRAY_TASK_ID to parameter combinations
batch_idx=$((SLURM_ARRAY_TASK_ID % num_batch_sizes))
lr_idx=$(((SLURM_ARRAY_TASK_ID / num_batch_sizes) % num_learning_rates))
preset_idx=$((SLURM_ARRAY_TASK_ID / (num_batch_sizes * num_learning_rates)))

batch_size=${batch_sizes[$batch_idx]}
learning_rate=${learning_rates[$lr_idx]}
preset=${presets[$preset_idx]}

# Create unique output directory for this job
output_dir="outputs/job_${SLURM_ARRAY_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"
mkdir -p $output_dir

# Run training with parameters specific to this array task
python scripts/train_flexible.py \
    --batch-size $batch_size \
    --lr $learning_rate \
    --epochs 30 \
    --mixed-precision \
    --preset $preset \
    --output-dir $output_dir \
    --dataset gigaspeech \
    --subset xs

# Save job info to output directory
cat > $output_dir/job_info.txt << EOF
Job completed successfully
SLURM_JOB_ID: $SLURM_JOB_ID
SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID
Batch size: $batch_size
Learning rate: $learning_rate
Preset: $preset
Completion time: $(date)
EOF