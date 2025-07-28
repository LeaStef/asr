# SLURM Training Guide for PyTorch LMU-ASR

This guide explains how to run training jobs on a SLURM cluster using the provided batch scripts.

## Prerequisites

1. Clone the repository: `git clone https://github.com/h6ly/asr.git`
2. Set up virtual environment:
   ```bash
   cd ~/asr
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Update paths and email address in the `.sh` files

## Available Scripts

### 1. Single Job Training (`train_single.sh`)

For basic single-GPU training with fixed parameters.

```bash
sbatch train_single.sh
```

**Configuration:**
- 1 GPU, 32GB RAM, 24 hours
- Uses GigaSpeech dataset (xs subset)
- Batch size: 64, Learning rate: 4e-3

### 2. Hyperparameter Sweep (`train_array.sh`)

For running multiple experiments with different hyperparameters.

```bash
sbatch train_array.sh
```

**Configuration:**
- Array job: 10 tasks (0-9)
- Tests combinations of:
  - Batch sizes: [32, 64, 128]
  - Learning rates: [1e-3, 2e-3, 4e-3, 8e-3]
  - Presets: ["default", "l40s-1gpu"]
- Each job gets unique output directory
- 1 GPU, 32GB RAM, 12 hours per task

### 3. Multi-GPU Training (`train_multi_gpu.sh`)

For distributed training on multiple GPUs.

```bash
sbatch train_multi_gpu.sh
```

**Configuration:**
- 2 GPUs, 64GB RAM, 12 hours
- Uses torchrun for distributed training
- Optimized for L40S GPU configuration

## Customization

### Update Environment Path

In each `.sh` file, update these lines:
```bash
# Change this to your actual paths  
log_dir=$HOME/asr
source $HOME/asr/venv/bin/activate
```

### Update Email Address

Replace `h6ly@uwaterloo.ca` with your actual email:
```bash
#SBATCH --mail-user=your-watid@uwaterloo.ca
```

### Modify Resources

Adjust SLURM parameters based on your needs:
```bash
#SBATCH --time=24:00:00        # Time limit
#SBATCH --mem=32GB             # Memory
#SBATCH --cpus-per-task=8      # CPU cores
#SBATCH --gres=gpu:1           # GPU count
#SBATCH --partition=gpu        # Queue/partition
```

### Change Training Parameters

Modify the training command in each script:
```bash
python scripts/train_flexible.py \
    --batch-size 64 \
    --lr 4e-3 \
    --epochs 50 \
    --mixed-precision \
    --preset default \
    --output-dir ./outputs \
    --dataset gigaspeech \
    --subset xs
```

## Monitoring Jobs

### Check job status:
```bash
squeue -u $USER
```

### View job details:
```bash
scontrol show job <job_id>
```

### Cancel a job:
```bash
scancel <job_id>
```

### View output logs:
```bash
tail -f logs/slurm-<job_id>.out
```

## Output Files

### Checkpoints
- Saved to `outputs/checkpoints/` (or custom `--output-dir`)
- Files: `checkpoint_epoch_{epoch}.pt` and `checkpoint_epoch_{epoch}_best.pt`

### Logs
- SLURM logs: `logs/slurm-<job_id>.out` and `logs/slurm-<job_id>.err`
- Training logs: Written to output directory

### Job Information
- Array jobs save `job_info.txt` with parameters used

## Example Checkpoint Usage

```python
import torch

# Load checkpoint
checkpoint = torch.load('outputs/checkpoints/checkpoint_epoch_10_best.pt')

# Extract information
epoch = checkpoint['epoch']
model_state = checkpoint['model_state_dict']
optimizer_state = checkpoint['optimizer_state_dict']
loss = checkpoint['loss']

print(f"Loaded checkpoint from epoch {epoch} with loss {loss}")
```

## Tips

1. **Start small**: Test with short time limits and small datasets first
2. **Monitor resources**: Check GPU/memory usage with `nvidia-smi`
3. **Use arrays**: For hyperparameter sweeps, use array jobs instead of submitting many individual jobs
4. **Save frequently**: The scripts are configured for regular checkpointing
5. **Check logs**: Add plenty of print statements for debugging

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure the conda environment is properly activated
2. **CUDA errors**: Verify GPU availability and PyTorch CUDA installation
3. **Memory errors**: Reduce batch size or increase `--mem`
4. **Time limits**: Increase `--time` for longer training runs

### Debug Mode

For testing, you can run a quick debug session:
```bash
python scripts/train_flexible.py --debug --dry-run
```

This will validate your configuration without actually training.