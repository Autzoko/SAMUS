#!/bin/bash
#SBATCH --job-name=autosamus_allbreast
#SBATCH --output=logs/autosamus_allbreast_%j.out
#SBATCH --error=logs/autosamus_allbreast_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Activate conda environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/ll5582/3DSAM/envs/segmamba

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment"
    exit 1
fi

echo "Conda environment activated successfully"
echo "Python path: $(which python)"
echo ""

# Change to SAMUS working directory
cd /scratch/ll5582/3DSAM/TransSAM/SAMUS
echo "Changed to directory: $(pwd)"
echo ""

# Create logs directory
mkdir -p logs

# Fix: train.py hardcodes CUDA_VISIBLE_DEVICES='3'; use SLURM-allocated GPU instead
# SLURM sets CUDA_VISIBLE_DEVICES automatically; override before Python imports torch
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}

# ============================================================================
# Train AutoSAMUS on combined breast dataset (all 5 datasets)
# ============================================================================
echo "Starting AutoSAMUS training on AllBreast..."
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""

# NOTE: You may need to edit train.py line 3 to remove/comment:
#   os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# Or replace it with:
#   os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

python train.py \
    --modelname AutoSAMUS \
    --task AllBreast \
    --encoder_input_size 256 \
    --low_image_size 128 \
    --vit_name vit_b \
    --batch_size 8 \
    --base_lr 0.0005 \
    --warmup \
    --warmup_period 250 \
    -keep_log True

if [ $? -eq 0 ]; then
    echo ""
    echo "Training completed successfully!"
else
    echo ""
    echo "Training failed with exit code $?"
    exit 1
fi

echo "End Time: $(date)"
