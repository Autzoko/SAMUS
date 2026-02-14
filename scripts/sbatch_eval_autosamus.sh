#!/bin/bash
#SBATCH --job-name=eval_autosamus
#SBATCH --output=logs/eval_autosamus_%j.out
#SBATCH --error=logs/eval_autosamus_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo ""

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/ll5582/3DSAM/envs/segmamba

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment"
    exit 1
fi

echo "Conda environment activated: $(which python)"
echo ""

# Change to SAMUS working directory
cd /scratch/ll5582/3DSAM/TransSAM/SAMUS
echo "Changed to directory: $(pwd)"
echo ""

mkdir -p logs result/AllBreast

# ============================================================================
# Evaluate AutoSAMUS on all 5 breast val sets
# ============================================================================

# Set checkpoint path — update this to your best checkpoint
CHECKPOINT="${1:-checkpoints/AllBreast/AutoSAMUS_best.pth}"
echo "Checkpoint: $CHECKPOINT"
echo ""

python scripts/eval_autosamus_breast.py \
    --checkpoint "$CHECKPOINT" \
    --data-root SAMUS_DATA \
    --eval-datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --device cuda:0 \
    --save-json result/AllBreast/eval_results.json

if [ $? -eq 0 ]; then
    echo ""
    echo "Evaluation completed successfully!"
else
    echo ""
    echo "Evaluation failed with exit code $?"
    exit 1
fi

echo "End Time: $(date)"
