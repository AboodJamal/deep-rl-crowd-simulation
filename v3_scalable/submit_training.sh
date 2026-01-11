#!/bin/bash
#SBATCH --job-name=test3_robust
#SBATCH --output=logs/train_robust_%j.out
#SBATCH --error=logs/train_robust_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --account=hai_pedtrajfore

# ROBUST TRAINING for test3 Multi-Agent Navigation
# Features:
#   - Solid obstacles (no passing through)
#   - Faster agents (max_speed=3.0)
#   - Extended curriculum training
#   - 5 agents

echo "=============================================="
echo "TEST3 ROBUST TRAINING"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=============================================="

# Setup
cd /p/home/jusers/al-harrem1/juwels/multii_test_Abd_julich/test3
mkdir -p logs

# Activate environment
source /p/home/jusers/al-harrem1/juwels/multii_test_Abd_julich/multii_test_Abd/venv/bin/activate

# Show GPU info
nvidia-smi

# Run training
echo ""
echo "Starting robust training..."
echo ""

python train_robust.py \
    --output-dir models_robust \
    --num-agents 5 \
    --device cuda

echo ""
echo "=============================================="
echo "Training completed at $(date)"
echo "=============================================="

