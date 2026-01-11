#!/bin/bash
#SBATCH --job-name=train_corridor
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --account=hai_pedtrajfore

echo "========================================"
echo "TRAIN CORRIDOR NAVIGATION MODEL"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

# Setup environment
cd /p/home/jusers/al-harrem1/juwels/multii_test_Abd_julich/multii_test_Abd
export PYTHONUNBUFFERED=1

# Run training with curriculum learning
# The script will train on: straight -> wall_bottom -> wall_top -> s_curve -> narrow -> chicane -> mixed
python train_simple_robust.py \
    --num-agents 3 \
    --output-dir models/new_training_$(date +%Y%m%d_%H%M%S)

echo ""
echo "========================================"
echo "Training complete: $(date)"
echo "========================================"
