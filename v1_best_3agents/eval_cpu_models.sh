#!/bin/bash
#SBATCH --job-name=multi_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --account=hai_pedtrajfore

echo "========================================"
echo "EVALUATE BEST MODEL"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

# Setup environment
cd /p/home/jusers/al-harrem1/juwels/multii_test_Abd_julich/multii_test_Abd
export PYTHONUNBUFFERED=1

echo "Python: $(which python)"
echo ""

# Best model path
MODEL_PATH="models/best_model/mixed/final_model.pt"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found: $MODEL_PATH"
    exit 1
fi

echo "Evaluating model: $MODEL_PATH"
echo ""

# Run evaluation
python evaluate_cpu_models.py \
    --model-path "$MODEL_PATH" \
    --num-agents 3 \
    --num-episodes 50 \
    --device cuda

echo ""
echo "========================================"
echo "Evaluation complete: $(date)"
echo "========================================"

