#!/bin/bash
#SBATCH --job-name=test3_eval_cpu
#SBATCH --output=logs/eval_cpu_%j.out
#SBATCH --error=logs/eval_cpu_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --account=hai_pedtrajfore

echo "========================================"
echo "TEST3 - Evaluate CPU-Trained Models"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

# Setup environment
cd /p/home/jusers/al-harrem1/juwels/multii_test_Abd_julich/test3
source /p/home/jusers/al-harrem1/juwels/multii_test_Abd_julich/multii_test_Abd/venv/bin/activate
export PYTHONUNBUFFERED=1

echo "Python: $(which python)"
echo ""

# Most recent CPU-trained model
MODEL_PATH="models_robust_20251207_145134/straight/final_model.pt"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found: $MODEL_PATH"
    echo "Available models:"
    find models_robust* -name "final_model.pt" | head -5
    exit 1
fi

echo "Evaluating model: $MODEL_PATH"
echo ""

# Run evaluation
python evaluate_cpu_models.py \
    --model-path "$MODEL_PATH" \
    --num-agents 5 \
    --num-episodes 50 \
    --device cuda

echo ""
echo "========================================"
echo "Evaluation complete: $(date)"
echo "========================================"

