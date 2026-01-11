#!/bin/bash
#SBATCH --job-name=test3_eval_3agents
#SBATCH --output=logs/eval_3agents_%j.out
#SBATCH --error=logs/eval_3agents_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --account=hai_pedtrajfore

echo "========================================"
echo "TEST3 - Evaluate with 3 AGENTS (Fixed Goals)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

echo "NOTE: Model was trained on 5 agents, testing with 3 agents"
echo "      (should work better with fewer agents + fixed goal positioning)"
echo ""

# Setup environment
cd /p/home/jusers/al-harrem1/juwels/multii_test_Abd_julich/test3
source /p/home/jusers/al-harrem1/juwels/multii_test_Abd_julich/multii_test_Abd/venv/bin/activate
export PYTHONUNBUFFERED=1

echo "Python: $(which python)"
echo ""

# FINAL GPU-trained model (mixed stage - best model)
MODEL_PATH="models_robust_20251207_145134/mixed/final_model.pt"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found: $MODEL_PATH"
    echo "Available models:"
    find models_robust_20251207_145134 -name "final_model.pt"
    exit 1
fi

echo "Evaluating with 3 AGENTS: $MODEL_PATH"
echo "Model was trained on 5 agents, testing generalization to 3 agents"
echo "Goal positioning has been FIXED to prevent wall issues"
echo ""

# Run evaluation with 3 agents
python evaluate_cpu_models.py \
    --model-path "$MODEL_PATH" \
    --num-agents 3 \
    --num-episodes 50 \
    --device cuda \
    --video-dir videos_3agents

echo ""
echo "========================================"
echo "Evaluation complete: $(date)"
echo "========================================"

