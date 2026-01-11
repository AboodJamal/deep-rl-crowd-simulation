#!/bin/bash
#SBATCH --job-name=eval_video
#SBATCH --output=logs/eval_video_%j.out
#SBATCH --error=logs/eval_video_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=develbooster
#SBATCH --account=hai_pedtrajfore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

echo "=========================================="
echo "🎬 Enhanced Video Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

# Load modules
module purge
module load Stages/2024
module load GCCcore/.12.3.0
module load Python/3.11.3

# Create logs directory
mkdir -p logs

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Check for required packages
python -c "import torch; import matplotlib; import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install --user torch matplotlib numpy pillow
fi

echo ""
echo "Starting enhanced evaluation with videos..."
echo ""

# Run evaluation with multiple attempts per scenario
python evaluate_with_videos.py \
    --model models/best_model/mixed/final_model.pt \
    --scenarios straight single_wall_bottom single_wall_top s_curve narrow_passage chicane mixed \
    --num-agents 3 \
    --num-episodes 10 \
    --video-episodes 1 \
    --num-tries 3 \
    --speedup 1 \
    --output-dir results/videos

echo ""
echo "=========================================="
echo "✅ Evaluation complete!"
echo "Videos saved to: videos_enhanced/"
echo "=========================================="
