#!/bin/bash
#SBATCH --job-name=eval_robust
#SBATCH --account=hai_pedtrajfore
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_videos_%j.out
#SBATCH --error=logs/eval_videos_%j.err

echo "======================================================================"
echo "  MULTII_TEST_ABD2 - Evaluation & Video Generation"
echo "======================================================================"
echo "Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

# Load modules
module purge
module load Stages/2025
module load GCC/13.3.0
module load Python/3.12.3
module load CUDA/12
module load cuDNN/9.5.0.50-CUDA-12
module load PyTorch/2.5.1
module load SciPy-bundle/2024.05
module load matplotlib/3.9.2

# Install imageio
pip install --user imageio pillow 2>/dev/null || true
pip install imageio pillow 2>/dev/null || true

echo "Modules loaded"
python -c "import torch; print('PyTorch:', torch.__version__)"
echo ""

cd /p/home/jusers/al-harrem1/juwels/multii_test_Abd_julich/multii_test_Abd2

python evaluate_videos.py 2>&1 | tee logs/eval_$(date +%Y%m%d_%H%M%S).txt

echo ""
echo "Done! $(date)"
echo "Videos saved in: evaluation_videos/"
