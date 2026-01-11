#!/bin/bash
#SBATCH --job-name=realistic_nav
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=booster
#SBATCH --account=hai_pedtrajfore

echo "=========================================="
echo "REALISTIC Multi-Agent Navigation Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

cd /p/home/jusers/al-harrem1/juwels/multii_test_Abd_julich/test3

# Load modules
module load Stages/2025 2>/dev/null || true
module load PyTorch/2.1.0-CUDA-12.1 2>/dev/null || module load PyTorch/2.1.0 2>/dev/null || true

export PYTHONPATH=""
unset PYTHONUSERBASE

# Try to use venv
if [ -f "../multii_test_Abd/venv/bin/activate" ]; then
    source ../multii_test_Abd/venv/bin/activate 2>/dev/null || true
fi

# Verify PyTorch
echo ""
echo "Checking PyTorch..."
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo ""
echo "=========================================="
echo "Starting Training..."
echo "=========================================="

mkdir -p models logs videos

python3 -u trainer.py

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
