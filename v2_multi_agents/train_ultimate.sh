#!/bin/bash
#SBATCH --job-name=train_ultimate
#SBATCH --partition=booster
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --output=logs/train_ultimate_%j.out
#SBATCH --error=logs/train_ultimate_%j.err
#SBATCH --account=hai_pedtrajfore

echo "=========================================="
echo "ULTIMATE TRAINER - multii_test_Abd2"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Create log directory
mkdir -p logs

# Load modules for JUWELS Booster
module purge
module load Stages/2025
module load GCC/13.3.0
module load Python/3.12.3
module load CUDA/12
module load cuDNN/9.5.0.50-CUDA-12
module load PyTorch/2.5.1

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "../venv" ]; then
    source ../venv/bin/activate
fi

# Set environment variables
export PYTHONUNBUFFERED=1

# Check GPU
echo "GPU Check:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Check Python and PyTorch
echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

echo "Starting training..."
echo "=========================================="

# Run training (ROBUST v2 - Direct Velocity Control)
cd /p/home/jusers/al-harrem1/juwels/multii_test_Abd_julich/multii_test_Abd2

python trainer_ROBUST_v2.py 2>&1 | tee logs/training_robust_$(date +%Y%m%d_%H%M%S).txt

echo ""
echo "=========================================="
echo "Training finished at: $(date)"
echo "=========================================="
