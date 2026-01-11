#!/bin/bash
#SBATCH --job-name=make_vids
#SBATCH --account=hai_pedtrajfore
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/make_vids_%j.out
#SBATCH --error=logs/make_vids_%j.err

echo "======================================================================"
echo "  Creating REAL Animated Videos"
echo "======================================================================"
echo "Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

module purge
module load Stages/2025
module load GCC/13.3.0
module load Python/3.12.3
module load CUDA/12
module load cuDNN/9.5.0.50-CUDA-12
module load PyTorch/2.5.1
module load SciPy-bundle/2024.05
module load matplotlib/3.9.2
module load Pillow/10.4.0

echo "Modules loaded"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "from PIL import Image; print('Pillow OK')"
python -c "from matplotlib.animation import PillowWriter; print('PillowWriter OK')"

cd /p/home/jusers/al-harrem1/juwels/multii_test_Abd_julich/multii_test_Abd2

# Clean old videos
rm -rf videos/
mkdir -p videos

python make_videos.py

echo ""
echo "Done! $(date)"
echo "Videos:"
ls -la videos/
