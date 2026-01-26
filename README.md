# Deep RL Crowd Simulation: Multi-Agent Experiments

This branch contains the multi-agent DRL corridor navigation experiments, organized into three main folders:

## Folders

- **v1_best_3agents/**
  - Best performing 3-agent models and results (96-100% success)
  - Contains: trained models, evaluation scripts, and 21 GIF videos for 7 scenarios

- **v2_multi_agents/**
  - Multi-agent experiments (5, 7, 10 agents)
  - Contains: evaluation scripts, 15+ GIFs for various agent counts and scenarios

- **v3_scalable/**
  - Scalable version for larger agent counts and more scenarios
  - Contains: models, multiple video folders (3agents, cpu, gpu), evaluation scripts

## Key Features
- PyTorch PPO with curriculum learning
- Evaluation and video generation scripts
- All models, results, and videos are tracked in git (see `.gitignore`)

## Usage
- See each folder for specific training, evaluation, and video generation scripts
- All important results and artifacts are included for reproducibility

## Branch Info
- This branch is for multi-agent experiments only
- The `master` branch remains unchanged

---
For more details, see scripts and documentation in each experiment folder.

## Quick Start

### 1. Environment Setup

```bash
# (Recommended) Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate  # On Windows
# or
source .venv/bin/activate      # On Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Training Multi-Agent DRL

```bash
cd v3_scalable/
# Start training (example script)
python train_multiagent.py --config configs/multiagent_default.yaml
```

- Training logs and models will be saved in `v3_scalable/models/`.

### 3. Evaluating Multi-Agent DRL

```bash
cd v3_scalable/
# Evaluate a trained model
python evaluate_multiagent.py --model models/final_multiagent.zip --output-dir evaluation/
```

- Evaluation results (videos, metrics) will be saved in the specified output directory.

### 4. Visualization

```bash
cd v3_scalable/
# Visualize trajectories and agent interactions
python visualize_multiagent.py --model models/final_multiagent.zip --output-dir visualizations/
```

---

## Notes
- For best results, use a GPU for training.
- See experiment folders for detailed documentation and advanced usage.

---

## Citation
If you use this branch for research, please cite the main project and relevant references in `docs/PRESENTATION_GUIDE.md`.
