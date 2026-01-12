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
