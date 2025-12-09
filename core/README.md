# Core Training and Evaluation Code

This directory contains the main implementation files for the Deep RL pedestrian navigation system.

## Files

### Training
- **`ultimate_curriculum_trainer.py`** - Main training script with 12-stage curriculum learning
- **`ultimate_domain_randomization_env.py`** - Gymnasium environment with domain randomization
- **`advanced_policy_network.py`** - Custom CNN + Attention + LSTM policy architecture

### Evaluation
- **`ultimate_evaluation.py`** - Comprehensive evaluation script with video generation

### Utilities
- **`numpy_compat_fix.py`** - Compatibility fix for numpy/stable-baselines3

## Usage

### Training
```bash
# Train from scratch (3.9M steps, ~17 hours)
python core/ultimate_curriculum_trainer.py --timesteps 3900000

# Continue training from checkpoint
python core/ultimate_curriculum_trainer.py --timesteps 500000 --continue-from models/stage6
```

### Evaluation
```bash
# Evaluate trained model
python core/ultimate_evaluation.py \
    --model models/ultimate_generalized_agent.zip \
    --output-dir evaluation/results/eval_test \
    --episodes-per-scenario 5
```

## Architecture Overview

### Environment
- **Observation Space:** 50D (position, velocity, 36 raycasts, enhanced features)
- **Action Space:** 2D continuous (linear velocity, angular velocity)
- **Corridor Types:** Standard, L-shaped, T-shaped
- **Difficulty Levels:** super_easy, easy, medium, hard, mixed, ultra

### Neural Network
```
Input (50D)
  ├─> Base Features (11) → MLP
  ├─> Raycasts (36) → CNN → Attention → LSTM
  └─> Enhanced Features (3) → MLP
        ↓
  Concatenate → Actor/Critic Heads
        ↓
  Actions (2D) / Value (1D)
```

### Reward Structure
- Progress: +10.0 per meter toward goal
- Goal reached: +1000.0
- Collision: -20.0 (progressive penalty)
- Bad behavior: Spinning, backward movement, stalling penalties

## Curriculum Stages

| Stage | Name | Difficulty | Shapes | Steps |
|-------|------|------------|--------|-------|
| 1 | Super Easy Standard | super_easy | standard | 300k |
| 2 | Standard Sparse | easy | standard | 300k |
| 3 | Standard Dense Easy | easy | standard | 300k |
| 4 | Standard Medium | medium | standard | 300k |
| 5 | Standard Hard | hard | standard | 300k |
| 6 | L/T Super Easy | super_easy | lshaped, tshaped | 300k |
| 7 | L/T Easy | easy | lshaped, tshaped | 300k |
| 8 | L/T Medium | medium | lshaped, tshaped | 300k |
| 9 | L/T Hard | hard | lshaped, tshaped | 300k |
| 10 | All Mixed | mixed | all | 600k |
| 11 | Pattern Navigation | medium | standard | 300k |
| 12 | Ultra Challenge | ultra | all | 300k |

**Total:** 3,900,000 timesteps

## Configuration

Edit parameters in `ultimate_curriculum_trainer.py`:

```python
# Training hyperparameters
learning_rate = 3e-4
batch_size = 256
n_epochs = 10
gamma = 0.99
clip_range = 0.2

# Environment parameters
max_episode_steps = 500
agent_radius = 0.225
max_velocity = 1.4
max_angular_velocity = 1.8
```

## Outputs

### Training
- **Models:** `models/ultimate_generalized_agent_stageN.zip`
- **Normalization:** `models/ultimate_generalized_agent_stageN_vecnormalize.pkl`
- **Logs:** `curriculum_logs/ultimate_training_summary.json`
- **Tensorboard:** `runs/ultimate_YYYYMMDD_HHMMSS/`

### Evaluation
- **Results:** `evaluation/results/eval_NAME/complete_results.json`
- **Videos:** `evaluation/results/eval_NAME/SCENARIO/epNNN_[success|failure].mp4`
- **Trajectories:** Embedded in complete_results.json

## Dependencies

See `requirements.txt` in root directory.

Key dependencies:
- `stable-baselines3` - PPO implementation
- `gymnasium` - RL environment interface
- `torch` - Neural network framework
- `numpy` - Numerical operations
- `opencv-python` - Video generation
- `wandb` - Experiment tracking (optional)
