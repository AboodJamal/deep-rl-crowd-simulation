# Trained Models

This directory contains trained DRL agent checkpoints.

## Main Model

**`ultimate_generalized_agent.zip`**
- **Training Steps:** 3,900,000
- **Training Stages:** 12 curriculum stages
- **Training Time:** ~17.5 hours
- **Success Rate:** 90% (eval_2m)
- **Date:** December 6, 2025

**`ultimate_generalized_agent_vecnormalize.pkl`**
- Normalization statistics for observations
- **Required** for proper model inference

## Stage Checkpoints

| Stage | File | Steps | Success Rate |
|-------|------|-------|--------------|
| 1 | `ultimate_generalized_agent_stage1.zip` | 300k | 28% |
| 2 | `ultimate_generalized_agent_stage2.zip` | 600k | 96% |
| 3 | `ultimate_generalized_agent_stage3.zip` | 900k | 97% |
| 4 | `ultimate_generalized_agent_stage4.zip` | 1.2M | 94% |
| 5 | `ultimate_generalized_agent_stage5.zip` | 1.5M | 88% |
| 6 | `ultimate_generalized_agent_stage6.zip` | 1.8M | 99% |
| 7 | `ultimate_generalized_agent_stage7.zip` | 2.1M | 96% |
| 8 | `ultimate_generalized_agent_stage8.zip` | 2.4M | 97% |
| 9 | `ultimate_generalized_agent_stage9.zip` | 2.7M | 91% |
| 10 | `ultimate_generalized_agent_stage10.zip` | 3.3M | 86% |
| 11 | `ultimate_generalized_agent_stage11.zip` | 3.6M | 96% |
| 12 | `ultimate_generalized_agent_stage12.zip` | 3.9M | 90% |

Each stage also has a corresponding `_vecnormalize.pkl` file.

## Loading Models

### Basic Loading
```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

# Load model
model = PPO.load("models/ultimate_generalized_agent")

# Load normalization stats
vec_norm = VecNormalize.load("models/ultimate_generalized_agent_vecnormalize.pkl", env)
```

### For Evaluation
```python
# Load with deterministic policy
model = PPO.load("models/ultimate_generalized_agent")
vec_norm = VecNormalize.load("models/ultimate_generalized_agent_vecnormalize.pkl", env)
vec_norm.training = False  # Disable training mode
vec_norm.norm_reward = False  # Don't normalize rewards during evaluation

# Get action
obs = vec_norm.normalize_obs(raw_obs)
action, _ = model.predict(obs, deterministic=True)
```

## Model Architecture

```
Input: 50D observation
  ├─> Base Features (11) → MLP [256]
  ├─> Raycasts (36) → CNN [64→128] → Attention → LSTM [128]
  └─> Enhanced (3) → MLP [64]
        ↓
  Concatenate [448D]
        ↓
  Actor Head → Action (2D continuous)
  Critic Head → Value (1D)
```

### Observation Space (50D)
- Position (x, y): 2D
- Velocity (vx, vy): 2D
- Heading (cos, sin): 2D
- Distance to goal: 1D
- Angle to goal: 1D
- Speed: 1D
- Raycasts: 36D (36 rays × 12m range)
- Corner awareness: 1D
- Goal visibility: 1D
- Time normalized: 1D

### Action Space (2D)
- Linear velocity: [-1.4, 1.4] m/s
- Angular velocity: [-1.8, 1.8] rad/s

## Training Hyperparameters

```python
algorithm = "PPO"
learning_rate = 3e-4 (adaptive)
batch_size = 256
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.02 (adaptive)
vf_coef = 0.5
max_grad_norm = 0.5
```

## File Sizes

**Note:** Model files are large (500-800 MB each due to LSTM states).
- Main model: ~750 MB
- VecNormalize: ~1-2 KB
- Stage checkpoints: ~750 MB each

## Git LFS Recommendation

If pushing to GitHub, use Git LFS for model files:

```bash
git lfs track "models/*.zip"
git lfs track "models/*.pkl"
```

## Inference Performance

- **Forward pass:** ~5ms (CPU), ~1ms (GPU)
- **Real-time capable:** Yes (100+ Hz)
- **Batch inference:** Supported via VecEnv

## Model Limitations

1. **Dense crowds:** Performance drops in very dense obstacle scenarios
2. **Path comfort:** May take narrow paths that humans would avoid
3. **Deterministic:** Same input always produces same output
4. **Single agent:** Trained for single-agent navigation only

## Future Models

Planned improvements:
- Multi-agent training
- Stochastic policies for path diversity
- Comfort-aware reward shaping
- Transfer learning from classical models
