# Dynamic Obstacles Navigation - DRL Agent

This folder contains the final working implementation for training and testing a DRL agent navigating through dynamic obstacles in a corridor environment.

## 📁 Folder Structure

```
dynamic_obstacles_working/
├── train_corridor_env.py    # Training script with PPO
├── test_and_visualize.py    # Testing script that generates videos
├── trained_model/            # Trained PPO model (1M timesteps)
│   └── final_model.zip
├── videos/                   # Generated test videos
│   ├── episode_01_success.mp4
│   ├── episode_02_success.mp4
│   ├── episode_03_success.mp4
│   ├── episode_04_success.mp4
│   └── episode_05_success.mp4
└── results/                  # (Future: JSON results, metrics)
```

## 🎯 Environment Details

**DynamicCorridorEnv** - Simple corridor with moving obstacles:
- **Size**: 12m × 6m corridor
- **Agent**: Starts at left (1.0, 3.0), radius 0.25m
- **Goal**: Right side (11.0, 3.0)
- **Obstacles**: 3 dynamic circles moving vertically (up/down)
  - Positions: x = [4.0, 6.5, 9.0]
  - Radius: 0.35m
  - Velocity: 0.5-0.9 m/s (randomized)
  - Bounce at y boundaries (0.8 and 5.2)

## 🔍 Observation Space (18 dimensions)

**Base features (6):**
- dx, dy to goal
- distance to goal
- heading error
- velocity (vx, vy)

**Raycasts (12):**
- 12 rays at 30° intervals
- 3.0m range
- Detect walls and dynamic obstacles
- Normalized output (0=max range, 1=collision)

## 🎁 Reward Function

- **Progress**: +30 per meter toward goal
- **Success**: +300 for reaching goal
- **Collision**: -20 penalty (does NOT terminate episode)
- **Time step**: -1 per step
- **Movement**: Small velocity reward

## 🤖 Training Configuration

- **Algorithm**: PPO (Stable-Baselines3)
- **Network**: [256, 256] MLP
- **Learning rate**: 3e-4
- **Total timesteps**: 1,000,000
- **Batch size**: 128
- **n_steps**: 4096
- **Training time**: ~20-30 minutes

## 📊 Performance Results

**Test results (20 episodes):**
- ✅ **Success Rate**: 100%
- 🎯 **Avg Collisions**: 0.5-1.4 per episode
- ⏱️ **Avg Steps**: 66-90 steps to goal
- 📏 **Final Distance**: 0.25-0.40m from goal center

## 🎬 Video Generation

The `test_and_visualize.py` script generates multiple test videos:
- ONE agent per video
- Different obstacle patterns (randomized start positions)
- Arrow shows actual movement direction
- Agent properly STOPS at obstacles (cannot pass through)

## 🚀 How to Use

### Training
```bash
python train_corridor_env.py
```

### Testing & Video Generation
```bash
python test_and_visualize.py
```

### Load Model and Test Manually
```python
from stable_baselines3 import PPO
model = PPO.load("trained_model/final_model")
```

## ✨ Key Features

1. **Realistic Physics**: Agent cannot pass through obstacles
2. **Raycast Perception**: Like static environment, uses 12 raycasts
3. **Collision Handling**: Penalty-based (not terminal) - agent learns avoidance
4. **100% Success**: Agent reliably navigates around moving obstacles
5. **Efficient Navigation**: Minimal collisions, fast goal reaching

## 📝 Notes

- Collision detection checks BEFORE position update
- Agent stops when collision detected (realistic physics)
- Obstacles move independently with different speeds
- Environment resets with randomized obstacle positions
- Videos show proper arrow rotation based on movement direction

## 🎓 Training Strategy

Simple single-stage training (no curriculum):
- Train directly on full environment
- Strong penalty for collisions (-20)
- High reward for success (+300)
- Agent learns to navigate efficiently within 1M steps

Previous failed attempts with progressive curriculum (Stage 1-3) showed misleading metrics. Direct training proved more reliable.
