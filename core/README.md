# Core Module - Shared Neural Network Components

This module contains the shared deep learning architecture used by the DRL agent for VGA dataset navigation training.

---

## 📋 Contents

### 1. `advanced_policy_network.py`

**Purpose:** Custom Actor-Critic policy network for PPO training

**Architecture:**
```python
Input: 41-dimensional observation
    ↓
Shared Feature Extractor:
    Dense(256) + ReLU
    Dense(256) + ReLU  
    Dense(128) + ReLU
    ↓
┌─────────────────────┬─────────────────────┐
│   Policy Head       │    Value Head       │
│   (Actor)           │    (Critic)         │
│   Dense(64) + ReLU  │   Dense(64) + ReLU  │
│   Dense(2) [mean]   │   Dense(1) [value]  │
│   log_std param     │                     │
└─────────────────────┴─────────────────────┘
         ↓                      ↓
  Action Distribution      State Value V(s)
   ~ N(mean, std)
```

**Features:**
- Shared feature extractor (efficient)
- Separate policy and value heads (stable learning)
- Gaussian policy for continuous actions
- ~250K trainable parameters

**Usage:**
```python
from advanced_policy_network import AdvancedActorCriticPolicy
from stable_baselines3 import PPO

model = PPO(
    AdvancedActorCriticPolicy,
    env,
    verbose=1
)
```

**Used by:**
- `drl_vga_experiments/train_vga_drl.py` - Main training script

---

### 2. `numpy_compat_fix.py`

**Purpose:** NumPy 2.x compatibility fixes for Stable-Baselines3

**Issue Resolved:**
- Stable-Baselines3 uses deprecated NumPy aliases (np.float, np.int)
- NumPy 2.0+ removed these aliases
- This module patches the compatibility issues

**Usage:**
```python
# Import at the start of training scripts
from numpy_compat_fix import *
```

**Fixes:**
- `np.float` → `np.float64`
- `np.int` → `np.int64`
- `np.bool` → `np.bool_`

---

## 🔗 Integration with Project

### DRL Training (`drl_vga_experiments/`)

The DRL training script imports from this core module:

```python
# drl_vga_experiments/train_vga_drl.py
import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

# Import policy
from advanced_policy_network import AdvancedActorCriticPolicy
from numpy_compat_fix import *
```

### Why Shared Module?

1. **Reusability** - Same policy can be used for different training experiments
2. **Consistency** - Ensures identical architecture across different runs
3. **Maintainability** - Single place to update network architecture
4. **Clean Separation** - Core ML code separate from experiment-specific code

---

## 📊 Policy Network Details

### Observation Space

**Input:** 41-dimensional continuous vector

```python
observation = [
    # 36 ray-casting distances (10° resolution, 360° coverage)
    ray_0, ray_1, ..., ray_35,     # Each: [0, 8.0] meters
    
    # Goal information (relative to agent)
    goal_distance,                  # [0, ~14] meters
    goal_angle,                     # [-π, +π] radians
    
    # Agent state
    velocity_x,                     # [-1.6, +1.6] m/s
    velocity_y,                     # [-1.6, +1.6] m/s
    heading_angle                   # [-π, +π] radians
]
```

**Normalization:**
- All observations normalized by VecNormalize during training
- Running mean & variance computed online
- Normalization stats saved with model

### Action Space

**Output:** 2-dimensional continuous Gaussian distribution

```python
action = [acceleration_x, acceleration_y]
Range: [-1.0, +1.0] (normalized)
Scaled: [-2.5, +2.5] m/s² (MAX_ACCELERATION)
```

**Policy:**
- Mean: Output of policy head (2D vector)
- Std: Learned log_std parameter (shared across dimensions)
- Action sampled from: N(mean, exp(log_std))

---

## 🎓 Training Configuration

### Hyperparameters (used in drl_vga_experiments/)

```python
LEARNING_RATE = 3e-4          # Adaptive (linear decay)
N_STEPS = 2048                # Per environment per update
BATCH_SIZE = 64               # Minibatch size
N_EPOCHS = 10                 # Optimization epochs per update
GAMMA = 0.99                  # Discount factor
GAE_LAMBDA = 0.95             # Advantage estimation
CLIP_RANGE = 0.2              # PPO clipping
ENT_COEF = 0.01               # Entropy (exploration)
VF_COEF = 0.5                 # Value function coefficient
MAX_GRAD_NORM = 0.5           # Gradient clipping
```

### Training Infrastructure

```python
N_ENVS = 8                    # Parallel environments
VEC_NORMALIZE = True          # Observation/reward normalization
TOTAL_TIMESTEPS = 2_000_000   # Training budget
CURRICULUM_STAGES = 6         # Progressive difficulty
```

---

## 🧪 Performance

### Training Results (VGA Dataset)

**Final Model:**
- Trained timesteps: 2,000,000
- Training time: ~2 hours (RTX 3050)
- Success rate: 98.9% on test set

**Per-Scenario Performance:**
- SOSP (1 obstacle): 100.0%
- MOSP_A (4 obstacles): 100.0%
- MOSP_B (7 obstacles): 98.9%
- MOSP_C (12 obstacles): 100.0%
- MOSP_D (16 obstacles): 99.3%

---

## 📦 Dependencies

```python
torch >= 2.0            # Neural network backend
numpy >= 1.21           # Numerical computations
stable-baselines3 >= 2.0  # RL training framework
gymnasium >= 0.28       # Environment interface
```

---

## 🔧 Customization

### Modifying Network Architecture

Edit `advanced_policy_network.py`:

```python
# Change hidden layer sizes
class AdvancedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            net_arch=[
                dict(pi=[256, 256, 128],    # Policy layers
                     vf=[256, 256, 128])    # Value layers
            ]
        )
```

### Adding New Features

1. Modify observation space in `vga_experimental_env.py`
2. Update network input dimension in policy
3. Retrain from scratch with new architecture

---

## 📁 File Details

```
core/
├── advanced_policy_network.py      # Custom Actor-Critic policy
│   - AdvancedActorCriticPolicy class
│   - Network architecture definition
│   - ~250K parameters
│
├── numpy_compat_fix.py             # NumPy 2.x compatibility
│   - Patches deprecated NumPy aliases
│   - No-op for NumPy < 2.0
│
└── README.md                       # This file
```

---

## 🚀 Quick Reference

### Import Policy in Training Script

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from advanced_policy_network import AdvancedActorCriticPolicy
from numpy_compat_fix import *
```

### Use Policy with PPO

```python
from stable_baselines3 import PPO

model = PPO(
    AdvancedActorCriticPolicy,
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    verbose=1
)
```

---

## 📚 Additional Documentation

- **DRL Training:** See `drl_vga_experiments/README.md`
- **Environment:** See `drl_vga_experiments/vga_experimental_env.py`
- **Full Project:** See root `README.md`

---

**Last Updated:** December 21, 2025  
**Version:** 2.0  
**Status:** ✅ Stable - Used in production DRL training
