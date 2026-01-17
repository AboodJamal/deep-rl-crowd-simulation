# VGA Dataset Pedestrian Navigation Study

**Deep Reinforcement Learning vs Physics-Based Navigation on Real Human Data**

---

## Project Overview

This project implements and compares two fundamentally different approaches to autonomous pedestrian navigation using the **VGA (Virtual Guidance Assistance) experimental dataset**:

| Approach | Method | Success Rate | Nature |
|----------|--------|--------------|--------|
| **DRL Agent** | Deep Reinforcement Learning (PPO) | 98.9% | Data-driven, learned |
| **VGA+UPL** | Variable Goal Approach + Universal Power Law | 100% | Physics-based, geometric |

**Dataset:** 941 real human pedestrian navigation trials from the [VGA Experimental Dataset](https://github.com/kanika201293/Pedestrian-Experimental-Data)

---

## Project Structure

```
project/
│
├── drl_training/                   # Deep Reinforcement Learning Agent
│   ├── train_vga_drl.py            # PPO curriculum training
│   ├── evaluate_vga_drl.py         # Evaluation & visualization generator
│   ├── vga_experimental_env.py     # Gymnasium environment
│   ├── models/                     # Trained DRL models
│   │   ├── vga_drl_final.zip      # Final trained policy (98.9% success)
│   │   └── vga_drl_stage6_vecnormalize.pkl  # CRITICAL normalization stats
│   ├── evaluation_final/           # Full evaluation results
│   └── README.md                  # DRL-specific documentation
│
├── vga_baseline/                   # VGA + UPL Physics-Based Algorithm
│   ├── models/
│   │   ├── vga_upl_planner_v4.py  # VGA algorithm (100% success)
│   │   └── upl_physics.py         # Universal Power Law physics
│   ├── scripts/
│   │   ├── generate_v4_visualizations.py      # Deterministic visualizations
│   │   └── generate_stochastic_visualizations.py  # Stochastic visualizations
│   ├── results/
│   │   ├── vga_v4_Det/            # Deterministic results (videos, images)
│   │   └── vga_v4_Stochastic/     # Stochastic results (path distributions)
│   ├── VGA_UPL_ALGORITHM.md       # Algorithm documentation
│   └── README.md                  # VGA+UPL documentation
│
├── comparisons/                    # All comparison results
│   ├── static_obstacles/          # DRL vs VGA static comparison
│   ├── dynamic_obstacles/         # Dynamic obstacles comparison
│   ├── narrow_passages/           # Narrow passage navigation
│   ├── multiagent/                # Multi-agent scenarios
│   └── real_experiment/           # Real experiment comparison
│
├── data/                          # VGA Experimental Dataset
│   ├── VGA-Experimental-Data/     # Original dataset (941 trials)
│   ├── DATASET_DOCUMENTATION.md   # Comprehensive dataset documentation
│   └── scenario_visualizations/   # Obstacle layout visualizations
│
├── docs/                          # Presentation materials
│   └── presentation_drl_vs_vga.md # Marp presentation slides
│
├── core/                          # Shared utilities
│   ├── advanced_policy_network.py # Actor-Critic policy (for DRL)
│   └── numpy_compat_fix.py        # NumPy compatibility
│
├── README.md                       # This file
└── requirements.txt               # Python dependencies
```

---

## Experimental Scenarios

Both methods are tested on identical VGA dataset scenarios:

| Scenario | Obstacles | Trials | Description | Difficulty |
|----------|-----------|--------|-------------|------------|
| **SOSP** | 1 | 54 | Single Obstacle Single Pedestrian | Easy |
| **MOSP_A** | 4 | 239 | Low density (sparse) | Easy |
| **MOSP_B** | 7 | 188 | Medium density (TIGHT spacing) | Hard |
| **MOSP_C** | 12 | 184 | High density | Medium |
| **MOSP_D** | 16 | 276 | Very high density | Hard |

**Arena Dimensions:** 10m × 3.5m (matches real VGA experiments)  
**Agent Radius:** 0.2m  
**Obstacle Radius:** 0.25m

---

## Results Comparison

### DRL Agent Performance

```
Method: Proximal Policy Optimization (PPO)
Training: 6-stage curriculum, 2M timesteps

Overall Success Rate: 98.9% (932/941)
├── SOSP:    100.0% (54/54)
├── MOSP_A:  100.0% (239/239)
├── MOSP_B:   98.9% (186/188)  ← Hardest (tight spacing)
├── MOSP_C:  100.0% (184/184)
└── MOSP_D:   99.3% (274/276)
```

### VGA+UPL Performance

```
Method: Variable Goal Approach + Universal Power Law
Mode: Deterministic (+ Stochastic option for path variety)

Overall Success Rate: 100% (941/941)
├── SOSP:    100.0% (54/54)
├── MOSP_A:  100.0% (239/239)
├── MOSP_B:  100.0% (188/188)  ← Perfect!
├── MOSP_C:  100.0% (184/184)
└── MOSP_D:  100.0% (276/276)
```

---

## Quick Start

### Option 1: Run DRL Evaluation

```bash
cd drl_training/

# Evaluate pre-trained model (MUST include --vec-normalize)
python evaluate_vga_drl.py \
    --model models/vga_drl_final.zip \
    --vec-normalize models/vga_drl_stage6_vecnormalize.pkl \
    --output-dir evaluation_test/
```

### Option 2: Run VGA+UPL

```bash
cd vga_baseline/

# Generate deterministic results (100% success)
python scripts/generate_v4_visualizations.py

# Generate stochastic visualizations (paper-style paths)
python scripts/generate_stochastic_visualizations.py
```

---

## Key Technical Differences

| Aspect | DRL Agent | VGA+UPL |
|--------|-----------|---------|
| **Decision Making** | Neural network inference | Geometric subgoal computation |
| **Training Required** | Yes (2M timesteps, ~2 hours) | No (analytical) |
| **Interpretability** | Black box | Fully interpretable |
| **Tuning** | Hyperparameters + reward | Physics parameters |
| **Path Variety** | Stochastic policy | Stochastic mode option |
| **Generalization** | Within training distribution | Physics-based principles |

---

## Documentation

| Document | Description |
|----------|-------------|
| [drl_training/README.md](drl_training/README.md) | DRL training & evaluation guide |
| [vga_baseline/README.md](vga_baseline/README.md) | VGA+UPL usage guide |
| [vga_baseline/VGA_UPL_ALGORITHM.md](vga_baseline/VGA_UPL_ALGORITHM.md) | Deep dive into VGA+UPL algorithm |
| [data/DATASET_DOCUMENTATION.md](data/DATASET_DOCUMENTATION.md) | Comprehensive dataset documentation |
| [comparisons/static_obstacles/](comparisons/static_obstacles/) | DRL vs VGA static comparison results |
| [docs/presentation_drl_vs_vga.md](docs/presentation_drl_vs_vga.md) | Presentation slides (Marp format) |

---

## Installation

```bash
# Clone or download the repository
cd "trains on everything- eval on everything"

# Install dependencies
pip install -r requirements.txt

# Required packages:
# - gymnasium >= 0.28
# - stable-baselines3 >= 2.0
# - torch >= 2.0
# - numpy, pandas, matplotlib, opencv-python
```

---

## Dataset

VGA experimental data is included in the project:
```
data/VGA-Experimental-Data/
```

This is the default location. You can also modify `data_root` parameter in scripts if needed.

**Dataset Files:**
- `SOSP_initialFinalPos_feed.txt` / `SOSP_obstPos_feed.txt`
- `MOSP_CaseA/B/C/D_initialFinalPos_feed.txt`
- `MOSP_CaseA/B/C/D_obstPos_feed.txt`

---

## Hardware Used

- **GPU:** NVIDIA RTX 3050 (4GB VRAM)
- **CPU:** Multi-core processor
- **RAM:** 16GB
- **OS:** Windows 11

---

## Key Findings

### DRL Strengths
- Learns directly from data (no physics modeling)
- Fast inference (real-time capable)
- Generalizes across density levels
- Can capture subtle behavioral patterns

### VGA+UPL Strengths
- 100% success rate (higher than DRL)
- Fully interpretable (geometric reasoning)
- No training required
- Consistent, reproducible results
- Stochastic mode shows human-like path variety

### Trade-offs
- DRL requires significant training compute
- VGA+UPL requires careful parameter tuning
- DRL is a "black box" (less interpretable)
- VGA+UPL assumes accurate physics model

---

## Attribution

**VGA Experimental Dataset:**
- Source: https://github.com/kanika201293/Pedestrian-Experimental-Data
- Paper: "Virtual Guidance Assistance in Crowd Navigation"

---

**Last Updated:** January 18, 2026  
**Version:** 4.0  
**Status:** Production Ready
