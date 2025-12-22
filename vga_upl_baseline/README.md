# VGA+UPL Baseline Implementation

**Variable Goal Approach (VGA) + Universal Power Law (UPL) for Pedestrian Navigation**

---

## 🎯 Overview

This implementation achieves **100% success rate** on all VGA experimental scenarios (941 trials total).

### Results Summary

| Scenario | Trials | Success Rate | Mode |
|----------|--------|--------------|------|
| SOSP | 54 | **100%** | Deterministic |
| MOSP_A | 239 | **100%** | Deterministic |
| MOSP_B | 188 | **100%** | Deterministic |
| MOSP_C | 184 | **100%** | Deterministic |
| MOSP_D | 276 | **100%** | Deterministic |
| **Total** | **941** | **100%** | |

---

## 📁 Project Structure

```
vga_upl_baseline/
├── models/
│   ├── vga_upl_planner_v4.py       # Main VGA+UPL planner (det & stoch)
│   ├── upl_physics.py              # Universal Power Law physics engine
│   ├── model_base.py               # Base class for navigation models
│   └── __init__.py
│
├── scripts/
│   ├── generate_v4_visualizations.py      # Deterministic visualizations
│   └── generate_stochastic_visualizations.py  # Stochastic visualizations
│
├── data_loading/
│   ├── vga_dataset.py              # VGA dataset loader
│   ├── bottleneck_dataset.py       # Bottleneck scenario loader
│   └── utils.py
│
├── results/
│   ├── vga_v4_Det/                 # Deterministic results
│   │   ├── videos/                 # 50 navigation videos
│   │   ├── images/                 # 50 trajectory images
│   │   ├── evaluation_metrics.json # Per-trial metrics (941 trials)
│   │   └── aggregate_metrics.json  # Summary statistics
│   │
│   └── vga_v4_Stochastic/          # Stochastic results
│       ├── summary_images/         # Paper-style path distributions
│       ├── videos/                 # Multi-path overlay videos
│       └── stochastic_results.json
│
├── archive/                        # Deprecated files (reference only)
│
├── VGA_UPL_ALGORITHM.md           # 📖 DEEP algorithm documentation
├── COMMANDS.txt                   # 📋 All commands to run
├── README.md                      # This file
└── experimental_data_config.yaml  # Data paths configuration
```

---

## 🚀 Quick Start

### 1. Generate Deterministic Results

```bash
cd vga_upl_baseline
python scripts/generate_v4_visualizations.py
```

**Outputs:**
- 10 videos per scenario (50 total)
- 10 images per scenario (50 total)
- `evaluation_metrics.json` - Metrics for ALL 941 trials
- `aggregate_metrics.json` - Summary statistics

### 2. Generate Stochastic Visualizations

```bash
python scripts/generate_stochastic_visualizations.py
```

**Outputs:**
- Paper-style summary images (path distributions with percentages)
- Multi-path overlay videos
- `stochastic_results.json` - Path clustering analysis

---

## ⚙️ Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `agent_radius` | 0.2m | Robot body radius |
| `min_clearance` | 0.02m | Safety margin from obstacles |
| `desired_speed` | 1.34m/s | Target walking speed (Weidmann) |
| `max_steps` | 2000 | Maximum simulation steps |
| `dt` | 0.05s | Time step |

---

## 🔄 Operating Modes

### Deterministic Mode (`use_probabilistic=False`)

- **Behavior:** Always selects the minimum-deviation path around obstacles
- **Reproducibility:** 100% reproducible (same input → same output)
- **Use Case:** Evaluation, benchmarking, consistent results

```python
from models.vga_upl_planner_v4 import VGAUPLPlannerV4

planner = VGAUPLPlannerV4(use_probabilistic=False)
result = planner.simulate(start, goal, obstacles=obstacles)
```

### Stochastic Mode (`use_probabilistic=True`)

- **Behavior:** Boltzmann selection for path variety
- **Path Variety:** Multiple distinct routes around obstacles
- **Use Case:** Human-like behavior simulation, path diversity analysis

```python
planner = VGAUPLPlannerV4(use_probabilistic=True)
# Each run may produce different paths
```

---

## 📊 Metrics Computed

The planner computes comprehensive metrics for each trial:

### Navigation Metrics
- `success` - Goal reached within threshold
- `travel_time` - Total navigation time
- `path_length` - Actual distance traveled
- `path_efficiency` - Optimal / Actual path length

### Safety Metrics
- `num_collisions` - Collision count
- `min_clearance` - Minimum distance to obstacles
- `danger_zone_ratio` - Time spent in danger zone

### Smoothness Metrics
- `average_speed`, `speed_variance`
- `average_acceleration`, `max_acceleration`
- `average_jerk`, `max_jerk`
- `direction_changes`, `oscillation_index`

### VGA-Specific Metrics
- `num_subgoals` - Subgoals used during navigation
- `subgoal_switch_rate` - Subgoal changes per second
- `subgoal_history` - Full subgoal sequence

---

## 📖 Algorithm Documentation

For a **deep dive** into how VGA+UPL works (both deterministic and stochastic modes), see:

📄 **[VGA_UPL_ALGORITHM.md](VGA_UPL_ALGORITHM.md)**

This includes:
- Mathematical foundations
- Geometric subgoal computation
- Stochastic decision making
- Code walkthrough with line references

---

## 📋 Commands Reference

See **[COMMANDS.txt](COMMANDS.txt)** for all available commands.

---

## 📂 Data Requirements

VGA experimental data is included in the project:
```
../data/VGA-Experimental-Data/
```

**Required Files:**
- `SOSP_initialFinalPos_feed.txt`
- `SOSP_obstPos_feed.txt`
- `MOSP_CaseA_initialFinalPos_feed.txt`
- `MOSP_CaseA_obstPos_feed.txt`
- (Same for MOSP_B, MOSP_C, MOSP_D)

---

## 🔬 Comparison with DRL

| Aspect | VGA+UPL | DRL Agent |
|--------|---------|-----------|
| Success Rate | **100%** | 98.9% |
| Training Required | No | Yes (2M steps) |
| Interpretability | High | Low |
| Path Variety | Stochastic mode | Policy variance |
| Computation | ~0.01s/trial | ~0.05s/trial |

---

**Last Updated:** December 22, 2025  
**Version:** 4.0  
**Status:** ✅ Production Ready - 100% Success
