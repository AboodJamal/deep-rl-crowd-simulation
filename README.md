# VGA Dataset Pedestrian Navigation Study

**Deep Reinforcement Learning vs Physics-Based Navigation on Real Human Data**

---

## Project Overview

This project implements and compares two fundamentally different approaches to autonomous pedestrian navigation using the **VGA (Virtual Guidance Assistance) experimental dataset**:

| Approach            | Method                                       | Success Rate | Nature                   |
| ------------------- | -------------------------------------------- | ------------ | ------------------------ |
| **DRL Agent** | Deep Reinforcement Learning (PPO)            | 100%         | Data-driven, learned     |
| **VGA+UPL**   | Variable Goal Approach + Universal Power Law | 100%         | Physics-based, geometric |

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
│   │   ├── vga_drl_final.zip      # Final trained policy (100% success)
│   │   └── vga_drl_stage6_vecnormalize.pkl  # CRITICAL normalization stats
│   ├── evaluation_final/           # Full evaluation results
│   └── README.md                  # DRL-specific documentation
│
├── vga_baseline/                   # VGA + UPL Physics-Based Algorithm
│   ├── models/
│   │   ├── vga_upl_planner_v4.py  # VGA algorithm (100% success)
│   │   └── upl_physics.py         # Universal Power Law physics
│   ├── 2501.05100v2.pdf           # VGA+UPL Paper (arXiv:2501.05100)
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

| Scenario         | Obstacles | Trials | Description                       | Difficulty |
| ---------------- | --------- | ------ | --------------------------------- | ---------- |
| **SOSP**   | 1         | 54     | Single Obstacle Single Pedestrian | Easy       |
| **MOSP_A** | 4         | 239    | Low density (sparse)              | Easy       |
| **MOSP_B** | 7         | 188    | Medium density (TIGHT spacing)    | Hard       |
| **MOSP_C** | 12        | 184    | High density                      | Medium     |
| **MOSP_D** | 16        | 276    | Very high density                 | Hard       |

**Arena Dimensions:** 10m × 3.5m (matches real VGA experiments)
**Agent Radius:** 0.2m
**Obstacle Radius:** 0.25m

---

## Results Comparison

### DRL Agent Performance

```
Method: Proximal Policy Optimization (PPO)
Training: 6-stage curriculum, 2M timesteps

Overall Success Rate: 100% (941/941)
├── SOSP:    100.0% (54/54)
├── MOSP_A:  100.0% (239/239)
├── MOSP_B:  100.0% (188/188)
├── MOSP_C:  100.0% (184/184)
└── MOSP_D:  100.0% (276/276)
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

| Aspect                      | DRL Agent                    | VGA+UPL                       |
| --------------------------- | ---------------------------- | ----------------------------- |
| **Decision Making**   | Neural network inference     | Geometric subgoal computation |
| **Training Required** | Yes (2M timesteps, ~2 hours) | No (analytical)               |
| **Interpretability**  | Black box                    | Fully interpretable           |
| **Tuning**            | Hyperparameters + reward     | Physics parameters            |
| **Path Variety**      | Stochastic policy            | Stochastic mode option        |
| **Generalization**    | Within training distribution | Physics-based principles      |

---

## Documentation

| Document                                                            | Description                          |
| ------------------------------------------------------------------- | ------------------------------------ |
| [drl_training/README.md](drl_training/README.md)                       | DRL training & evaluation guide      |
| [vga_baseline/README.md](vga_baseline/README.md)                       | VGA+UPL usage guide                  |
| [vga_baseline/VGA_UPL_ALGORITHM.md](vga_baseline/VGA_UPL_ALGORITHM.md) | Deep dive into VGA+UPL algorithm     |
| [data/DATASET_DOCUMENTATION.md](data/DATASET_DOCUMENTATION.md)         | Comprehensive dataset documentation  |
| [comparisons/static_obstacles/](comparisons/static_obstacles/)         | DRL vs VGA static comparison results |
| [docs/presentation_drl_vs_vga.md](docs/presentation_drl_vs_vga.md)     | Presentation slides (Marp format)    |

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

- 100% success rate
- Learns directly from data (no physics modeling)
- Fast inference (real-time capable)
- Generalizes across density levels
- Can capture subtle behavioral patterns

### VGA+UPL Strengths

- 100% success rate
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

## Multi-Agent Experiments

A dedicated branch [`drl-multiagents`](https://github.com/AboodJamal/deep-rl-crowd-simulation/tree/drl-multiagents) extends this project to multi-agent DRL navigation. See that branch for code, results, and a Quick Start guide.

**References for metrics and evaluation can be found at the end of `docs/PRESENTATION_GUIDE.md`.**

---

## Metrics Used
All results are averaged over 30 trials per scenario, for a total of 150 trials.

| **Metric**                        | **Definition**                                                                                                                     |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Success Rate (↑)**               | Fraction of trials where the agent reaches the goal without collision. Higher is better.                                           |
| **SPL (↑)**                        | Success weighted by path length: combines success and path efficiency. Higher is better.                                          |
| **Travel Time (s) (↓)**            | Time taken to reach the goal. Lower is better.                                                                                     |
| **Collision Rate (↓)**             | Fraction of trials with any collision. Lower is better.                                                                            |
| **Average Jerk (m/s³) (↓)**        | Smoothness of motion: lower jerk means smoother, more natural movement. Lower is better.                                          |
| **Minimum Clearance (m) (↑)**      | Closest distance to any obstacle during navigation. Higher is safer.                                                              |
| **Oscillation Index (↓)**          | Measures unnecessary zigzagging or back-and-forth motion. Lower is better.                                                        |

---

| **Metric**                        | **VGA + UPL** | **DRL (PPO)** |
| ---------------------------------- | ------------ | ------------- |
| **Success Rate (↑)**               | **100.0%**   | **100.0%**    |
| **SPL (↑)**                        | **1.00**     | 0.949         |
| **Travel Time (s) (↓)**            | 6.46         | **6.36**      |
| **Collision Rate (↓)**             | 2.0%         | **0.0%**      |
| **Average Jerk (m/s³) (↓)**        | **21.20**    | 96.17         |
| **Minimum Clearance (m) (↑)**      | **0.277**    | **0.277**     |
| **Oscillation Index (↓)**          | **4.01**     | 20.58         |

_Bolded values indicate the best (most desirable) result for each metric._

---

| **Metric**                               | **Definition**                                                                                                                     |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Generalization / Adaptability**        | Ability of the agent to handle unseen environments, layouts, or configurations beyond those encountered during training.           |
| **Dynamic Obstacle Handling**            | Capability to react safely and effectively to moving obstacles and changing environments in real time.                             |
| **Multi-Agent Scalability**              | Ability to coordinate, avoid deadlocks, and operate robustly when multiple agents interact in shared space.                        |
| **Natural Motion**                       | Degree to which the generated trajectories resemble smooth, continuous, human-like movement rather than sharp or artificial paths. |
| **Trajectory Diversity & Stochasticity** | Ability to produce varied paths under identical conditions, reflecting natural variability in human decision-making.               |
| **Legibility**                           | How clearly the trajectory communicates the agent’s intent (goal and direction) to observers or other agents.                      |
| **Hesitation Behavior**                  | How decisively the agent commits to a path near obstacles, avoiding unnecessary stops or oscillations.                             |
| **Comfort / Personal Space**             | Extent to which the agent respects comfortable distances from obstacles and others, aligning with human comfort zones.             |

---

| #     | Dimension                                | VGA + UPL                 | DRL (PPO)    | Winner  | Reason                                                                                                                     |
| ----- | ---------------------------------------- | ------------------------- | ------------ | ------- | -------------------------------------------------------------------------------------------------------------------------- |
| **1** | **Generalization / Adaptability**        | ⚠️ Limited                | ✅ Excellent  | **DRL** | VGA has no learning and struggles in narrow corridors or trapped spaces. DRL learns and adapts, similar to human behavior. |
| **2** | **Dynamic Obstacle Handling**            | ❌ Poor                    | ✅ Excellent  | **DRL** | VGA assumes a static world and fails with moving obstacles. DRL adapts in real time.                                       |
| **3** | **Multi-Agent Scalability**              | ❌ Failed                  | ✅ Good       | **DRL** | VGA deadlocks with other agents. DRL exhibits coordination and partial cooperation.                                        |
| **4** | **Natural Motion**                       | ✅ OK (Not very realistic) | ✅ Human-like | **DRL** | VGA produces sharp waypoint turns. DRL generates smooth, continuous trajectories.                                          |
| **5** | **Trajectory Diversity & Stochasticity** | ❌ None                    | ✅ High       | **DRL** | VGA is deterministic (same path every run). DRL is stochastic, producing varied, human-like paths.                         |
| **6** | **Legibility**                           | ✅ Good                    | ✅ Good       | **TIE** | Both methods produce trajectories that clearly communicate intent.                                                         |
| **7** | **Hesitation Behavior**                  | ✅ Good                    | ✅ Decisive   | **TIE** | VGA behaves cautiously near obstacles. DRL commits confidently to a path.                                                  |
| **8** | **Comfort / Personal Space**             | ✅ Good                    | ✅ Good       | **TIE** | Both respect obstacle clearances; DRL often maintains slightly smoother margins.                                           |

---

## Summary & Discussion

This work investigated whether Deep Reinforcement Learning (DRL) can produce more realistic pedestrian navigation compared to a classical model-based approach (VGA + UPL).

**VGA+UPL excels in:**
- Path efficiency (SPL)
- Smooth motion (low jerk, low oscillation)
- Predictable, rule-consistent behaviour

**DRL (PPO) excels in:**
- Adaptability & generalization
- Dynamic obstacle handling
- Multi-agent coordination
- Natural variability and human-like motion

Quantitative metrics alone are insufficient to capture navigation realism.
Qualitative evaluation reveals critical differences in natural motion, adaptability, stochasticity, and coordination.
As a secondary goal, this work extended the DRL framework to multi-agent systems.

**Limitations:**
- VGA+UPL required full reimplementation due to missing code and the lack of trajectories data was challenging to find strong metrics.
- DRL training, especially in multi-agent settings, is computationally expensive and sensitive to reward design

**Future work:**
- Learn reward functions directly from human trajectory data
- Explore hybrid models that combine DRL adaptability with VGA interpretability
- Improve perception models and enhance scalable multi-agent coordination
