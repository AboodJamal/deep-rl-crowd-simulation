# Validation Framework

This directory contains the scientific validation framework for comparing the DRL pedestrian navigation model against:
1. Real human experimental data (VGA dataset + Bottleneck dataset)
2. VGA algorithm (Variable Goal Approach)
3. Classical baseline (Social Force / JuPedSim-like model)

## Directory Structure

```
validation/
├── data_loading/          # Parsers for experimental datasets
│   ├── vga_dataset.py     # VGA GitHub dataset loader (Head-On, SOSP, MOSP)
│   ├── bottleneck_dataset.py  # Jülich bottleneck dataset loader
│   └── utils.py           # Common data processing utilities
│
├── models/                # Model interfaces and wrappers
│   ├── drl_policy_interface.py    # Wrapper for trained DRL agent
│   ├── vga_planner.py             # VGA algorithm implementation
│   ├── social_force_baseline.py   # Classical Social Force model
│   └── model_base.py              # Abstract base class for models
│
├── experiments/           # Validation experiment scripts
│   ├── head_on_validation.py      # Head-On scenario experiments
│   ├── bottleneck_validation.py   # Bottleneck scenario experiments
│   ├── sosp_validation.py         # Single Obstacle Single Pedestrian
│   ├── mosp_validation.py         # Multiple Obstacles Single Pedestrian
│   └── experiment_runner.py       # Common experiment execution logic
│
├── metrics/               # Trajectory analysis and metrics
│   ├── trajectory_metrics.py      # Core metrics (smoothness, speed, etc.)
│   ├── comparison_metrics.py      # Model vs human comparison metrics
│   └── statistical_analysis.py    # Statistical tests and distributions
│
├── plots/                 # Visualization utilities
│   ├── trajectory_plots.py        # Trajectory overlay plots
│   ├── velocity_profiles.py       # Velocity vs position plots (VGA Fig 2)
│   ├── stochastic_paths.py        # Path distribution plots (VGA Fig 7)
│   └── publication_style.py       # Matplotlib style configuration
│
├── notebooks/             # Interactive analysis notebooks
│   ├── 01_data_exploration.ipynb       # Explore VGA and Bottleneck datasets
│   ├── 02_head_on_analysis.ipynb       # Head-On validation results
│   ├── 03_bottleneck_analysis.ipynb    # Bottleneck validation results
│   ├── 04_sosp_mosp_analysis.ipynb     # SOSP/MOSP validation results
│   └── 05_comprehensive_comparison.ipynb  # Final results summary
│
└── README.md              # This file
```

## Datasets

### 1. VGA GitHub Dataset
**Source:** https://github.com/kanika201293/Pedestrian-Experimental-Data

**Scenarios:**
- Head-On encounters
- Single Obstacle Single Pedestrian (SOSP)
- Multiple Obstacles Single Pedestrian (MOSP)
- Parallel Pedestrian (overtaking)

**Data Format:**
- Initial/final position files: `*_initialFinalPos_feed.txt`
- Full trajectory files: Available in repository
- Metadata: Scenario type, case ID, trial information

### 2. Bottleneck Individuals Dataset (Jülich)
**Source:** https://ped.fz-juelich.de/da/doku.php?id=bottleneck_individuals

**Paper:** Boomers et al., 2024

**Experiment Details:**
- One pedestrian at a time through bottleneck
- Bottleneck structure: 4m × 2m × 1m
- Variables:
  - Length: 0.2m, 1.0m, 2.0m
  - Width: 0.4m, 0.5m, 0.6m, 0.7m, 0.8m, 1.0m
  - Approach angle: -90°, -60°, -30°, 0°, 30°, 60°, 90°
  - Motivation: normal vs hurry
  - Goal location: straight, 30° left, 30° right

**Data Format:**
- Metadata: JSON files
- Trajectories: HDF5 and TXT formats
- Videos and MoCap data available

## Metrics

### Path Quality Metrics
1. **Path Smoothness**: Integral of squared curvature
2. **Speed Deviation**: Standard deviation normalized by mean
3. **Oscillation**: Measure of direction changes and backwards movement
4. **Travel Time**: Time from start to goal

### Comparison Metrics
1. **Trajectory Deviation**: Mean Euclidean distance from human path
2. **Velocity Profile**: V/V_des vs position (VGA Figure 2 style)
3. **Collision Rate**: Overlap with obstacles or other agents
4. **Stuck Rate**: Fraction of failed runs

### Stochastic Analysis
1. **Path Distribution**: Probability of different path choices (VGA Figure 7 style)
2. **Variance Analysis**: Spread in trajectories from same initial conditions

## Usage

### 1. Download Datasets
```bash
# Download VGA dataset
cd validation/data_loading
# TODO: Add download instructions

# Download Bottleneck dataset
# TODO: Add download instructions
```

### 2. Run Validation Experiments
```bash
# Head-On validation
python validation/experiments/head_on_validation.py --num-runs 100 --output results/head_on

# Bottleneck validation
python validation/experiments/bottleneck_validation.py --num-runs 50 --output results/bottleneck
```

### 3. Generate Plots
```bash
# Generate all comparison plots
python validation/plots/trajectory_plots.py --data results/head_on
python validation/plots/velocity_profiles.py --data results/bottleneck
```

### 4. Interactive Analysis
```bash
# Launch Jupyter
jupyter notebook validation/notebooks/
```

## Requirements

```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scipy>=1.10.0
h5py>=3.8.0
gymnasium>=0.28.0
stable-baselines3>=2.0.0
torch>=2.0.0
scikit-learn>=1.3.0
```

## Validation Goals

The validation framework aims to answer:

1. **How does DRL compare to classical models?**
   - VGA (geometric planning)
   - Social Force (physics-based)

2. **How realistic is DRL compared to human behavior?**
   - Trajectory similarity
   - Velocity profiles
   - Path choice distributions

3. **What are the strengths and limitations?**
   - Success rates across scenarios
   - Edge cases and failure modes
   - Computational efficiency

## Publication-Ready Outputs

All plots are configured for publication quality:
- Vector graphics (PDF/SVG)
- LaTeX-style labels
- Consistent color schemes
- Clear legends and annotations
- Matching VGA paper figure styles

## References

1. **VGA Paper**: "Variable Goal Approach (VGA): Incorporating Human Intelligence into Microscopic Pedestrian Dynamics Models"
   - arXiv:2501.05100v2

2. **Bottleneck Paper**: Boomers et al., 2024, "How Approaching Angle, Bottleneck Width and Walking Speed Affect the Use of a Bottleneck by Individuals"

3. **Social Force Model**: Helbing & Molnár, 1995

4. **JuPedSim**: https://www.jupedsim.org/
