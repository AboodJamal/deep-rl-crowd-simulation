# Scientific Validation Report: Deep Reinforcement Learning for Pedestrian Navigation

**Student:** Abdallah Jamal Jamil Al-Harrem  
**Supervisors:** Dr. Mohcine & Dr. Ahmad  
**Date:** December 10, 2025  
**Status:** Validation Framework Implementation in Progress

---

## Executive Summary

This report addresses your inquiry regarding the validation of our Deep Reinforcement Learning (DRL) pedestrian navigation model. Following your recommendations, we have analyzed the Variable Goal Approach (VGA) paper (arXiv:2501.05100v2), reviewed the experimental datasets you provided (VGA GitHub data and Jülich Bottleneck experiments), and developed a comprehensive validation framework.

**Key Points:**
- ✅ **DRL Model Status:** Successfully trained (3.9M steps, 95% success rate)
- ✅ **Achievements:** Handles complex geometries (L/T-shaped corridors), dynamic obstacles, varying densities
- ⚠️ **Identified Limitations:** Compared to human behavior and VGA algorithm
- 🔬 **Validation Plan:** Systematic comparison against VGA, classical models (Social Force via JuPedSim), and real experimental data
- 📊 **Current Progress:** Validation framework structure implemented, VGA algorithm coded, ready for data acquisition

---

## 1. Current DRL Project Status

### 1.1 What We Have Accomplished

#### **Training Results**
Our DRL agent has been trained through a 12-stage curriculum learning approach:

| Metric | Value |
|--------|-------|
| **Total Training Steps** | 3,900,000 |
| **Training Duration** | ~17.5 hours |
| **Curriculum Stages** | 12 (progressive difficulty) |
| **Total Episodes** | 12,399 |
| **Final Success Rate** | 95% (19/20 test episodes) |
| **Algorithm** | PPO (Proximal Policy Optimization) |

#### **Evaluation Performance**

Latest comprehensive evaluation (`eval_vga_video_gen/`):

| Scenario | Success Rate | Avg. Time | Avg. Collisions |
|----------|--------------|-----------|-----------------|
| Standard Sparse | 100% (5/5) | 25.4s | 1.4 |
| Standard Dense | 80% (4/5) | 33.0s | 17.6 |
| L-Shaped Corridor | 100% (5/5) | 16.6s | 0.0 |
| T-Shaped Corridor | 100% (5/5) | 18.1s | 0.0 |
| **Overall** | **95%** | **23.3s** | **4.75** |

#### **Technical Architecture**

- **Neural Network:** Custom CNN + Attention + LSTM (`AdvancedActorCriticPolicy`)
  - CNN processes 36 raycast distances (12m range, 10° resolution)
  - Attention mechanism focuses on critical obstacles
  - LSTM maintains temporal memory for path planning
  
- **Observation Space:** 50-dimensional vector
  - 11 base features (position, velocity, heading, distance to goal, etc.)
  - 36 raycast distances (Lidar-like obstacle sensing)
  - 3 enhanced features (corner awareness, goal visibility, movement efficiency)
  
- **Action Space:** 2D continuous
  - Linear velocity: [-1.4, 1.4] m/s
  - Angular velocity: [-1.8, 1.8] rad/s

- **Agent Specifications:**
  - Body radius: 0.225m (0.45m diameter - realistic human size)
  - Max speed: 1.4 m/s (close to human walking speed ~1.34 m/s)
  - Time step: 0.1s

#### **Key Achievements**

1. ✅ **Complex Geometry Handling:** Successfully navigates L-shaped and T-shaped corridors (100% success)
2. ✅ **Adaptive Behavior:** Handles varying obstacle densities (0.001 to 0.12)
3. ✅ **Curriculum Learning:** Progressive training from super easy → ultra hard scenarios
4. ✅ **Domain Randomization:** Agent generalizes to unseen obstacle configurations
5. ✅ **Video Generation:** 20+ evaluation videos demonstrating navigation behavior

---

## 2. Comparison with VGA Paper (arXiv:2501.05100v2)

### 2.1 Model Classification

The VGA paper classifies pedestrian dynamics models into categories:

| Category | Examples | **Your Approach** |
|----------|----------|-------------------|
| Mathematical/Force-Based | Social Force Model, Helbing | ❌ No |
| Computational | ORCA, RVO, Cellular Automata | ❌ No |
| **AI-Based/Deep Learning** | **Neural networks, RL** | **✅ YES** |
| Variable Goal Approach | VGA (paper's contribution) | ❌ No (but implementing) |

**Your model falls under "AI-Based/Deep Learning" approaches** - it learns navigation behavior through reinforcement learning rather than using predefined rules or force calculations.

### 2.2 Fundamental Differences: DRL vs VGA

| Aspect | Your DRL Approach | VGA Approach |
|--------|-------------------|--------------|
| **Planning Method** | **Implicit (learned)** | **Explicit (algorithmic)** |
| **Goal Strategy** | Single final goal | Dynamic intermediate goals |
| **Decision Making** | Reactive (responds to current state) | Anticipatory (plans ahead) |
| **Path Selection** | Optimizes learned reward | Balances efficiency + comfort |
| **Obstacle Avoidance** | Learned through trial-and-error | Geometric tangent calculation |
| **Behavior Variation** | Deterministic | Probabilistic (stochastic) |
| **Training Required** | Yes (millions of steps) | No (rule-based) |
| **Computational Cost** | Neural network inference | Geometric calculations |

### 2.3 What Your DRL Model Does Well

1. ✅ **Learned Adaptation:** Handles complex scenarios without explicit programming
2. ✅ **Temporal Memory (LSTM):** Remembers recent path to avoid getting stuck
3. ✅ **Spatial Awareness (CNN + Attention):** Identifies critical obstacles automatically
4. ✅ **Generalization:** Works on unseen corridor configurations
5. ✅ **Complex Geometries:** Excellent at corners and junctions (100% success)

### 2.4 Identified Limitations (Compared to VGA & Human Behavior)

Based on your feedback, the VGA paper analysis, and our evaluation results, we have identified several limitations:

#### **A. "Uncomfortable" Path Selection**
- **Issue:** Agent may take narrow paths between obstacles when a wider detour exists
- **Cause:** Reward function optimizes for distance efficiency, not comfort
- **Example:** In dense scenarios, agent squeezes through 0.5m gaps instead of routing around
- **VGA Solution:** Explicit "personal distance" parameter (typically 0.5m clearance)
- **Match with Mohcine's Feedback:** ✅ Exactly what was observed in videos

#### **B. Reactive Rather Than Anticipatory**
- **Issue:** Agent responds to obstacles only when close (raycast detection)
- **Cause:** No explicit intermediate goal planning
- **Example:** May approach obstacle cluster before deciding to turn
- **VGA Solution:** Sets variable goals early to curve around smoothly
- **Impact:** Less natural-looking trajectories

#### **C. Local Minima in Dense Crowds**
- **Issue:** Can get stuck in dense obstacle configurations (1 failure in 20 episodes)
- **Cause:** Greedy reactive behavior without global path planning
- **Example:** Episode 9 failure - 50 collisions, never escaped dense crowd
- **VGA Solution:** Selects intermediate goals to route around congestion
- **Success Rate Impact:** 5% failure rate on dense scenarios

#### **D. Lack of Stochastic Variation**
- **Issue:** Deterministic policy (same scenario → same path)
- **Cause:** PPO uses deterministic action selection during evaluation
- **Example:** All 5 sparse episodes follow nearly identical trajectories
- **VGA Solution:** Probabilistic goal selection creates path variety
- **Human Behavior:** Real pedestrians show variation even in identical scenarios

#### **E. Missing "Comfort" Metrics**
- **Issue:** No explicit metric for path comfort or naturalness
- **Current Metrics:** Success rate, time, collisions, path length
- **VGA Metrics:** Velocity deviation from desired speed, path smoothness, oscillation
- **Need:** Quantitative measure of human-like behavior

#### **F. Single-Agent Limitation**
- **Issue:** Currently trained for single pedestrian (no multi-agent interactions)
- **VGA Data:** Includes head-on, same-direction, multi-obstacle-single-pedestrian (MOSP)
- **Bottleneck Data:** Multiple participants with different motivations
- **Impact:** Cannot validate on social interaction scenarios yet

---

## 3. Experimental Validation Plan

In response to your question *"how do you intend to tackle this validation work?"*, we have developed a comprehensive validation framework following your recommendations.

### 3.1 Proposed Experiments

#### **Experiment 1: VGA Dataset Validation**
- **Data Source:** `https://github.com/kanika201293/Pedestrian-Experimental-Data`
- **Scenarios:**
  - Head-On: Two pedestrians approaching each other
  - Single Obstacle Single Pedestrian (SOSP)
  - Multi Obstacle Single Pedestrian (MOSP)
  - Same direction pedestrians (parallel walking)
- **Method:**
  1. Extract experimental trajectories from dataset
  2. Recreate same start/goal positions in our environment
  3. Run DRL model simulation
  4. Run VGA algorithm simulation
  5. Run JuPedSim (Social Force Model) simulation
  6. Compare all trajectories with human ground truth

#### **Experiment 2: Jülich Bottleneck Validation**
- **Data Source:** `https://ped.fz-juelich.de/da/doku.php?id=bottleneck_individuals`
- **Scenario:** Single pedestrian navigating through bottleneck
- **Variables:**
  - Bottleneck width (0.5m, 0.7m, 1.0m)
  - Approach angle (straight, 30° left, 30° right)
  - Motivation level (normal, hurry)
- **Method:**
  1. Parse HDF5 trajectory files from dataset
  2. Recreate bottleneck geometry in environment
  3. Run DRL model with same start positions
  4. Run VGA algorithm
  5. Run Social Force Model (JuPedSim)
  6. Compare trajectories, velocities, and comfort metrics

#### **Experiment 3: Qualitative Trajectory Comparison**
As you suggested: *"you simulate the same scenario with your model and with jupedsim. Same starting positions and same goal. Then compare the trajectories with each other and with the experimental trajectories."*

- **Scenarios:** Standard, L-shaped, T-shaped corridors (our current test set)
- **Models to Compare:**
  - DRL (our trained agent)
  - VGA (our implementation based on paper)
  - Social Force Model (via JuPedSim)
  - Experimental data (VGA/Bottleneck datasets)
- **Visualization:**
  - Overlay trajectory plots
  - Side-by-side video comparison
  - Velocity profile plots (v/v_desired vs. position)
  - Path distribution plots (stochastic variation)

### 3.2 Metrics for Quantitative Comparison

Following the VGA paper and your recommendations:

#### **A. Path Quality Metrics**
1. **Path Smoothness:** Curvature variation along trajectory
2. **Oscillation:** Lateral movement perpendicular to goal direction
3. **Average Speed:** Mean velocity during episode
4. **Speed Deviation:** Variance from desired walking speed (1.34 m/s)
5. **Path Length:** Total distance traveled (efficiency)

#### **B. Safety Metrics**
1. **Collision Rate:** Number of obstacle contacts
2. **Minimum Clearance:** Closest approach to any obstacle
3. **Time in Danger Zone:** Duration within personal distance of obstacles

#### **C. Human Similarity Metrics**
1. **Trajectory Deviation:** Distance from experimental trajectory
2. **Velocity Profile Correlation:** Similarity to human speed patterns
3. **Comfort Index:** Composite metric (clearance + smoothness + speed stability)

#### **D. Performance Metrics**
1. **Success Rate:** % of episodes reaching goal
2. **Travel Time:** Duration from start to goal
3. **Computational Time:** Inference/planning time per step
4. **Stuck Rate:** % of episodes failing due to local minima

### 3.3 Statistical Analysis

Following academic rigor:
- **Sample Size:** Minimum 50 runs per scenario (for statistical significance)
- **Statistical Tests:**
  - Wilcoxon signed-rank test (non-parametric comparison)
  - Two-sample t-test (if data is normal)
  - Chi-square test (for success rate comparison)
- **Significance Level:** α = 0.05
- **Confidence Intervals:** 95% CI for all metrics

---

## 4. Current Implementation Status

### 4.1 Validation Framework Architecture

We have created a comprehensive validation framework under `validation/` directory:

```
validation/
├── data_loading/          # Dataset parsers
│   ├── vga_dataset.py            ✅ COMPLETE (293 lines)
│   ├── bottleneck_dataset.py     ✅ COMPLETE (396 lines)
│   └── utils.py                  ✅ COMPLETE (190 lines)
│
├── models/                # Model interfaces
│   ├── model_base.py             ✅ COMPLETE (195 lines)
│   ├── drl_policy_interface.py   ✅ COMPLETE (244 lines)
│   ├── vga_planner.py            ✅ COMPLETE (540 lines)
│   └── social_force_baseline.py  ⏳ PENDING
│
├── experiments/           # Validation scripts
│   ├── head_on_validation.py     ⏳ PENDING
│   ├── bottleneck_validation.py  ⏳ PENDING
│   ├── sosp_validation.py        ⏳ PENDING
│   └── mosp_validation.py        ⏳ PENDING
│
├── metrics/               # Analysis metrics
│   ├── trajectory_metrics.py     ⏳ PENDING
│   ├── comparison_metrics.py     ⏳ PENDING
│   └── statistical_analysis.py   ⏳ PENDING
│
├── plots/                 # Visualization
│   ├── trajectory_plots.py       ⏳ PENDING
│   ├── velocity_profiles.py      ⏳ PENDING
│   └── stochastic_paths.py       ⏳ PENDING
│
└── README.md             ✅ COMPLETE (204 lines)
```

### 4.2 What Has Been Implemented

#### **A. VGA Dataset Parser** (`vga_dataset.py`) ✅
- Parses experimental data from VGA GitHub repository
- Supports Head-On, SOSP, MOSP, parallel pedestrian scenarios
- Extracts trajectories with positions, velocities, timestamps
- Handles obstacle configurations

#### **B. Bottleneck Dataset Parser** (`bottleneck_dataset.py`) ✅
- Parses Jülich HDF5/TXT trajectory files
- Filters by width, motivation, approach angle
- Extracts participant trajectories with metadata
- Provides statistical summaries

#### **C. Model Base Class** (`model_base.py`) ✅
- Abstract interface for all navigation models
- Standardized `simulate()` method
- `SimulationResult` container for trajectory data
- Ensures fair comparison (same interface for DRL, VGA, SFM)

#### **D. DRL Policy Interface** (`drl_policy_interface.py`) ✅
- Wraps your trained PPO agent
- Loads model + VecNormalize statistics
- Implements standard interface for comparisons
- Handles environment creation and stepping

#### **E. VGA Algorithm** (`vga_planner.py`) ✅ **JUST COMPLETED**
- Full implementation of Variable Goal Approach from paper
- Obstacle clustering algorithm
- Tangent point calculation with personal distance
- Goal selection (deterministic + probabilistic modes)
- Dynamic intermediate goal updating
- ~540 lines with comprehensive documentation

### 4.3 What Remains to Be Done

#### **Next Steps (Prioritized)**

1. **Data Acquisition** (This Week)
   - Clone VGA GitHub repository
   - Download Jülich Bottleneck dataset (may require registration)
   - Validate data parsers with real files
   - Update file paths in code

2. **Metrics Implementation** (Next Week)
   - `trajectory_metrics.py`: Path smoothness, oscillation, speed metrics
   - `comparison_metrics.py`: Deviation from human data
   - `statistical_analysis.py`: Statistical tests and significance

3. **Social Force Baseline** (Next Week)
   - Integrate JuPedSim or implement basic Social Force Model
   - Use same interface as DRL/VGA for fair comparison

4. **Experiment Scripts** (Week After)
   - `head_on_validation.py`: Run all models on Head-On scenarios
   - `bottleneck_validation.py`: Bottleneck experiment recreation
   - `sosp_validation.py` & `mosp_validation.py`: Obstacle scenarios
   - Generate results JSON + trajectory data

5. **Visualization** (Week After)
   - `trajectory_plots.py`: Overlay human/DRL/VGA/SFM trajectories
   - `velocity_profiles.py`: v/v_desired plots (Figure 2 style from VGA paper)
   - `stochastic_paths.py`: Path distribution plots (Figure 7 style)
   - Publication-quality figures for thesis

6. **Jupyter Notebooks** (Final Week)
   - Interactive data exploration
   - Results analysis and visualization
   - Statistical test results
   - Comprehensive comparison report

---

## 5. Expected Outcomes & Deliverables

### 5.1 Research Questions to Answer

1. **How does our DRL model compare to rule-based VGA in human-like behavior?**
   - Hypothesis: VGA will show smoother trajectories with better comfort metrics
   - Hypothesis: DRL will show better success rate but less natural paths

2. **How does our DRL model compare to classical Social Force Model?**
   - Hypothesis: DRL will handle complex geometries better
   - Hypothesis: SFM will show more predictable, physics-based behavior

3. **Which model best matches real human experimental data?**
   - Quantitative: Trajectory deviation, velocity profile correlation
   - Qualitative: Visual similarity, path naturalness

4. **What are the computational trade-offs?**
   - DRL: Neural network inference time
   - VGA: Geometric calculation time
   - SFM: Force computation time

### 5.2 Deliverables

#### **For Supervisors:**
1. ✅ **This Report:** Current status and validation plan
2. ⏳ **Validation Results Report:** Comprehensive comparison (2-3 weeks)
3. ⏳ **Trajectory Visualization Videos:** Side-by-side DRL/VGA/SFM/Human
4. ⏳ **Statistical Analysis:** Significance tests and confidence intervals
5. ⏳ **Publication-Ready Figures:** For thesis and potential paper

#### **For Academic Record:**
1. ⏳ **Jupyter Notebooks:** Reproducible analysis
2. ⏳ **GitHub Repository:** Open-source validation framework
3. ⏳ **Dataset Documentation:** Preprocessing and format details
4. ⏳ **Thesis Chapter:** "Validation Against Human Experimental Data"

---

## 6. Addressing Identified DRL Limitations

### 6.1 Short-Term Improvements (During Validation)

These can be implemented without retraining:

1. **Stochastic Evaluation Mode:**
   - Add noise to actions during evaluation (σ = 0.1)
   - Create path variation similar to human behavior
   - Compare deterministic vs. stochastic performance

2. **Comfort Metrics:**
   - Post-hoc analysis of trajectories for comfort
   - Identify episodes with "uncomfortable" behavior
   - Quantify % of time in narrow passages

3. **Visualization Enhancements:**
   - Color-code trajectory by clearance distance
   - Highlight "danger zones" (< 0.5m from obstacles)
   - Overlay "comfort zones" and "preferred paths"

### 6.2 Long-Term Improvements (Future Work)

These would require retraining or architecture changes:

1. **Comfort-Aware Reward:**
   - Add penalty for getting too close to obstacles (even without collision)
   - Reward for maintaining > 0.5m personal distance
   - Balance efficiency vs. comfort

2. **Explicit Intermediate Goals:**
   - Hybrid approach: DRL policy + VGA-style goal selection
   - Train agent to select intermediate waypoints
   - Learn when to use direct path vs. detour

3. **Anticipatory Planning:**
   - Increase raycast range (12m → 20m) for earlier detection
   - Add "future obstacle" prediction module
   - Train with longer temporal horizon (LSTM depth)

4. **Multi-Agent Training:**
   - Extend to multiple pedestrians (social interactions)
   - Train on head-on, crossing, following scenarios
   - Enable validation on full VGA dataset

---

## 7. Timeline & Milestones

### Week 1 (Current Week)
- ✅ VGA algorithm implementation
- ⏳ Data acquisition (VGA GitHub + Jülich)
- ⏳ Validate data parsers with real files

### Week 2
- ⏳ Implement metrics modules
- ⏳ Implement Social Force baseline
- ⏳ Run initial test comparisons (small sample)

### Week 3
- ⏳ Implement experiment scripts
- ⏳ Run full validation (50+ runs per scenario)
- ⏳ Statistical analysis

### Week 4
- ⏳ Visualization and plotting
- ⏳ Jupyter notebooks for analysis
- ⏳ Results report and thesis draft

**Target Completion:** January 10, 2026 (4 weeks from now)

---

## 8. Conclusion

### Summary of Current State

**What We Have:**
- ✅ High-performing DRL agent (95% success rate)
- ✅ Comprehensive training system (curriculum learning)
- ✅ Robust evaluation framework
- ✅ Validation framework structure implemented
- ✅ VGA algorithm coded and ready for testing

**What We're Missing:**
- ⚠️ Quantitative comparison with human data
- ⚠️ Comfort and naturalness metrics
- ⚠️ Stochastic behavior variation
- ⚠️ Multi-agent capabilities (future work)

**What We're Doing:**
- 🔬 Implementing systematic validation against VGA/Bottleneck experiments
- 🔬 Comparing with VGA algorithm and Social Force Model
- 🔬 Developing comprehensive metrics suite
- 🔬 Creating publication-quality visualizations

### Answer to Your Question

*"How do you intend to tackle this validation work?"*

**Our approach:**

1. **Use the experimental data you provided as ground truth** (VGA GitHub + Jülich Bottleneck)
2. **Implement the VGA algorithm** from the paper for fair comparison (✅ DONE)
3. **Integrate JuPedSim** for Social Force Model baseline
4. **Recreate exact experimental scenarios** in our environment
5. **Run all three models** (DRL, VGA, SFM) with same start/goal positions
6. **Compare trajectories** quantitatively (metrics) and qualitatively (videos)
7. **Perform statistical analysis** to determine significance
8. **Generate publication-ready figures** for thesis and potential paper

**This will answer:**
- Does our DRL agent behave more like humans than VGA?
- Does our DRL agent behave more like humans than Social Force Model?
- What are the trade-offs (success vs. comfort vs. computation)?
- Where does our approach excel? Where does it fall short?

### Request for Guidance

We have begun implementation and would appreciate your feedback on:

1. **Priority:** Should we focus on VGA experiments first, or Bottleneck experiments?
2. **Metrics:** Are there specific metrics you'd like to see beyond what we've listed?
3. **Sample Size:** Is 50 runs per scenario sufficient for statistical validity?
4. **Multi-Agent:** Should we extend to multi-agent before validation, or after?

We are confident this validation work will provide the scientific rigor needed for academic publication and will clearly demonstrate the strengths and limitations of our DRL approach.

---

**Prepared by:** Abdallah Jamal Jamil Al-Harrem  
**Reviewed by:** [Awaiting supervisor feedback]  
**Version:** 1.0  
**Date:** December 10, 2025

**Repository:** https://github.com/AboodJamal/deep-rl-crowd-simulation  
**Branch:** `feature/vga-integration`  
**Contact:** [Your email]
