# 🎯 Deep Reinforcement Learning for Crowd Simulation - Complete Project Overview

**Project:** Deep Reinforcement Learning for Pedestrian Navigation in Complex Corridors  
**Student:** Abdallah Jamal Jamil Al-Harrem  
**Supervisors:** Mohcine and Ahmad  
**Status:** ✅ Training Complete | 📊 Evaluation Phase | 🔬 Validation Pending

---

## 📋 Table of Contents

1. [Project Summary](#project-summary)
2. [Approach & Methodology](#approach--methodology)
3. [File Structure & Purpose](#file-structure--purpose)
4. [Current State](#current-state)
5. [Training Results](#training-results)
6. [Evaluation Results](#evaluation-results)
7. [Comparison with VGA Paper](#comparison-with-vga-paper)
8. [Next Steps & Validation Plan](#next-steps--validation-plan)

---

## 🎯 Project Summary

### What We're Building

A **Deep Reinforcement Learning (DRL)** agent that navigates complex corridor environments (Standard, L-shaped, T-shaped) with dynamic obstacles. The agent learns to:
- Navigate from start to goal position
- Avoid obstacles (static boxes)
- Handle complex geometries (corners, junctions)
- Adapt to varying obstacle densities (0.001 to 0.12)
- Make human-like navigation decisions

### Key Achievement

**95% Success Rate** on evaluation scenarios after **3.9 million training steps** across 12 curriculum stages.

---

## 🔬 Approach & Methodology

### **Category: AI-Based / Deep Learning** (from VGA paper classification)

**Your Approach:**
- **Algorithm:** PPO (Proximal Policy Optimization) from Stable-Baselines3
- **Architecture:** CNN + Attention + LSTM (via `AdvancedActorCriticPolicy`)
- **Learning Method:** Trial-and-error through rewards (Reinforcement Learning)
- **Observation:** 50-dimensional vector (11 base features + 36 raycast distances + 3 enhanced features)
- **Action:** Continuous 2D (linear velocity, angular velocity)

**What This Means:**
- ✅ **NOT** a physics-based model (like Social Force Model)
- ✅ **NOT** using Variable Goal Approach (VGA) - no explicit intermediate goals
- ✅ **IS** a Deep Reinforcement Learning approach that learns navigation implicitly

### **How It Works:**

1. **Agent receives observations:**
   - Position, velocity, heading
   - Distance to goal
   - 36 raycast distances (Lidar-like sensing)
   - Corner awareness, goal visibility

2. **Neural network processes:**
   - CNN processes raycast data (spatial understanding)
   - Attention mechanism focuses on important obstacles
   - LSTM maintains temporal memory (remembers recent path)

3. **Agent outputs action:**
   - Linear velocity (forward/backward)
   - Angular velocity (turn left/right)

4. **Environment provides reward:**
   - Progress toward goal: `+10.0` per meter closer
   - Goal reached: `+1000.0`
   - Collision: `-20.0` (progressive penalty)
   - Spinning/backward/stalling: Negative penalties

5. **Agent learns:**
   - Through millions of episodes, learns which actions lead to success
   - Develops implicit "path planning" (not explicit like VGA)

---

## 📁 File Structure & Purpose

### **Core Training Files**

#### `ultimate_curriculum_trainer.py`
**Purpose:** Main training script that orchestrates curriculum learning  
**Key Features:**
- Defines 12 training stages (progressive difficulty)
- Creates PPO model with `AdvancedActorCriticPolicy`
- Manages domain randomization
- Saves checkpoints after each stage
- Logs to Weights & Biases (W&B)
- **Usage:** `python ultimate_curriculum_trainer.py --timesteps 3900000`

**What It Does:**
1. Creates environment with difficulty level and allowed shapes
2. Wraps in `DummyVecEnv` (parallel environments) and `VecNormalize` (normalization)
3. Creates/loads PPO model
4. Trains for specified timesteps
5. Saves model checkpoint
6. Moves to next stage

#### `ultimate_domain_randomization_env.py`
**Purpose:** Gymnasium environment that simulates corridor navigation  
**Key Features:**
- Supports 3 corridor types: `standard`, `lshaped`, `tshaped`
- Domain randomization: obstacle density, positions, goal/start locations
- Raycasting (36 rays, 12m range) for obstacle detection
- Collision detection (agent radius = 0.225m)
- Reward calculation
- **Observation Space:** 50 values (11 base + 36 rays + 3 enhanced)
- **Action Space:** 2 values (velocity, angular_velocity)

**Difficulty Levels:**
- `super_easy`: 0.001-0.005 density
- `easy`: 0.005-0.015 density
- `medium`: 0.015-0.04 density
- `hard`: 0.04-0.08 density
- `mixed`: 0.01-0.08 density
- `ultra`: 0.06-0.12 density

#### `advanced_policy_network.py`
**Purpose:** Custom neural network architecture for the agent  
**Key Components:**
- `RaycastingCNN`: Processes raycast data (spatial understanding)
- `SpatialAttention`: Attention mechanism (focuses on important obstacles)
- `AdvancedFeaturesExtractor`: Combines CNN + Attention + LSTM
- `AdvancedActorCriticPolicy`: PPO policy using the advanced architecture

**Architecture Flow:**
```
Observations (50) 
  → Base Features (11) → MLP
  → Raycast (36) → CNN → Attention → LSTM
  → Enhanced (3) → MLP
  → Concatenate → Actor/Critic Heads
  → Action (2) / Value (1)
```

### **Evaluation Files**

#### `ultimate_evaluation.py`
**Purpose:** Comprehensive evaluation script  
**Key Features:**
- Tests on 4 scenarios: Standard Sparse, Standard Dense, L-shaped, T-shaped
- Runs multiple episodes per scenario
- Generates MP4 videos for each episode
- Saves trajectory data (JSON)
- **Usage:** `python ultimate_evaluation.py --model models/ultimate_generalized_agent.zip --episodes-per-scenario 5`

**Output:**
- `eval_[name]/complete_results.json`: All episode data
- `eval_[name]/[scenario]/ep###_[success|failure].mp4`: Videos

### **Supporting Files**

#### `numpy_compat_fix.py`
**Purpose:** Compatibility fix for numpy/stable-baselines3 version conflicts

#### `reqs.txt`
**Purpose:** Python dependencies list

#### `VGA_Explanation.md`
**Purpose:** Explanation of the Variable Goal Approach (VGA) paper (arXiv:2501.05100v2)  
**Content:** Detailed breakdown of VGA methodology, comparison with your approach

---

## 📊 Current State

### **Training Status: ✅ COMPLETE**

**Final Model:** `models/ultimate_generalized_agent.zip`  
**Training Duration:** 3.9 million timesteps across 12 stages  
**Total Training Time:** ~17.5 hours  
**Final Success Rate:** 90% (Ultra Challenge stage)

### **Model Checkpoints Available:**

```
models/
├── ultimate_generalized_agent.zip              # Final model (all stages)
├── ultimate_generalized_agent_vecnormalize.pkl # Normalization stats
├── ultimate_generalized_agent_stage1.zip       # After stage 1
├── ultimate_generalized_agent_stage2.zip       # After stage 2
├── ... (stages 3-12)
└── ultimate_generalized_agent_stage12.zip      # After stage 12
```

### **Evaluation Status: ✅ COMPLETE**

**Latest Evaluation:** `eval_vga_video_gen/`  
**Overall Success Rate:** 95.0% (19/20 episodes)  
**Scenarios Tested:**
- Standard Sparse: 100% (5/5)
- Standard Dense: 80% (4/5) ← One failure in dense crowd
- L-Shaped Corridor: 100% (5/5)
- T-Shaped Corridor: 100% (5/5)

**Videos Generated:** 20 MP4 files showing agent navigation

---

## 📈 Training Results

### **Curriculum Stages (12 Total)**

| Stage | Name | Difficulty | Shapes | Steps | Success Rate | Episodes |
|-------|------|------------|--------|-------|--------------|----------|
| 1 | Super Easy Standard | super_easy | standard | 300k | 28% | 643 |
| 2 | Standard Sparse | easy | standard | 300k | 96% | 673 |
| 3 | Standard Dense Easy | easy | standard | 300k | 97% | 921 |
| 4 | Standard Medium | medium | standard | 300k | 94% | 702 |
| 5 | Standard Hard | hard | standard | 300k | 88% | 891 |
| 6 | L/T Super Easy | super_easy | lshaped, tshaped | 300k | 99% | 1403 |
| 7 | L/T Easy | easy | lshaped, tshaped | 300k | 96% | 1546 |
| 8 | L/T Medium | medium | lshaped, tshaped | 300k | 97% | 1021 |
| 9 | L/T Hard | hard | lshaped, tshaped | 300k | 91% | 1150 |
| 10 | All Mixed | mixed | standard, lshaped, tshaped | 600k | 86% | 2418 |
| 11 | Pattern Navigation | medium | standard | 300k | 96% | 877 |
| 12 | Ultra Challenge | ultra | standard, lshaped, tshaped | 300k | 90% | 1154 |

**Total:** 3,900,000 timesteps, 12,399 episodes

### **Key Observations:**

1. **Stage 1 (Super Easy) had low success (28%):** Agent was still learning basics
2. **Stages 2-5:** Rapid improvement on standard corridors (96-97% success)
3. **L/T stages:** Excellent performance (96-99% success) - agent handles corners well
4. **Mixed stages:** Slight drop (86-90%) due to increased complexity
5. **Final stage (Ultra):** 90% success on hardest scenarios

---

## 📊 Evaluation Results

### **Latest Evaluation (`eval_vga_video_gen/`)**

**Command Used:**
```bash
python ultimate_evaluation.py --model models/ultimate_generalized_agent.zip --output-dir eval_vga_video_gen --episodes-per-scenario 5
```

**Results:**

| Scenario | Success Rate | Avg Time | Avg Collisions | Notes |
|----------|--------------|----------|----------------|-------|
| Standard Sparse | 100% (5/5) | 25.4s | 1.4 | Perfect navigation |
| Standard Dense | 80% (4/5) | 33.0s | 17.6 | One failure (ep009) - got stuck in dense crowd |
| L-Shaped Corridor | 100% (5/5) | 16.6s | 0.0 | Excellent corner navigation |
| T-Shaped Corridor | 100% (5/5) | 18.1s | 0.0 | Perfect junction handling |

**Overall:** 95.0% success rate (19/20 episodes)

### **Failure Analysis:**

**ep009_failure.mp4 (Standard Dense):**
- **Issue:** Agent got stuck in very dense crowd (50 collisions)
- **Cause:** Agent tried to push through instead of routing around
- **This matches Mohcine's feedback:** Agent takes "uncomfortable" narrow paths

---

## 🔬 Comparison with VGA Paper

### **Paper Classification (arXiv:2501.05100v2)**

The paper classifies pedestrian models into:
1. **Mathematical/Force-Based** (e.g., Social Force Model)
2. **Computational** (e.g., ORCA, Cellular Automata)
3. **AI-Based/Deep Learning** ← **YOUR APPROACH**
4. **Variable Goal Approach (VGA)** - Their proposed method

### **Your Approach vs. VGA**

| Aspect | Your DRL Approach | VGA Approach |
|--------|-------------------|-------------|
| **Method** | Neural network learns navigation | Explicit intermediate goal calculation |
| **Planning** | Implicit (learned) | Explicit (algorithmic) |
| **Goals** | Single final goal | Dynamic intermediate goals |
| **Behavior** | Reactive (learns to react) | Anticipatory (plans ahead) |
| **Comfort** | May take narrow paths (optimizes distance) | Balances efficiency + comfort |
| **Stochasticity** | Deterministic (same state → same action) | Probabilistic (varied paths) |

### **Issues with Your Approach (According to Paper):**

1. **Reactive vs. Anticipatory:**
   - Your agent may get too close to obstacles before turning
   - VGA agents set temporary goals early to curve around smoothly

2. **"Uncomfortable" Path Selection:**
   - Your agent may squeeze through narrow gaps (shortest path)
   - VGA agents prefer wider, more comfortable paths
   - **This matches Mohcine's feedback!**

3. **Local Minima:**
   - Your agent can get stuck in dense crowds (ep009 failure)
   - VGA agents set intermediate goals to route around congestion

4. **Lack of Stochasticity:**
   - Your agent is deterministic (always same path for same scenario)
   - VGA introduces randomness for realistic variation

### **What You're Missing (VGA Features):**

- ❌ No explicit intermediate goal calculation
- ❌ No "comfort" metric in reward function
- ❌ No probabilistic path selection
- ❌ No anticipatory planning (only reactive learning)

---

## 🎯 Next Steps & Validation Plan

### **Supervisor Feedback Summary:**

**Mohcine's Requests:**
1. ✅ Compare with classical models (SFM via JuPedSim)
2. ✅ Compare realism (how close to human behavior)
3. ✅ Compare performance (execution time)
4. ✅ Validate against VGA experiment data (India study)
5. ✅ Validate against Jülich Bottleneck experiment

**Ahmad's Plan:**
1. Use real experimental data as ground truth
2. Recreate scenarios with both DRL and physics-based models
3. Compare trajectories and speeds
4. Quantitative validation

### **Validation Experiments Needed:**

#### **1. VGA Experiment (Variable Goal Approach)**
- **Paper:** arXiv:2501.05100v2
- **Data:** `https://github.com/kanika201293/`
- **Task:** Recreate scenarios, compare trajectories

#### **2. Jülich Bottleneck Experiment**
- **Data:** `https://ped.fz-juelich.de/da/doku.php?id=bottleneck_individuals`
- **Task:** Simulate bottleneck scenarios, compare with real data

#### **3. JuPedSim Comparison**
- **Tool:** `https://app.jupedsim.org/`
- **Task:** Run same scenarios in JuPedSim (SFM), compare:
  - Trajectories (qualitative)
  - Execution time (quantitative)
  - Realism (qualitative)

### **Current Status:**

- ✅ **Training:** Complete (3.9M steps, 95% success)
- ✅ **Evaluation:** Complete (videos generated)
- ⏳ **Validation:** Pending (need to run comparisons)
- ⏳ **Multi-agent:** In development (mentioned in email to supervisors)

### **Immediate Next Steps:**

1. **Set up JuPedSim:**
   - Install JuPedSim
   - Recreate evaluation scenarios
   - Run SFM simulations
   - Compare trajectories

2. **Download VGA Data:**
   - Access `github.com/kanika201293/`
   - Understand experiment setup
   - Recreate scenarios in your environment

3. **Download Jülich Data:**
   - Access bottleneck experiment data
   - Understand format
   - Adapt environment to match

4. **Performance Comparison:**
   - Measure execution time for your DRL model
   - Compare with JuPedSim execution time
   - Document results

---

## 📝 Technical Details

### **Environment Specifications:**

- **Agent Radius:** 0.225m
- **Max Velocity:** 1.4 m/s
- **Max Angular Velocity:** 1.8 rad/s
- **Time Step:** 0.1s
- **Max Steps per Episode:** 500
- **Raycast:** 36 rays, 12m range, 10° resolution

### **Neural Network Architecture:**

- **Policy:** `AdvancedActorCriticPolicy`
- **Features:** CNN (raycast) + Attention + LSTM
- **Hidden Layers:** 256-512 neurons
- **Activation:** ReLU/Tanh
- **Optimizer:** Adam (via PPO)

### **Training Hyperparameters:**

- **Algorithm:** PPO
- **Learning Rate:** 0.0003 (adaptive)
- **Batch Size:** 256
- **Clip Range:** 0.2
- **Entropy Coefficient:** 0.02 (adaptive)
- **Gamma (discount):** 0.99
- **GAE Lambda:** 0.95

### **Reward Structure:**

```python
# Progress reward
progress_reward = 10.0 * (previous_distance - current_distance)

# Goal reached
goal_reward = 1000.0

# Collision penalty (progressive)
collision_penalty = -20.0 * (1 + collision_count * 0.1)

# Behavior penalties
spinning_penalty = -2.0 if angular_vel > 2.0
backward_penalty = -1.0 if moving backward
stalling_penalty = -0.5 if stuck
```

---

## 📚 References & Resources

### **Papers:**
- **VGA Paper:** arXiv:2501.05100v2 - "Variable Goal Approach (VGA) Enhancing Pedestrian Dynamics Modeling"
- **Bottleneck Experiment:** Jülich Research Centre data

### **Tools:**
- **JuPedSim:** `https://app.jupedsim.org/` (Social Force Model simulator)
- **Stable-Baselines3:** PPO implementation
- **Weights & Biases:** Training logging

### **Data Sources:**
- **VGA Experiment:** `https://github.com/kanika201293/`
- **Jülich Bottleneck:** `https://ped.fz-juelich.de/da/doku.php?id=bottleneck_individuals`

---

## 🎓 Project Context

### **Academic Setting:**
- **Course:** Training/Research Project
- **Institution:** An-Najah National University
- **Supervisors:** Mohcine and Ahmad
- **Focus:** Deep Reinforcement Learning for Crowd Simulation

### **Communication History:**
- **Initial Demo:** Sent to supervisors with videos
- **Mohcine's Feedback:** Agent takes "uncomfortable" narrow paths, requested comparison with SFM
- **Ahmad's Plan:** Use real experimental data for validation
- **Current Status:** Awaiting validation experiments

---

## ✅ Summary

**What You Have:**
- ✅ Fully trained DRL agent (95% success rate)
- ✅ Comprehensive evaluation system
- ✅ Video generation for demos
- ✅ 12-stage curriculum learning system
- ✅ Advanced neural network architecture (CNN+Attention+LSTM)

**What You're Missing (Compared to VGA):**
- ❌ Explicit intermediate goal planning
- ❌ "Comfort" awareness in path selection
- ❌ Probabilistic behavior variation
- ❌ Anticipatory planning (only reactive)

**What's Next:**
- ⏳ Validation against real experimental data
- ⏳ Comparison with JuPedSim (SFM)
- ⏳ Performance benchmarking
- ⏳ Multi-agent extension (mentioned in email)

**Current Model:** `models/ultimate_generalized_agent.zip`  
**Latest Evaluation:** `eval_vga_video_gen/` (95% success)  
**Status:** Ready for validation experiments

---

**Last Updated:** December 6, 2025  
**Document Version:** 1.0

