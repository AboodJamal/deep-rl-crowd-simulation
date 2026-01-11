---
marp: true
theme: gaia
class: invert
paginate: true
header: "DRL Pedestrian Simulation Project"
footer: "Internship Final Presentation | [Your Name]"
style: |
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
  section {
    font-size: 26px;
  }
  code {
    font-size: 0.8em;
  }
---

# Exploring Deep Reinforcement Learning for Realistic Pedestrian Simulation

## 11-Week Internship Final Presentation

**Presenter:** [Your Name]
**Role:** DRL Engineer Intern

---

# Executive Summary

**Goal:** Develop a DRL system for realistic pedestrian behavior simulation.

- **Evolution:** Progressed from basic theoretical study to a complex Multi-Agent System (MAS) on HPC infrastructure.
- **Core Study:** Comparative analysis between modern DRL (MAPPO, CTDE) and classical velocity-based models (VGA + UPL).
- **Key Outcome:** Proven efficacy of DRL in generating human-like trajectories and generalizing to unseen scenarios.

---

# Phase 1: Foundations & Environment Design
### Weeks 1–3

**Focus:** Bridging the gap between mathematical theory and simulation.

- **Foundations:** Deep dive into RL math (States, Actions, Rewards).
- **Development:** Created a custom "Point-in-Corridor" 2D environment.
- **Challenge:** Initial Cartesian observation system was too simplistic.
- **Outcome:** Established a baseline agent with simple goal-seeking behavior.

---

<!-- _class: invert -->

<div class="columns">
<div>

# Phase 2: Advanced Perception

**Focus:** Enhancing how the agent "sees".

- **Ray-Casting System:** Implemented LiDAR-like perception to detect geometry/obstacles.
- **Technical Leap:** Shift from Cartesian to Ray-Casting allowed navigation in complex corridors.
- **Result:** High-quality demos showing smooth navigation without collisions.

</div>
<div>

### Ray-Casting Demo

![width:100%](path/to/raycasting_demo.mp4)

*Placeholder for simulation video showing rays detecting walls.*

</div>
</div>

---

<div class="columns">
<div>

# Phase 3: Scaling to MAS
### Weeks 5–6

- **Algorithm:** Multi-Agent PPO (MAPPO).
- **Architecture:** 
  - **CTDE:** Centralized Training, Decentralized Execution.
  - **LSTM:** For temporal memory.
  - **CNN:** For complex observations.
- **Infrastructure:** Deployed on **Jülich HPC** GPU clusters.
- **Challenges:** Debugging parallel execution logs and synchronization issues.

</div>
<div>

### MAPPO Implementation

```python
# Placeholder: MAPPO / Network Config
class MAPPOAgent(nn.Module):
    def __init__(self, obs_space, act_space):
        super().__init__()
        self.actor = Actor(obs_space, act_space)
        self.critic = Critic(obs_space)
        # LSTM for memory
        self.lstm = nn.LSTM(input_size, hidden_size)
        
    def forward(self, x, hidden):
        # ... implementation ...
        return action, value, new_hidden
```

</div>
</div>

---

<div class="columns">
<div>

# Phase 4: Comparative Study
### Weeks 7–10

**Focus:** DRL vs. Velocity-based Generalized Approach (VGA).

- **Implementation:** Built VGA + UPL from scratch (no libraries).
- **Experimentation:** 
  - Ran DRL on VGA datasets.
  - Created metrics for trajectory realism.
- **Validation:** Tested adaptability in VGA-specific environments.

</div>
<div>

### Comparison Results

![width:100%](path/to/comparison_plot.png)

*Placeholder: Trajectory comparison plot (DRL vs VGA).*

</div>
</div>

---

# Phase 5: Consolidation & Evaluation
### Week 11

**Focus:** Final Analysis & Synthesis.

- **Trajectory Analysis:** Proved DRL generalizes well to scenarios not explicitly hard-coded.
- **Conclusion:** DRL offers a competitive alternative to classical force-based models (SFM) and geometric models (VGA).
- **Deliverable:** Robust GitHub repository with:
    - DRL & VGA implementations.
    - Comparative visualization tools.

---

# Technical Skills Acquired

| Domain | Skills & Technologies |
| :--- | :--- |
| **Algorithms** | PPO, MAPPO, CTDE, LSTM, CNN, VGA, UPL, Social Force Model (SFM) |
| **Development** | Python, PyTorch, Ray-Casting, Environment Design, Custom Metrics |
| **Infrastructure** | GPU Optimization, Parallel Computing, Jülich HPC, Debugging Sync Errors |
| **Research** | Paper Implementation, Comparative Analysis, Trajectory Visualization |

---

# Thank You

### Questions?

*Internship Final Presentation*
*[Your Name]*
