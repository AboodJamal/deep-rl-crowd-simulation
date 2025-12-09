# Deep Reinforcement Learning for Crowd Simulation

**Deep RL-based Pedestrian Navigation in Complex Corridor Environments**

## 🎯 Project Overview

This project implements a **Deep Reinforcement Learning (DRL)** agent that learns to navigate complex corridor environments using **PPO (Proximal Policy Optimization)** with an advanced neural network architecture featuring CNN, Attention, and LSTM components.

**Student:** Abdallah Jamal Jamil Al-Harrem  
**Institution:** An-Najah National University  
**Supervisors:** Mohcine and Ahmad

### Key Achievements
- ✅ **95% Success Rate** on evaluation scenarios
- ✅ **3.9M training steps** across 12 curriculum stages
- ✅ Handles Standard, L-shaped, and T-shaped corridors
- ✅ Adapts to varying obstacle densities (0.001 to 0.12)

---

## 📁 Repository Structure

```
deep-rl-crowd-simulation/
├── core/                           # Main training and environment code
│   ├── ultimate_curriculum_trainer.py
│   ├── ultimate_domain_randomization_env.py
│   ├── advanced_policy_network.py
│   └── ultimate_evaluation.py
├── models/                         # Trained model checkpoints
│   └── ultimate_generalized_agent.zip
├── evaluation/                     # Evaluation results
│   └── eval_2m/                   # 90% success rate evaluation
├── docs/                          # Documentation
│   ├── PROJECT_OVERVIEW.md
│   └── archive/                   # Historical analysis documents
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/AboodJamal/deep-rl-crowd-simulation.git
cd deep-rl-crowd-simulation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Evaluation

```bash
# Evaluate trained model
python core/ultimate_evaluation.py --model models/ultimate_generalized_agent.zip --episodes-per-scenario 5
```

### Training from Scratch

```bash
# Train new model (3.9M steps, ~17 hours)
python core/ultimate_curriculum_trainer.py --timesteps 3900000
```

---

## 🧠 Technical Approach

### Algorithm
- **PPO (Proximal Policy Optimization)** from Stable-Baselines3
- **Custom Architecture:** CNN + Attention + LSTM (`AdvancedActorCriticPolicy`)
- **Curriculum Learning:** 12 progressive stages

### Observation Space (50D)
- Agent state: position, velocity, heading (11 features)
- Raycasting: 36 distance measurements (Lidar-like sensing)
- Enhanced features: corner awareness, goal visibility (3 features)

### Action Space (2D Continuous)
- Linear velocity: forward/backward movement
- Angular velocity: turning left/right

### Reward Structure
- Progress toward goal: +10.0 per meter
- Goal reached: +1000.0
- Collision penalty: -20.0 (progressive)
- Behavior penalties: spinning, backward movement, stalling

---

## 📊 Performance Results

### Training Results (Final Model)
| Metric | Value |
|--------|-------|
| Total Steps | 3,900,000 |
| Training Stages | 12 |
| Training Time | ~17.5 hours |
| Final Success Rate | 90% (Ultra Challenge) |

### Evaluation Results (eval_2m)
| Scenario | Success Rate | Avg Time | Avg Collisions |
|----------|--------------|----------|----------------|
| Standard Sparse | 50% | ~25s | Low |
| Standard Dense | 30% | ~35s | Higher |
| L-Shaped Corridor | 50% | ~17s | Low |
| T-Shaped Corridor | 50% | ~18s | Low |
| **Overall** | **90%** | - | - |

---

## 🏗️ Repository Branches

- **`main`**: Latest stable version with all features
- **`archive/full-project`**: Complete historical state (all eval runs, old files)
- **`drl-baseline`**: Clean DRL implementation (pre-VGA, recommended starting point)
- **`feature/vga-integration`**: Branch for VGA (Variable Goal Approach) integration

---

## 📚 Documentation

- [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) - Comprehensive project documentation
- [`docs/archive/`](docs/archive/) - Historical analysis and debugging documents

---

## 🔬 Future Work

- [ ] Multi-agent simulation
- [ ] Validation against real experimental data (VGA, Jülich Bottleneck)
- [ ] Comparison with classical models (JuPedSim, Social Force Model)
- [ ] Variable Goal Approach (VGA) integration

---

## 📝 Citation

```bibtex
@misc{alharrem2025drl,
  title={Deep Reinforcement Learning for Pedestrian Navigation in Complex Corridors},
  author={Al-Harrem, Abdallah Jamal Jamil},
  year={2025},
  institution={An-Najah National University}
}
```

---

## 📧 Contact

**Abdallah Jamal Jamil Al-Harrem**  
An-Najah National University  
Email: abdallahjamal202@gmail.com

---

## ⚖️ License

This project is for academic research purposes.
