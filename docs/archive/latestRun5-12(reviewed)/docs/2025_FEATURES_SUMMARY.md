# 2025 Features We're Using - Brief Summary

## ✅ **What We're Using (2025 State-of-the-Art)**

### **1. Enhanced Raycasting with Corner Awareness** ⭐
- **What:** Raycasting that skips internal junction walls
- **Why 2025:** Modern approach to handle non-convex geometries
- **Benefit:** Agent can see around corners (like human vision)

### **2. Semantic Observation Features** ⭐
- **Corner Awareness:** Detects when agent is at corner/junction
- **Goal Visibility:** Checks if goal is directly visible (line-of-sight)
- **Path Length Estimate:** Manhattan distance for L/T shapes
- **Why 2025:** Semantic features > raw sensor data (research shows better performance)

### **3. Potential-Based Reward Shaping (PBRS)**
- **What:** Reward shaping that preserves optimal policy
- **Why 2025:** Proven to not alter optimal policies (theoretical guarantee)
- **Benefit:** Safe reward engineering

### **4. Intrinsic Motivation (Novelty Bonus)**
- **What:** Rewards for exploring new states
- **Why 2025:** Addresses sparse reward problem
- **Benefit:** Better exploration, avoids local optima

### **5. Adaptive Hyperparameters**
- **What:** Learning rate, entropy, batch size adjust per training stage
- **Why 2025:** Research shows fixed hyperparameters are suboptimal
- **Benefit:** Better sample efficiency

### **6. Zone of Proximal Development Curriculum**
- **What:** Curriculum where tasks are just beyond current capability
- **Why 2025:** Proven to accelerate learning
- **Benefit:** Faster training, better generalization

### **7. Scaled Episode Length**
- **What:** Episode length scales with difficulty (500-2000 steps)
- **Why 2025:** Matches task complexity
- **Benefit:** Appropriate time budget for each difficulty

---

## ❌ **What We're NOT Using (But Could Add)**

### **1. Learned Representations (CNN/ViT)**
- **Current:** Fixed raycasting
- **2025 Alternative:** CNN or Vision Transformer to learn spatial features
- **Impact:** Better generalization, but requires more compute

### **2. Successor Representations**
- **Current:** Standard value function
- **2025 Alternative:** Decouple dynamics from rewards
- **Impact:** Better transfer learning, but more complex

### **3. Memory/Attention (LSTM/Transformer)**
- **Current:** MLP policy (no memory)
- **2025 Alternative:** LSTM or Transformer for temporal context
- **Impact:** Better for L/T shapes (remembers past turns)

---

## 📊 **Current Architecture (2025 Hybrid)**

```
Observation (50 values):
├── Base (11): Position, velocity, goal, heading, etc.
├── Raycasting (36): Distance to obstacles (FIXED: sees around corners)
└── Enhanced (3): Corner awareness, goal visibility, path estimate ⭐ NEW

Policy: MLP (Multi-Layer Perceptron)
Algorithm: PPO with adaptive hyperparameters
Reward: PBRS + Intrinsic Motivation
Curriculum: Zone of Proximal Development
```

---

## 🎯 **Bottom Line**

**What Makes This "2025":**
1. ✅ **Semantic features** (not just raw sensors)
2. ✅ **Corner-aware raycasting** (handles non-convex geometries)
3. ✅ **Potential-based rewards** (theoretically sound)
4. ✅ **Intrinsic motivation** (addresses exploration)
5. ✅ **Adaptive hyperparameters** (self-tuning)
6. ✅ **Research-based curriculum** (Zone of Proximal Development)

**What's Still "Classic":**
- ❌ Fixed feature extraction (raycasting, not learned)
- ❌ No memory/attention (MLP, not LSTM/Transformer)
- ❌ Standard value function (not Successor Representations)

**Verdict:** **Hybrid 2025 approach** - Modern reward shaping + curriculum + adaptive hyperparameters, but classic perception (raycasting). Good balance of performance and simplicity!

