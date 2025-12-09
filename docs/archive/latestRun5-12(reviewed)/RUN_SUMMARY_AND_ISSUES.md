# Run Summary: latestRun5-12 (Reviewed)
## Single-Agent Deep RL Pedestrian Navigation

**Date**: November 2025  
**Status**: Completed - Under Review  
**Run ID**: Ultimate Training Stages 1-12

---

## 📊 PROJECT OVERVIEW

This is a **Single-Agent Deep Reinforcement Learning Navigation System** for **pedestrian simulation** in corridors. The agent was trained to navigate through:

| Environment Type | Description |
|-----------------|-------------|
| **Standard Corridors** | Straight hallways with obstacles |
| **L-Shaped Corridors** | Corridors with 90° turns |
| **T-Shaped Corridors** | Corridors with T-junctions (choice of left/right) |

---

## 🏆 FINAL EVALUATION RESULTS

From evaluation run in `eval_ultimate/`:

| Scenario | Success Rate | Episodes |
|----------|-------------|----------|
| **Standard Sparse** | **70%** (7/10) | 10 |
| **Standard Dense** | **60%** (6/10) | 10 |
| **T-Shaped Corridor** | **50%** (5/10) | 10 |
| **L-Shaped Corridor** | **30%** (3/10) | 10 |
| **OVERALL** | **52.5%** (21/40) | 40 |

**40 evaluation videos generated** in `eval_ultimate/` folder.

---

## 📈 TRAINING HISTORY

Total Training: **3.9 Million timesteps** across 12 curriculum stages.

| Stage | Training Type | Steps | Success Rate |
|-------|--------------|-------|--------------|
| 1 | Super Easy Standard | 300K | 27% |
| 2 | Standard Sparse | 300K | 42% |
| 3 | Standard Dense Easy | 300K | 14% |
| 4 | Standard Medium | 300K | **74%** ✅ |
| 5 | Standard Hard | 300K | **97%** 🔥 |
| 6 | L/T Super Easy | 300K | 15% |
| 7 | L/T Easy | 300K | 21% |
| 8 | L/T Medium | 300K | 11% |
| 9 | L/T Hard | 300K | 16% |
| 10 | All Mixed | 600K | 20% |
| 11 | Pattern Navigation | 300K | 41% |
| 12 | Ultra Challenge | 300K | **58%** ✅ |

---

## 🛠️ TECHNOLOGIES & TECHNIQUES USED

### Algorithm
- **PPO (Proximal Policy Optimization)** - State-of-the-art RL algorithm

### Architecture (2025 Research-Based)
```
Input (50 features)
├── Base Features (11): Position, velocity, heading, goal info
├── Raycasting (36): 360° obstacle detection (36 rays × 10°)
└── Enhanced Semantic (3): Corner awareness, goal visibility, path estimate
        ↓
    CNN (1D) → Learns spatial patterns from raycasting
        ↓
    Multi-Head Attention (4 heads) → Focuses on relevant features
        ↓
    LSTM (2 layers × 256 units) → Temporal memory
        ↓
    Policy + Value Heads → Actions & Value estimates
```

### Key Features Used
| Feature | What It Does |
|---------|-------------|
| **CNN for Raycasting** | Learns "narrow passage", "wide opening" patterns |
| **LSTM Memory** | Remembers past observations (important for corners) |
| **Multi-Head Attention** | Focuses on goal direction vs obstacles dynamically |
| **Adaptive Hyperparameters** | Learning rate, entropy adapt to performance |
| **Potential-Based Reward Shaping** | Rewards that preserve optimal policy |
| **Novelty Bonus** | Intrinsic motivation for exploration |
| **12-Stage Curriculum** | Gradual difficulty progression |
| **8 Parallel Environments** | 8x faster training |

### Technical Specifications
| Parameter | Value |
|-----------|-------|
| Algorithm | PPO |
| Learning Rate | 3e-4 (adaptive) |
| Gamma (discount) | 0.995 |
| Batch Size | 256 |
| Parallel Envs | 8 |
| Max Velocity | 1.4 m/s |
| Ray Count | 36 (360°) |
| Ray Range | 12m (20m for L/T) |
| Episode Length | 500-2000 steps |
| Total Training | 3.9M timesteps |

---

## 📁 FILES IN THIS ARCHIVE

```
📂 latestRun5-12(reviewed)/
├── 📂 checkpoints/          # Training checkpoints (stage 1-12)
├── 📂 curriculum_logs/      # Training logs and summaries
├── 📂 eval_ultimate/        # Evaluation results + 40 videos
├── 📂 models/               # Final trained models
├── 📂 runs/                 # TensorBoard logs
├── 📂 code_files/           # Snapshot of code used
│   ├── ultimate_curriculum_trainer.py
│   ├── ultimate_domain_randomization_env.py
│   ├── ultimate_evaluation.py
│   ├── advanced_policy_network.py
│   └── numpy_compat_fix.py
└── 📂 docs/                 # Important documentation
    ├── COMPLETE_TECHNICAL_REPORT_UPDATED.md
    ├── 2025_FEATURES_SUMMARY.md
    ├── TIER1_IMPLEMENTATION_GUIDE.md
    └── ... (other docs)
```

---

# 🚨 KNOWN ISSUES (NEED TO BE FIXED)

## Issue #1: CRITICAL - Agent Cannot See Through Turn Junctions (Blind Visibility)

### Problem Description
The L-shaped and T-shaped corridors are simulated by placing **two straight corridors facing each other**. The wall/barrier between them blocks the agent's raycasting vision completely.

**This is NOT a real turn simulation!** In real L-shaped corridors, there should be no barrier at the turn - just an open corner. Currently:

```
CURRENT (WRONG):                    SHOULD BE:
┌─────────────┐                     ┌─────────────┐
│             │                     │             │
│    GOAL     │                     │    GOAL     │
│      ↑      │                     │      ↑      │
│    ┌─┴──┐   │  ← BARRIER!         │             │
│    │WALL│   │  Agent can't        │             │
│    └────┘   │  see through!       │             │
├─────────────┤                     └────────┐    │
│             │                              │    │
│  AGENT  →   │                     │  AGENT │→   │
│             │                     │        │    │
└─────────────┘                     └────────┴────┘
```

### Impact
- Agent is **trained on blind visibility** - it cannot see the goal until it's at the junction
- This makes L/T shaped corridor navigation extremely difficult (30-50% success vs 70% for standard)
- The agent has to "blindly" navigate to a turn it cannot see around

### Possible Solutions (TO INVESTIGATE)
1. **Change environment geometry** - Remove the internal wall at junctions, make it a real open corner
2. **Use different library** for corridor generation (e.g., Shapely for proper polygon corridors)
3. **Implement corner "see-through"** - Allow rays to wrap around corners
4. **Add "corner preview"** - Give agent semantic info about what's around the corner

### Was Agent Trained on This?
**YES** - The agent was trained on this "blind" version. It learned to navigate without seeing around corners. This is fundamentally harder than real corridor navigation.

---

## Issue #2: Agent Hesitation

### Problem Description
The agent shows hesitation behavior - it pauses or slows down unnecessarily during navigation, especially:
- When approaching obstacles
- At corridor transitions
- When deciding direction

### Impact
- Slower navigation times
- Lower success rates (timeouts)
- Unrealistic pedestrian behavior

### Possible Causes
- Entropy coefficient too high (too much exploration)
- Conflicting reward signals
- LSTM uncertainty about next action
- Action smoothing (EMA) may be too aggressive

---

## Issue #3: CRITICAL - Agent Goes Dumb Near Goal (Rotation/Oscillation)

### Problem Description
When the agent reaches **very close to the goal** (within 1-2 meters), it sometimes:
1. Starts rotating in place
2. Moves **away** from the goal
3. Circles around itself
4. Fails to reach the goal despite being right next to it

**This is extremely frustrating** - the agent was RIGHT THERE and then failed!

### Example Behavior
```
Agent path near goal:
    GOAL ←── Agent was HERE (0.5m away)
       ↑
       │    Agent then did THIS:
       │         ↺ rotate
       │         → move away
       │         ↺ rotate more
       │         × TIMEOUT
```

### Possible Causes
1. **Reward oscillation** - Agent may have learned to "dance" near goal for small progress rewards
2. **Heading alignment issue** - Goal requires specific heading to "enter"
3. **Goal detection radius too small** - Agent overshoots and panics
4. **LSTM confusion** - Memory gets confused when goal is very close
5. **Corner navigation trauma** - Bad experiences at turns may have corrupted near-goal behavior

### This Might Be Related To
The "blind turn" issue (#1) may have caused the agent to develop bad habits that also affect goal approach behavior.

---

## Issue #4: Goal Placement in L/T Corridors

### Problem Description
In L-shaped and T-shaped corridors, the goal is sometimes placed **on the wall at the turn junction** instead of in the corridor.

### Impact
- Agent cannot physically reach the goal (wall collision)
- Impossible episodes that always fail
- Wasted training time on unsolvable scenarios

### Required Fix
**Goal should ONLY be placed**:
- In the middle of the corridor (safe distance from walls)
- At the END of the corridor (not at the turn)
- With clearance from all walls (at least agent radius + buffer)

### Example
```
BAD Goal Placement:              GOOD Goal Placement:
┌─────────┐                      ┌─────────┐
│         │                      │         │
│    ×    │ ← Goal ON wall!      │   GOAL  │ ← Goal in corridor
│  WALL   │   Unreachable!       │         │
├────┬────┘                      ├────┬────┘
│    │                           │    │
│    │                           │    │
└────┘                           └────┘
```

---

## 📋 PRIORITY ORDER FOR FIXES

| Priority | Issue | Difficulty | Impact |
|----------|-------|------------|--------|
| 🔴 HIGH | #1 - Blind visibility at turns | Hard | Would significantly improve L/T performance |
| 🔴 HIGH | #3 - Oscillation near goal | Medium | Would improve all scenarios |
| 🟡 MEDIUM | #4 - Goal placement | Easy | Quick fix for impossible episodes |
| 🟢 LOW | #2 - Hesitation | Medium | Quality of life improvement |

---

## 📌 CONCLUSION

**Overall Assessment**: The project achieved decent results for standard corridors (60-70%) but struggles significantly with L-shaped (30%) and T-shaped (50%) corridors.

**Root Cause**: The primary issue is the fundamental design of L/T corridors - they are simulated as two separate corridors with a barrier, not as real continuous L/T shaped spaces.

**Recommendation**: Before further training, fix Issue #1 (environment geometry) to create proper L/T shaped corridors without internal barriers. Then retrain the agent on the corrected environments.

---

**Document Created**: December 5, 2025  
**Author**: AI Assistant (based on project analysis)  
**Status**: Issues documented, awaiting fixes

