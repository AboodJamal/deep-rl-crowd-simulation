# 🎬 How to Generate Videos - Complete Guide

## ✅ **Currently Running**

Evaluation is **running in the background** right now!

**Progress:**

- ⏳ Loading model (2-3 minutes)
- ⏳ Running 40 episodes (8-10 minutes)
- ⏳ Generating videos (2-3 minutes)
- **Total time:** ~15 minutes

---

## 📊 **How to Monitor Progress**

### **Check if it's still running:**

```bash
ps aux | grep "python.*ultimate_evaluation" | grep -v grep
```

### **Check for new videos:**

```bash
# Count videos (should increase to 40)
find eval_ultimate -name "*.mp4" 2>/dev/null | wc -l

# List recent videos
ls -lht eval_ultimate/*/*.mp4 2>/dev/null | head -5

# Watch progress
watch -n 10 'find eval_ultimate -name "*.mp4" 2>/dev/null | wc -l'
```

### **Check results:**

```bash
# When complete, you'll see:
ls -lh eval_ultimate/complete_results.json
```

---

## 🎬 **Where Videos Will Be**

```
eval_ultimate/                          ← NEW videos here!
├── standard_sparse/
│   ├── ep000.mp4
│   ├── ep001.mp4
│   └── ... (10 total)
├── standard_dense/
│   ├── ep010.mp4
│   └── ... (10 total)
├── l-shaped_corridor/
│   ├── ep020.mp4
│   └── ... (10 total)
├── t-shaped_corridor/
│   ├── ep030.mp4
│   └── ... (10 total)
├── complete_results.json               ← All data
└── visualizer_data/
    ├── ep000_standard_sparse.json
    └── ... (40 JSON files for 3D viewer)
```

---

## 🎯 **What Each Video Shows**

**MP4 Videos** (matplotlib-based):

- Agent trajectory (blue cylinder)
- Obstacles (boxes)
- Goal marker
- Raycasting (colored rays)
- Collision indicators
- Metrics overlay

**3D Visualizer** (Three.js - ENHANCED!):

- **RED glowing obstacles** (MUCH more visible!)
- **HUGE green goal with spotlight**
- **Grid floor** (1m divisions)
- Interactive camera controls
- Real-time playback

---

## 🔧 **How to Run Manually (Anytime)**

### **Basic Evaluation:**

```bash
# Activate environment
source .venv/Scripts/activate

# Run evaluation (generates 40 videos)
python ultimate_evaluation.py
```

### **Evaluate Specific Model:**

Edit `ultimate_evaluation.py`, line ~90:

```python
# Change this line:
model_path = "models/ultimate_generalized_agent.zip"

# To evaluate a checkpoint:
model_path = "checkpoints/ultimate_stage6_1715520_steps.zip"
```

### **Evaluate Specific Scenarios Only:**

Edit `ultimate_evaluation.py`, line ~600:

```python
scenarios = [
    # "standard_sparse",     # Comment out to skip
    # "standard_dense",      # Comment out to skip
    "l-shaped_corridor",     # Only evaluate this
    # "t-shaped_corridor"    # Comment out to skip
]
```

### **Change Number of Episodes:**

Edit `ultimate_evaluation.py`, line ~610:

```python
episodes_per_scenario = 10  # Change to 5, 20, etc.
```

---

## 🎨 **How to Use 3D Visualizer**

Once videos are generated:

### **Step 1: Open Visualizer**

```bash
# Double-click in File Explorer:
visualizer_3d.html

# Or from terminal (opens in default browser):
start visualizer_3d.html   # Windows
open visualizer_3d.html     # Mac
xdg-open visualizer_3d.html # Linux
```

### **Step 2: Load Episode**

1. Click **"Load JSON"** button
2. Navigate to: `eval_ultimate/visualizer_data/`
3. Choose any episode (e.g., `ep020_l-shaped_corridor.json`)

### **Step 3: Controls**

- **▶️ Play/Pause** - Start/stop playback
- **⏮️ Reset** - Go back to start
- **⏩ Fast** - Speed up 3x
- **🎥 Camera** - Cycle: follow → top → free
- **Mouse:**
  - Left drag: Rotate
  - Right drag: Pan
  - Scroll: Zoom

### **Step 4: Observe**

- **RED obstacles** (impossible to miss!)
- **HUGE goal** with "EXIT" text
- **Grid floor** (see exact distances)
- **Raycasting** (what agent sees)
- **Metrics** (distance, speed, time)

---

## 📈 **What to Look For in Videos**

### **L-Shaped Failures (ep020-029):**

Watch for:

- ❌ Agent reaches corner, stops
- ❌ Can't "see" around corner
- ❌ Oscillates back and forth
- ❌ Times out without finding goal
- 💡 **This confirms corner blindness issue!**

### **Standard Successes (ep000-009):**

Watch for:

- ✅ Smooth navigation
- ✅ Good obstacle avoidance
- ✅ Direct path to goal
- 💡 **Agent learned straight corridors well!**

### **Dense Failures (ep010-019):**

Watch for:

- ❌ Multiple collisions
- ❌ Gets stuck between obstacles
- ❌ High speed near obstacles
- 💡 **Needs adaptive speed control!**

### **T-Shaped Failures (ep030-039):**

Watch for:

- ❌ Similar to L-shaped issues
- ❌ Can't handle junctions
- ❌ Wrong turn choices
- 💡 **Needs junction detection!**

---

## 🚀 **Advanced: Batch Video Analysis**

### **Compare Success vs Failure:**

```bash
# Find successful episodes
grep -l '"success": true' eval_ultimate/visualizer_data/*.json

# Find failed episodes
grep -l '"success": false' eval_ultimate/visualizer_data/*.json
```

### **Extract Statistics:**

```python
import json

# Load results
with open('eval_ultimate/complete_results.json') as f:
    results = json.load(f)

# Analyze
for episode in results:
    if not episode['success']:
        print(f"Episode {episode['episode_id']}: "
              f"Failed at step {episode['steps']}, "
              f"Collisions: {episode['collisions']}")
```

### **Create Comparison Video:**

Use video editing software to create side-by-side:

- Left: Success (ep000.mp4)
- Right: Failure (ep020.mp4)

---

## 🎯 **Quick Reference Commands**

```bash
# Generate videos
python ultimate_evaluation.py

# Monitor progress
watch -n 10 'find eval_ultimate -name "*.mp4" 2>/dev/null | wc -l'

# Check if running
ps aux | grep ultimate_evaluation | grep -v grep

# View results summary
python -c "import json; results = json.load(open('eval_ultimate/complete_results.json')); print(f'Success: {sum(1 for r in results if r[\"success\"])}/40')"

# Open 3D visualizer
start visualizer_3d.html

# List all videos
find eval_ultimate -name "*.mp4" -exec echo {} \;
```

---

## ⏰ **Typical Timeline**

| Step                           | Time              | What's Happening           |
| ------------------------------ | ----------------- | -------------------------- |
| Model Loading                  | 2-3 min           | Loading PPO + VecNormalize |
| Episode 1-10 (Standard Sparse) | 2-3 min           | Running + rendering        |
| Episode 11-20 (Standard Dense) | 2-3 min           | Running + rendering        |
| Episode 21-30 (L-Shaped)       | 2-3 min           | Running + rendering        |
| Episode 31-40 (T-Shaped)       | 2-3 min           | Running + rendering        |
| Saving Results                 | 30 sec            | Writing JSON files         |
| **Total**                | **~15 min** | Complete                   |

---

## 🎬 **Current Status**

**Right Now:**

- ✅ Evaluation running in background
- ⏳ Loading model (takes 2-3 minutes)
- ⏳ Will generate 40 videos
- ⏳ Expected completion: ~15 minutes

**To Check Progress:**

```bash
# See if running
ps aux | grep ultimate_evaluation

# Count videos (should increase)
find eval_ultimate -name "*.mp4" 2>/dev/null | wc -l
```

**When Complete:**

- Videos: `eval_ultimate/[scenario]/*.mp4`
- Data: `eval_ultimate/complete_results.json`
- 3D Data: `eval_ultimate/visualizer_data/*.json`

---

## ✅ **Summary**

**To Generate Videos:**

```bash
python ultimate_evaluation.py
```

**To View Videos:**

- Double-click `.mp4` files
- OR use `visualizer_3d.html` (ENHANCED! RED obstacles, HUGE goal)

**To Monitor:**

```bash
find eval_ultimate -name "*.mp4" 2>/dev/null | wc -l
```

**Current:** Evaluation running, ~15 minutes to complete! 🎬
