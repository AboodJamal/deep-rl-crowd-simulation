# Code Flow: Training → Evaluation

## Overview
This project trains a PPO agent to navigate various corridor types using curriculum learning, then evaluates it on specific scenarios.

---

## 🟢 PHASE 1: TRAINING (`ultimate_curriculum_trainer.py`)

### Entry Point
```bash
python ultimate_curriculum_trainer.py --timesteps 1000000
```

### Step-by-Step Training Flow

#### 1. **Initialization** (lines 102-151)
- **Function**: `train_ultimate_curriculum()`
- Creates directories: `models/`, `checkpoints/`, `curriculum_logs/`
- Initializes Weights & Biases (W&B) for logging
- Defines curriculum stages (5 stages total)

#### 2. **Curriculum Stages** (lines 173-179)
Each stage progressively increases difficulty:

```python
Stage 1: "Standard Sparse"    → 150k steps  → easy, standard only
Stage 2: "Standard Dense"      → 200k steps  → medium, standard only
Stage 3: "L/T Easy"            → 200k steps  → easy, lshaped + tshaped
Stage 4: "L/T Medium"           → 250k steps  → medium, lshaped + tshaped
Stage 5: "All Mixed (Std+L/T)" → 250k steps  → mixed, all 3 shapes
```

#### 3. **For Each Stage** (lines 186-300)

**a) Environment Creation** (lines 196-204)
```python
def make_env():
    env = UltimateDomainRandomizedEnv(
        difficulty_level=stage['config']['difficulty'],
        allowed_shapes=stage['config']['shapes']
    )
    env = Monitor(env)  # Wraps for statistics
    return env

env = DummyVecEnv([make_env])  # Vectorized (parallel) environment wrapper
env = VecNormalize(env, ...)    # Normalizes observations & rewards
```
- **Creates**: `UltimateDomainRandomizedEnv` instance
- **Wraps**: `Monitor` → `DummyVecEnv` → `VecNormalize`
- **Purpose**: Provides normalized observations and rewards for stable learning

**b) Model Creation/Continuation** (lines 211-234)
```python
if model is None:
    # First stage: Create NEW PPO model
    model = PPO("MlpPolicy", env, learning_rate=2.5e-4, ...)
else:
    # Later stages: Continue from previous stage
    model.set_env(env)  # Update environment wrapper
```
- **First stage**: Creates new PPO model with neural network
- **Later stages**: Continues training same model (transfer learning)

**c) Training Loop** (lines 247-256)
```python
model.learn(
    total_timesteps=stage['timesteps'],
    callback=[...],  # Progress tracking
    reset_num_timesteps=False  # Keep global step counter
)
```

**What happens inside `model.learn()`:**
1. Agent interacts with environment via `env.step(action)`
2. Collects experiences (observations, actions, rewards)
3. Updates neural network using PPO algorithm
4. Callbacks log progress to W&B

**d) Save Intermediate Model** (lines 292-296)
```python
stage_save_path = save_path.replace(".zip", f"_stage{stage_idx+1}.zip")
model.save(stage_save_path)  # Saves PPO model weights
env.save(stage_save_path.replace(".zip", "_vecnormalize.pkl"))  # Saves normalization stats
```

#### 4. **Final Save** (lines 302-318)
```python
model.save(save_path)  # Final model: models/ultimate_generalized_agent.zip
vec_norm_final.save(save_path.replace(".zip", "_vecnormalize.pkl"))  # Normalization stats
```

**Files Created:**
- `models/ultimate_generalized_agent.zip` (final model)
- `models/ultimate_generalized_agent_vecnormalize.pkl` (normalization statistics)
- `models/ultimate_generalized_agent_stage1.zip` through `_stage5.zip` (intermediate checkpoints)
- `curriculum_logs/ultimate_training_summary.json` (training statistics)

---

## 🔵 PHASE 2: ENVIRONMENT INTERACTION (`ultimate_domain_randomization_env.py`)

### How Environment Works During Training

#### 1. **Episode Start: `reset()`** (lines 512-548)
```python
def reset(self, seed=None, options=None):
    # 1. Randomly select corridor type (standard, lshaped, tshaped, etc.)
    self._select_corridor_type()
    
    # 2. Define walkable regions based on corridor shape
    self._define_walkable_regions()
    
    # 3. Generate obstacles based on difficulty level
    self.obstacles = self._generate_obstacles()
    
    # 4. Place start/goal at corridor entrance/exit
    self.start_pos = ...
    self.goal_pos = ...
    
    # 5. Reset agent state
    self.agent_pos = self.start_pos.copy()
    self.agent_vel = np.zeros(2)
    self.agent_heading = atan2(dy, dx)  # Points toward goal
    
    # 6. Return initial observation
    return self._get_observation(), self._get_info()
```

**Observation Space** (11 values):
- Agent position (x, y)
- Agent velocity (vx, vy)
- Goal position (gx, gy)
- Distance to goal
- Angle to goal
- Agent heading
- Nearest obstacle distance
- Nearest obstacle angle

#### 2. **Agent Action: `step(action)`** (lines 604-670)
```python
def step(self, action):
    # action = [linear_velocity, angular_velocity]
    
    # 1. Process action (forward-only linear, filtered angular)
    desired_linear = clip(action[0], 0.0, MAX_VELOCITY)
    desired_angular_vel = low_pass_filter(action[1])
    
    # 2. Update heading
    self.agent_heading += desired_angular_vel * DT
    
    # 3. Compute velocity
    desired_vel = [desired_linear * cos(heading), desired_linear * sin(heading)]
    
    # 4. Apply acceleration limits
    self.agent_vel = update_with_acceleration_limit(desired_vel)
    
    # 5. Move agent
    new_pos = self.agent_pos + self.agent_vel * DT
    
    # 6. Check collisions (walls, obstacles)
    if collision:
        self.collision_count += 1
        self.agent_vel *= -0.3  # Bounce back
    else:
        self.agent_pos = new_pos
    
    # 7. Calculate reward
    reward = self._calculate_reward(collision, dist_to_goal, distance_moved)
    
    # 8. Check termination
    if dist_to_goal < AGENT_RADIUS * 3:
        terminated = True  # Success!
    if steps >= max_steps or collisions >= 50:
        truncated = True  # Timeout or too many collisions
    
    return observation, reward, terminated, truncated, info
```

#### 3. **Reward Calculation: `_calculate_reward()`** (lines 672-736)
```python
reward = 0.0

# Progress reward (moving toward goal)
reward += progress * 50.0

# Distance-based reward
reward += 10.0 / (1.0 + dist_to_goal)

# Forward movement bonus
reward += forward_component * 3.0

# Velocity efficiency
if 0.8 < vel < 1.3:
    reward += 1.0

# Penalties
if collision:
    reward -= 15.0 * (1.0 + collision_count / 10.0)  # Progressive penalty
    
if spinning_in_place:
    reward -= 5.0
    
if backward_movement:
    reward -= 2.0

if goal_reached:
    reward += 500.0  # Large success bonus
```

---

## 🟡 PHASE 3: EVALUATION (`ultimate_evaluation.py`)

### Entry Point
```bash
python ultimate_evaluation.py --model models/ultimate_generalized_agent.zip
```

### Step-by-Step Evaluation Flow

#### 1. **Initialization: `__init__()`** (lines 37-73)
```python
def __init__(self, model_path):
    # Load trained PPO model
    self.model = PPO.load(model_path)
    
    # Load VecNormalize (CRITICAL: must match training normalization!)
    self.vec_normalize = VecNormalize.load(vecnorm_path, dummy_env)
    self.vec_normalize.training = False  # Disable normalization updates
    
    self.all_results = []  # Store all episode results
```

**Why VecNormalize is critical:**
- Training normalizes observations/rewards based on training statistics
- Evaluation must use SAME normalization, otherwise model sees different input distribution
- Without it: agent receives unnormalized observations → poor performance

#### 2. **Scenario Definition: `run_complete_evaluation()`** (lines 312-392)
```python
scenarios = [
    ("Standard Sparse", lambda: UltimateDomainRandomizedEnv(
        difficulty_level="easy", allowed_shapes=["standard"]
    )),
    ("Standard Dense", lambda: UltimateDomainRandomizedEnv(
        difficulty_level="medium", allowed_shapes=["standard"]
    )),
    ("L-Shaped Corridor", lambda: UltimateDomainRandomizedEnv(
        difficulty_level="easy", allowed_shapes=["lshaped"]
    )),
    ("T-Shaped Corridor", lambda: UltimateDomainRandomizedEnv(
        difficulty_level="easy", allowed_shapes=["tshaped"]
    )),
]
```

#### 3. **For Each Scenario** (lines 336-356)
```python
for scenario_name, create_env_func in scenarios:
    for ep in range(episodes_per_scenario):  # Default: 10 episodes
        # Create fresh environment
        env = create_env_func()
        
        # Evaluate one episode
        result = self.evaluate_scenario(env, scenario_name, episode_counter)
        
        # Store results
        self.all_results.append(result)
        
        env.close()
```

#### 4. **Episode Evaluation: `evaluate_scenario()`** (lines 215-310)
```python
def evaluate_scenario(self, env, scenario_name, episode_id):
    # 1. Reset environment (with seed for reproducibility)
    obs, _ = env.reset(seed=episode_id)
    
    trajectory = []
    velocities = []
    headings = []
    episode_reward = 0
    
    # 2. Run episode until termination
    while not done:
        # Record state
        trajectory.append(env.agent_pos)
        velocities.append(norm(env.agent_vel))
        headings.append(env.agent_heading)
        
        # Normalize observation (CRITICAL!)
        if self.vec_normalize:
            obs_normalized = self.vec_normalize.normalize_obs(obs[None])[0]
        else:
            obs_normalized = obs
        
        # Get action from trained model
        action, _ = self.model.predict(obs_normalized, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated
    
    # 3. Compile results
    result = {
        "success": info.get("goal_reached", False),
        "reward": episode_reward,
        "time": info.get("time_elapsed", 0),
        "collisions": info.get("collisions", 0),
        "trajectory": trajectory,
        "obstacles": env.obstacles,
        "goal_pos": env.goal_pos,
        "start_pos": env.start_pos,
        "corridor_type": detect_corridor_type(env),
        ...
    }
    
    return result
```

**Key Points:**
- Uses `deterministic=True` for consistent evaluation (no exploration noise)
- Normalizes observations using training statistics
- Records full trajectory for video generation

#### 5. **Video Generation: `_create_video()`** (lines 425-595)
```python
def _create_video(self, result, filename):
    trajectory = result["trajectory"]
    obstacles = result["obstacles"]
    goal_pos = result["goal_pos"]
    corridor_type = result["corridor_type"]
    
    # Create frames for animation
    for step_idx in range(0, len(trajectory), frame_skip):
        ax.clear()
        
        # Draw corridor (based on corridor_type)
        if corridor_type == "lshaped":
            draw_lshaped_corridor(ax)
        elif corridor_type == "tshaped":
            draw_tshaped_corridor(ax)
        else:
            draw_standard_corridor(ax)
        
        # Draw obstacles
        for obs in obstacles:
            draw_obstacle(ax, obs)
        
        # Draw trajectory (path so far)
        ax.plot(trajectory[:step_idx, 0], trajectory[:step_idx, 1], "cyan")
        
        # Draw start/goal
        draw_circle(ax, start_pos, "blue")
        draw_circle(ax, goal_pos, "green", "EXIT")
        
        # Draw agent at current position
        agent_pos = trajectory[step_idx]
        draw_circle(ax, agent_pos, "blue")
        draw_arrow(ax, agent_pos, heading)  # Show direction
        
        # Save frame
        frames.append(ax_to_image(ax))
    
    # Save video using FFMpegWriter
    anim = FuncAnimation(fig, update, frames=len(frames))
    anim.save(filename, writer=FFMpegWriter(fps=10))
```

#### 6. **Results Summary** (lines 368-392)
```python
# Calculate statistics
scenario_results = {
    "Standard Sparse": {
        "success_rate": 50.0%,
        "avg_time": 49.4s,
        "avg_collisions": 26.8
    },
    ...
}

# Save to JSON
with open("eval_ultimate/complete_results.json", "w") as f:
    json.dump(self.all_results, f)
```

---

## 📊 Complete Flow Diagram

```
TRAINING PHASE
┌─────────────────────────────────────────────────────────┐
│ python ultimate_curriculum_trainer.py --timesteps 1M    │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 1: Standard Sparse (150k steps)                  │
│   ├─ Creates UltimateDomainRandomizedEnv               │
│   ├─ Wraps: Monitor → DummyVecEnv → VecNormalize        │
│   ├─ Creates PPO model (new)                           │
│   ├─ model.learn() → agent interacts with env           │
│   │    ├─ env.reset() → random corridor + obstacles    │
│   │    ├─ env.step(action) → move, collision, reward   │
│   │    └─ PPO updates neural network                    │
│   └─ Saves: model_stage1.zip + vecnormalize.pkl        │
└─────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 2-5: Progressive difficulty (200k-250k steps)    │
│   ├─ Continues SAME model (transfer learning)           │
│   ├─ Updates environment wrapper (set_env)               │
│   └─ Saves intermediate checkpoints                    │
└─────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│ Final Save:                                              │
│   ├─ models/ultimate_generalized_agent.zip              │
│   └─ models/ultimate_generalized_agent_vecnormalize.pkl │
└─────────────────────────────────────────────────────────┘


EVALUATION PHASE
┌─────────────────────────────────────────────────────────┐
│ python ultimate_evaluation.py --model <model_path>      │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│ Load Model & VecNormalize                               │
│   ├─ PPO.load(model_path) → neural network weights      │
│   └─ VecNormalize.load() → normalization statistics     │
└─────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│ For Each Scenario (Standard Sparse, Dense, L, T):       │
│   └─ For Each Episode (10 episodes):                   │
│       ├─ Create UltimateDomainRandomizedEnv            │
│       ├─ env.reset(seed) → deterministic setup          │
│       ├─ While not done:                                │
│       │   ├─ Normalize observation (vec_normalize)      │
│       │   ├─ model.predict(obs) → action              │
│       │   ├─ env.step(action) → move agent             │
│       │   └─ Record trajectory, reward, collisions    │
│       └─ Store result (success, time, collisions, ...) │
└─────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│ Generate Videos                                         │
│   ├─ For each episode result:                          │
│   │   ├─ Draw corridor (based on corridor_type)         │
│   │   ├─ Draw obstacles                                │
│   │   ├─ Draw trajectory (path)                          │
│   │   ├─ Draw agent (animated)                         │
│   │   └─ Save as MP4 video                             │
│   └─ Save: eval_ultimate/<scenario>/ep*.mp4            │
└─────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│ Save Results                                             │
│   ├─ eval_ultimate/complete_results.json                │
│   └─ Print summary statistics                           │
└─────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Concepts

### 1. **VecNormalize** (Observation/Reward Normalization)
- **Purpose**: Stabilizes training by normalizing inputs to neural network
- **Training**: Collects running mean/std of observations/rewards during training
- **Evaluation**: Uses SAME statistics to normalize observations
- **Critical**: If normalization doesn't match, model receives wrong input distribution → poor performance

### 2. **Curriculum Learning**
- **Progressive Difficulty**: Start easy (sparse obstacles, simple shapes) → hard (dense obstacles, complex shapes)
- **Transfer Learning**: Same model continues across stages, accumulating knowledge
- **Domain Randomization**: Each episode randomly selects corridor shape, obstacle placement, dimensions

### 3. **Environment Wrappers**
```
UltimateDomainRandomizedEnv
    ↓
Monitor (tracks episode statistics)
    ↓
DummyVecEnv (vectorized wrapper for parallel environments)
    ↓
VecNormalize (normalizes observations/rewards)
```

### 4. **Action Space**
- **Continuous**: `[linear_velocity, angular_velocity]`
- **Linear**: 0.0 to 1.4 m/s (forward-only in latest version)
- **Angular**: -1.8 to +1.8 rad/s (low-pass filtered for smoothness)

### 5. **Observation Space**
- 11 values: position, velocity, goal info, obstacle info, heading
- Normalized by VecNormalize before feeding to neural network

---

## 🐛 Common Issues & Solutions

### Issue 1: Evaluation fails (0% success)
**Cause**: VecNormalize mismatch
- Training uses `UltimateDomainRandomizedEnv` with VecNormalize
- Evaluation must use SAME environment class and normalization

**Solution**: Ensure `ultimate_evaluation.py` loads VecNormalize correctly

### Issue 2: Agent moves backward or spins
**Cause**: Reward function not penalizing backward movement enough
**Solution**: Enhanced reward penalties in `_calculate_reward()`

### Issue 3: Agent goes outside walls
**Cause**: Collision detection not strict enough
**Solution**: `_check_collision_at_position()` checks walls AND obstacles

### Issue 4: Obstacles/goal outside frame in videos
**Cause**: Video rendering limits not matching environment dimensions
**Solution**: Dynamic limits based on `walkable_regions` or `corridor_dims`

---

## 📁 File Dependencies

```
ultimate_curriculum_trainer.py
    ├─ imports: ultimate_domain_randomization_env.py
    └─ creates: models/*.zip, models/*_vecnormalize.pkl

ultimate_evaluation.py
    ├─ imports: ultimate_domain_randomization_env.py
    ├─ loads: models/ultimate_generalized_agent.zip
    ├─ loads: models/ultimate_generalized_agent_vecnormalize.pkl
    └─ creates: eval_ultimate/*.json, eval_ultimate/*/*.mp4

ultimate_domain_randomization_env.py
    └─ (standalone, no dependencies on other env files)
```

---

This is the complete flow from training to evaluation! 🎯

