# Evaluation Results

This directory contains evaluation results from testing the trained DRL agent.

## Directory Structure

```
evaluation/
├── results/
│   └── eval_2m/              # 90% success rate evaluation (2M steps model)
│       ├── complete_results.json
│       ├── standard_sparse/
│       ├── standard_dense/
│       ├── l-shaped_corridor/
│       └── t-shaped_corridor/
└── README.md
```

## Latest Results (eval_2m)

**Model:** `models/ultimate_generalized_agent.zip` (trained to 2M steps)  
**Date:** December 6, 2025  
**Overall Success Rate:** 90.0% (18/20 episodes)

### Performance by Scenario

| Scenario | Success Rate | Episodes | Avg Time | Avg Collisions |
|----------|--------------|----------|----------|----------------|
| Standard Sparse | 50% | 10 | ~25s | Low |
| Standard Dense | 30% | 10 | ~35s | Moderate |
| L-Shaped Corridor | 50% | 10 | ~17s | Low |
| T-Shaped Corridor | 50% | 10 | ~18s | Low |

### Key Observations

**Strengths:**
- ✅ Excellent performance in L-shaped and T-shaped corridors
- ✅ Good navigation in sparse obstacle scenarios
- ✅ Low collision rates in geometric corridors

**Weaknesses:**
- ⚠️ Lower success in dense obstacle scenarios (30%)
- ⚠️ Agent sometimes takes "uncomfortable" narrow paths
- ⚠️ Can get stuck in very dense crowds

## Running New Evaluations

```bash
# Basic evaluation (5 episodes per scenario)
python core/ultimate_evaluation.py \
    --model models/ultimate_generalized_agent.zip \
    --output-dir evaluation/results/eval_custom \
    --episodes-per-scenario 5

# Extended evaluation (10 episodes per scenario)
python core/ultimate_evaluation.py \
    --model models/ultimate_generalized_agent.zip \
    --output-dir evaluation/results/eval_extended \
    --episodes-per-scenario 10
```

## Output Format

### complete_results.json
```json
[
  {
    "episode_id": 0,
    "scenario": "Standard Sparse",
    "success": true,
    "reward": 1087.73,
    "time": 27.3,
    "collisions": 0,
    "distance_traveled": 37.77,
    "final_distance_to_goal": 0.80,
    "avg_velocity": 1.38,
    "trajectory": [[x1, y1], [x2, y2], ...]
  },
  ...
]
```

### Video Files
- Format: MP4 (H.264)
- Resolution: 640x480 (configurable)
- FPS: 10
- Naming: `epNNN_[success|failure].mp4`

## Scenarios Tested

### Standard Sparse
- **Corridor:** 40m × 10m
- **Obstacle Density:** 0.001-0.005
- **Start:** (1.5, 5.0)
- **Goal:** (38.5, 5.0)

### Standard Dense  
- **Corridor:** 40m × 10m
- **Obstacle Density:** 0.04-0.08
- **Start:** (1.5, 5.0)
- **Goal:** (38.5, 5.0)

### L-Shaped Corridor
- **Shape:** Two perpendicular corridors
- **Dimensions:** 20m × 10m each segment
- **Corner:** 90-degree turn
- **Obstacle Density:** 0.015-0.04

### T-Shaped Corridor
- **Shape:** Three-way junction
- **Dimensions:** 15m × 10m segments
- **Junction:** T-intersection
- **Obstacle Density:** 0.015-0.04

## Metrics Explained

- **Success:** Agent reached goal within 1.0m radius
- **Reward:** Cumulative reward over episode
- **Time:** Episode duration in seconds
- **Collisions:** Number of obstacle/wall collisions
- **Distance Traveled:** Total path length
- **Final Distance to Goal:** Distance at episode end
- **Avg Velocity:** Mean velocity magnitude

## Historical Evaluations

See `docs/archive/` for historical evaluation runs and bug fixes:
- `eval_new` - Initial broken evaluation (8.3%)
- `eval_fixed` - Environment fix (50%)
- `eval_final` - Full test (65%)
- `eval_2m` - Current best (90%)

## Next Steps

- [ ] Validate against real experimental data (VGA, Jülich)
- [ ] Compare with classical models (JuPedSim, Social Force)
- [ ] Test multi-agent scenarios
- [ ] Improve dense crowd performance
