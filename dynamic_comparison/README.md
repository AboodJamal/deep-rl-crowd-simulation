# Dynamic Obstacle Comparison: DRL vs VGA-UPL

## Overview

This experiment compares the DRL agent (trained specifically for dynamic obstacles) against the VGA-UPL baseline on the same dynamic obstacle scenarios with **realistic collision physics** where agents cannot pass through obstacles.

## Environment Setup

- **Corridor**: 12m x 6m
- **Agent**: Radius 0.25m, starts at left (1.0, 3.0)
- **Goal**: Right side (11.0, 3.0)
- **Dynamic Obstacles**: 3 circular obstacles (radius 0.35m)
  - Located at x = 4.0, 6.5, 9.0
  - Move vertically with velocity 0.5-0.9 m/s
  - Bounce at corridor boundaries
- **Max Steps**: 400
- **⚠️ CRITICAL**: Collision with obstacles causes **immediate failure**

## Results Summary (With Proper Collision Physics)

### DRL Agent (Trained for Dynamic Obstacles)
- **Success Rate**: 80.0% (4/5 trials)
- **Avg Collisions**: 0.2 per trial
- **Avg Steps to Goal**: 68.2 steps
- **Performance**: Strong - successfully navigates most scenarios without collision

### VGA-UPL Baseline
- **Success Rate**: 40.0% (2/5 trials)  
- **Avg Collisions**: 0.6 per trial
- **Avg Steps to Goal**: 96.0 steps (for successful trials only)
- **Performance**: Poor - fails in 60% of trials due to collisions

## Key Findings

1. **DRL Vastly Outperforms VGA-UPL**: 
   - **2x better success rate** (80% vs 40%)
   - **3x fewer collisions** (0.2 vs 0.6)
   - **41% faster** when successful (68.2 vs 96.0 steps)

2. **Why DRL Wins**:
   - Trained to anticipate obstacle movement
   - Learns safe margins around moving obstacles
   - Optimizes trajectories to avoid future collision zones
   - Smooth, predictive navigation

3. **Why VGA-UPL Fails**:
   - Cannot predict obstacle movement (designed for static obstacles)
   - Reacts only to current obstacle positions
   - By the time it reacts, obstacle has already moved into collision path
   - No learning or adaptation to dynamic patterns

4. **Collision Analysis**:
   - **DRL**: Only 1 collision across 5 trials (20% failure rate)
   - **VGA-UPL**: 3 collisions across 5 trials (60% failure rate)
   - VGA-UPL's reactive approach is fundamentally insufficient for dynamic obstacles

## Video Analysis

The comparison videos (in `videos/`) show frame-by-frame behavior with **proper collision physics**:
- **Blue agent** (left): DRL - anticipatory avoidance, maintains safe distance
- **Purple agent** (right): VGA-UPL - reactive corrections, often too late
- **Red circles**: Dynamic obstacles moving vertically
- **Red X**: Marks collision point (when agent fails)

Key observations:
- DRL adjusts trajectory early to avoid predicted obstacle paths
- VGA-UPL often gets trapped or collides when obstacle moves into its path
- DRL maintains smooth velocity; VGA-UPL shows erratic corrections
- Failed trials clearly show agent stopping at collision point (cannot pass through)

## Files

- `compare_dynamic_obstacles.py` - Main comparison script with proper collision physics
- `videos/` - Side-by-side comparison videos for each trial
- `results/comparison_results.json` - Detailed numerical results

## Conclusion

With **realistic collision physics** where agents cannot pass through obstacles:

**DRL demonstrates clear superiority** for dynamic obstacle navigation:
- ✅ 2x better success rate (80% vs 40%)
- ✅ 3x fewer collisions (0.2 vs 0.6)
- ✅ 41% faster navigation
- ✅ Trained to predict and avoid moving obstacles

**VGA-UPL is inadequate** for dynamic environments:
- ❌ 60% failure rate due to collisions
- ❌ Reactive approach fails when obstacles move
- ❌ No prediction or learning capability
- ❌ Designed for static obstacles only

This validates the necessity of DRL training for dynamic obstacle environments. Classical reactive planning methods like VGA-UPL cannot handle the prediction and anticipation required for moving obstacles.
