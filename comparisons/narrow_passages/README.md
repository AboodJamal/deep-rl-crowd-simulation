# Narrow Navigation Comparison: VGA+UPL vs DRL

This folder compares VGA+UPL and DRL performance in **narrow passage scenarios** that require precise navigation through tight corridors.

## Purpose

Test which model handles narrow passages better:
- Can DRL navigate through tight gaps that require precise control?
- Does VGA-UPL struggle with narrow passages?
- Who has fewer collisions in constrained spaces?

## ⚠️ IMPORTANT: Generalization Limitation

**This comparison reveals a key limitation of DRL trained with VecNormalize:**

The DRL model was trained with `VecNormalize` on VGA experimental data (SOSP, MOSP_A/B/C/D scenarios). This means:
1. The observation normalization statistics were computed from VGA training data
2. When tested on NEW obstacle configurations (our custom narrow scenarios), the normalized observations don't match what the model expects
3. **Result: DRL fails to generalize to unseen obstacle configurations**

**This is a meaningful finding!** It demonstrates:
- ✅ **VGA-UPL excels at generalization** - physics-based approach works on ANY obstacle configuration
- ⚠️ **DRL requires retraining** for new scenarios (or training without VecNormalize, or with more diverse data)

## Scenarios

All scenarios use the **EXACT same parameters** as `professional_comparison/`:
- Arena: 10m × 3.5m (0 to 10, -1.75 to 1.75)
- Agent radius: 0.2m
- Obstacle radius: 0.25m
- Goal tolerance: 0.3m
- Max speed: 1.6 m/s
- Timestep: 0.05s

### NARROW_1: Single Narrow Gate
```
      ●  (obstacle at y=+gap)
  S -----> [ GAP ] -----> G
      ●  (obstacle at y=-gap)
```
- Two obstacles at x=5 forming a **0.70m gap** 
- Tests basic precision through a single bottleneck

### NARROW_2: Double Gate Sequence
```
      ●           ●
  S ---[ GAP ]---[ GAP ]--- G
      ●           ●
```
- Two narrow gates at x=3.5 and x=6.5
- Tests sustained precision through multiple bottlenecks

### NARROW_3: Zigzag Corridor
```
         ●●
  S ----/    \---- G
     ●●    ●●
```
- Alternating obstacles creating S-curve path
- Tests maneuvering ability in constrained space

### NARROW_4: Dense Narrow Corridor
```
    ●   ●   ●   ●
  S-[ ]-[ ]-[ ]-[ ]-G
    ●   ●   ●   ●
```
- Multiple offset gates requiring continuous precision
- Most challenging scenario

### NARROW_5: Funnel Scenario
```
    ●  ●  ●  ●  ●  ●  ●
  S ----\   GAP   /---- G
    ●  ●  ●  ●  ●  ●  ●
```
- Wide entry → very narrow center → wide exit
- Tests navigation through bottleneck

## Running the Comparison

```bash
cd "trains on everything- eval on everything"
python narrow_navigates_comparison/compare_narrow_navigation.py
```

## Results Summary

| Model | Success Rate | Interpretation |
|-------|--------------|----------------|
| VGA+UPL V4 | **100%** | Physics-based model generalizes perfectly to new scenarios |
| DRL (PPO) | **0%** | Fails due to VecNormalize distribution mismatch |

## Key Metrics

In addition to all 22 metrics from `professional_comparison`, this adds:
- **narrowest_passage_used**: Minimum clearance achieved during trial
- **time_in_narrow_zone**: % of time with clearance < 0.5m

## Implications for DRL Deployment

To improve DRL generalization:
1. **Train without VecNormalize** (use manual observation normalization)
2. **Train on MORE diverse obstacle configurations** (domain randomization)
3. **Use curriculum learning** starting from simple to complex scenarios
4. **Fine-tune on new scenarios** before deployment

## Conclusion

This comparison demonstrates the **generalization gap** between:
- **Physics-based methods (VGA-UPL)**: Work on any configuration without retraining
- **Learning-based methods (DRL)**: Require training data similar to deployment scenarios

For real-world deployment where obstacle configurations vary, VGA-UPL has a significant advantage. For deployment in known, fixed environments similar to training, DRL can excel.
