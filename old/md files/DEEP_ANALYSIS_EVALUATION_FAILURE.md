# 🔍 DEEP ANALYSIS: Why Evaluation Results Are Bad

## 📊 TRAINING vs EVALUATION COMPARISON

### Training Results (During Training):
| Stage | Success Rate | Key Performance |
|-------|-------------|-----------------|
| Stage 1 (Standard Easy) | 66% | Density 0.02-0.05, clear paths |
| Stage 2 (Standard Medium) | 72% | Density 0.05-0.10, clear paths |
| Stage 3 (Shaped Easy) | 68% | L: 74%, T: 68%, U: 22% |
| Stage 4 (Shaped Hard) | 30% | L: 26%, T: 46%, U: 4.7% |
| Stage 5 (All Mixed) | 55% | L: 70%, T: 71%, Standard: 59% |
| Stage 6 (Ultra Challenge) | 27% | T: 67%, U: 1.3%, Multiroom: 0% |

### Evaluation Results:
| Scenario | Success Rate | Training Equivalent |
|----------|-------------|---------------------|
| Standard Sparse | 20% | Stage 1-2 (72%) |
| Standard Dense | 10% | Stage 2-4 (30-72%) |
| Narrow Passages | 0% | Never trained |
| Zigzag Path | 0% | Never trained |
| Clustered Obstacles | 0% | Never trained |
| L-Shaped | 60% | Stage 5 (70%) ✓ |
| T-Shaped | 10% | Stage 6 (67%) ❌ |
| U-Shaped | 0% | Stage 6 (1.3%) |
| Multi-Room | 0% | Stage 6 (0%) |

---

## 🐛 CRITICAL PROBLEMS IDENTIFIED

### Problem 1: ENVIRONMENT MISMATCH! 🔴

**Training Environment:** `UltimateDomainRandomizedEnv`
- Standard corridors: NO clear path restriction
- Obstacles: Random placement anywhere in walkable area
- Obstacle generation: Random OR clustered (50% chance)
- Obstacle size: 0.8-1.2 (variable)

**Evaluation Environment:** `RealisticPedCorridorEnv` (for Standard Sparse/Dense)
- Standard corridors: **2.5m clear path in center** (we fixed this to block path)
- BUT our fix creates **CUSTOM obstacles** that are:
  - Larger (1.0-1.5 instead of 0.8-1.2)
  - **Blocking the direct path** (training never saw this pattern!)
  - Fixed positions (not randomized like training)
  - Different patterns (staggered, zigzag-like)

**This is a MAJOR distribution shift!**

### Problem 2: T-Shaped MASSIVE Drop (67% → 10%) 🔴

**Training:** Used `UltimateDomainRandomizedEnv` (T-shaped)
- Success: 67% in Stage 6

**Evaluation:** Uses `TShapedCorridorEnv` (separate class!)
- Success: 10%
- **Different environment = different observation distributions!**

**Root Cause:** 
- Training env generates T-shapes with `UltimateDomainRandomizedEnv`
- Evaluation uses `TShapedCorridorEnv` (different implementation)
- Different obstacle generation, different dimensions
- Different observation scales!

### Problem 3: Custom Obstacles Too Hard 🔴

**Standard Sparse/Dense evaluation:**
- Creates 8-14 obstacles **blocking the path**
- Training saw: Random obstacles with clear paths or sparse random placement
- **Agent never learned to navigate around intentionally blocking obstacles**

### Problem 4: Multi-Room Was Never Learned 🔴

- Training: 0% success (193 episodes, all failed)
- Evaluation: 0% success (expected)
- **Multi-room is too hard for current curriculum**

### Problem 5: Curriculum Too Aggressive ⚠️

**Stage 4 (Shaped Hard):**
- Jumped from 68% (easy) to 30% (hard)
- **Difficulty spike too large!**
- Should have gradual increase

**Stage 6 (Ultra Challenge):**
- Multiroom + U-shaped + ultra density
- All extremely hard scenarios
- Success rate crashed to 27%

---

## ✅ WHAT WORKED

1. **L-Shaped: 60%** ✅
   - Training: 70% in Stage 5
   - Evaluation: 60% (reasonable drop, good generalization)

2. **Training Progress:**
   - Agent DID learn L/T shapes (70-71% in Stage 5)
   - Agent DID generalize to some extent

---

## 🔧 ROOT CAUSES

### 1. Evaluation Uses DIFFERENT Environments Than Training

**Training:**
```python
UltimateDomainRandomizedEnv(difficulty="easy", shapes=["standard"])
# Generates obstacles: random placement, clear paths possible
```

**Evaluation:**
```python
RealisticPedCorridorEnv(obstacle_density=0.03)  # Has clear path in center
# OR our custom obstacles that BLOCK the path
```

**Result:** Distribution shift = agent fails!

### 2. Custom Obstacles Are TOO Different From Training

**Training obstacles:**
- Random placement
- Size: 0.8-1.2
- Sometimes clustered
- **No intentional path blocking**

**Evaluation custom obstacles:**
- **Intentional path blocking**
- Larger (1.4-3.2)
- Fixed patterns (staggered, center-blocking)
- **Agent never saw this pattern!**

### 3. T-Shaped Environment Mismatch

**Training:** `UltimateDomainRandomizedEnv` with T-shape
**Evaluation:** `TShapedCorridorEnv` (separate class)

Different:
- Obstacle generation
- Dimensions
- Observation distributions
- Normalization stats

### 4. VecNormalize May Still Be Wrong

Even with fixes, if VecNormalize stats don't match evaluation environments, normalization is wrong.

---

## 💡 SOLUTIONS

### Solution 1: Make Evaluation Match Training (RECOMMENDED)

**Change Standard Sparse/Dense to use `UltimateDomainRandomizedEnv`:**

```python
("Standard Sparse", lambda: UltimateDomainRandomizedEnv(
    difficulty_level="easy",
    allowed_shapes=["standard"]
)),
("Standard Dense", lambda: UltimateDomainRandomizedEnv(
    difficulty_level="hard",
    allowed_shapes=["standard"]
)),
```

**Benefits:**
- Same environment as training
- Same obstacle generation
- Same observation distributions
- Should get much better results!

### Solution 2: Fix T-Shaped Evaluation

**Use `UltimateDomainRandomizedEnv` instead of `TShapedCorridorEnv`:**

```python
("T-Shaped Corridor", lambda: UltimateDomainRandomizedEnv(
    difficulty_level="medium",
    allowed_shapes=["tshaped"]
)),
```

### Solution 3: Remove/Adjust Multi-Room

**Options:**
1. Remove from evaluation (never worked)
2. Make it easier in training (start earlier, less density)
3. Accept 0% (it's extremely hard)

### Solution 4: Adjust Custom Obstacles

**Make them more similar to training:**
- Random placement instead of blocking
- Smaller obstacles (0.8-1.2)
- Less aggressive blocking
- Or just use `UltimateDomainRandomizedEnv` (Solution 1)

### Solution 5: Fix Curriculum

**Make it more gradual:**
- Stage 4: Medium difficulty instead of hard
- Stage 6: Don't include multiroom until it's learned
- Add intermediate stages

---

## 🎯 EXPECTED RESULTS AFTER FIXES

If we use `UltimateDomainRandomizedEnv` for evaluation:

| Scenario | Current | Expected After Fix |
|----------|---------|-------------------|
| Standard Sparse | 20% | 50-70% |
| Standard Dense | 10% | 30-50% |
| L-Shaped | 60% | 65-75% (maintain) |
| T-Shaped | 10% | 60-70% (match training) |
| U-Shaped | 0% | 5-15% (still hard) |
| Multi-Room | 0% | 0-10% (very hard) |

---

## 📝 RECOMMENDATIONS

### Immediate Fix:
1. **Change Standard Sparse/Dense to use `UltimateDomainRandomizedEnv`**
2. **Change T-Shaped to use `UltimateDomainRandomizedEnv`**
3. **Re-run evaluation**

### Longer Term:
1. **Gradual curriculum** (easier difficulty progression)
2. **More training on U-shapes** (they're too hard)
3. **Remove or simplify multi-room** (it's too extreme)
4. **Custom obstacles should match training patterns**

---

## ✅ CONCLUSION

**Main Issues:**
1. ❌ Evaluation uses different environments than training
2. ❌ Custom obstacles too different from training
3. ❌ T-Shaped uses different environment class
4. ❌ Curriculum too aggressive (difficulty jumps)
5. ❌ Multi-room too hard (never learned)

**Is Training Bad?**
- **NO** - Agent learned L/T shapes (70-71%)
- **YES** - Curriculum too aggressive for U/multiroom

**Is Evaluation Bad?**
- **YES** - Using wrong environments (distribution shift)
- **YES** - Custom obstacles too different

**Next Steps:**
1. Fix evaluation to use same environments as training
2. Re-run evaluation
3. Results should improve significantly!



