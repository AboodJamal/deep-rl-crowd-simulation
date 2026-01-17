================================================================================
FOLDER: special_comparison_with_realExperiment
================================================================================

WHAT IS THIS?
-------------
Comparison between VGA Stochastic and DRL-VGA against REAL EXPERIMENT results
from the VGA paper on SPECIFIC trials.

TARGET EXPERIMENTS:
-------------------
- MOSP A: Row 183, Exp #184
- MOSP B: Row 23, Exp #24
- MOSP C: Row 153, Exp #154
- MOSP D: Row 160, Exp #161

HOW IT WORKS:
-------------
1. Loads specific trial data (start, goal, obstacles) from VGA dataset
2. Runs VGA Stochastic planner 100 times
3. Runs DRL-VGA model 100 times
4. Clusters similar trajectories
5. Calculates percentage for each path variant
6. Creates visualizations matching paper Figure 7 style

OUTPUT:
-------
- results/MOSP_A_exp184/
  - vga_stochastic_paths.png
  - drl_vga_paths.png
  - combined_comparison.png
- results/MOSP_B_exp24/
- results/MOSP_C_exp154/
- results/MOSP_D_exp161/
- results/comparison_results.json

HOW TO RUN:
-----------
python compare_with_experiment.py

RELATED TO:
-----------
- VGA Paper Figure 7 (real experiment path distributions)
- professional_comparison.py (general comparison across all trials)
- generate_stochastic_visualizations.py (VGA stochastic)

CREATED: January 2026
================================================================================
