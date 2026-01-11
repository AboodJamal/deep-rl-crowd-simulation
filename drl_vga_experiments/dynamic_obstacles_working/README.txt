================================================================================
FOLDER: dynamic_obstacles_working (V1)
================================================================================

WHAT IS THIS?
-------------
First working version of DRL training for dynamic obstacle avoidance.
Contains the trained model that achieves ~80% success rate on dynamic obstacles.

WHEN CREATED?
-------------
Before V2 - this was the initial successful dynamic obstacle training.

WHY MADE?
---------
To train a DRL agent that can navigate through a corridor with MOVING obstacles.
This was different from the original VGA experiments which used STATIC obstacles.

WHAT'S INSIDE?
--------------
- train_corridor_env.py   : Training script with environment and PPO training
- trained_model/          : Saved model files (final_model.zip, best_model.zip)
- videos/                 : Test videos showing agent performance

KEY PARAMETERS (V1):
--------------------
- Robot Speed: 1.34 m/s (same as human walking speed)
- Obstacles: 2-3 dynamic obstacles
- Obstacle Speed: 0.3-0.8 m/s
- Observation: 12 rays + goal info
- Training: ~2M steps

RESULTS:
--------
- Success Rate: ~80% on dynamic obstacles
- Compared to VGA-UPL: DRL 80% vs VGA 20%

RELATED TO:
-----------
- Superseded by: dynamic_obstacles_working_v2 (HARD-trained, better)
- Comparison in: dynamic_comparison/ folder

STATUS: ARCHIVED - Use V2 instead for better performance
================================================================================
