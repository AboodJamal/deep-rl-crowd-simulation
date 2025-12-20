"""
Simple DRL Model Wrapper for Validation
========================================

Directly loads and runs trained DRL model on experimental scenarios.
No complex imports - just load model and run.
"""

import numpy as np
import sys
import os
from pathlib import Path
from typing import List, Dict
import torch

# Add paths
script_dir = Path(__file__).parent.absolute()
core_dir = script_dir.parent.parent / "core"
sys.path.insert(0, str(core_dir))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from ultimate_domain_randomization_env import UltimateDomainRandomizedEnv


class SimpleDRLModel:
    """
    Simple wrapper to run trained DRL model on validation scenarios.
    """

    def __init__(self, checkpoint_path: str, vecnorm_path: str = None):
        """
        Load DRL model.

        Args:
            checkpoint_path: Path to .zip checkpoint
            vecnorm_path: Path to .pkl vecnormalize file (optional)
        """
        print(f"Loading DRL model from: {checkpoint_path}")
        self.model = PPO.load(checkpoint_path, device="cpu")
        print("[OK] DRL model loaded!")

        self.vecnorm_path = vecnorm_path
        if self.vecnorm_path and os.path.exists(self.vecnorm_path):
            print(f"[OK] Found VecNormalize: {self.vecnorm_path}")
        else:
            print(f"Warning: VecNormalize not found at {self.vecnorm_path}")

        self.env = None
        self.vec_env = None

    def simulate(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        obstacles: List[Dict],
        max_steps: int = 1000,
    ):
        """
        Run DRL model on EXPERIMENTAL scenario with EXACT obstacles from data.

        Args:
            start_pos: [x, y] - from experimental data
            goal_pos: [x, y] - from experimental data
            obstacles: List of {'position': [x, y], 'radius': r} - from experimental data
            max_steps: Maximum steps

        Returns:
            SimulationResult-like dict
        """

        # Create environment - we'll override everything
        def make_env():
            env = UltimateDomainRandomizedEnv(
                difficulty_level="medium", allowed_shapes=["standard"]
            )
            return env

        self.vec_env = DummyVecEnv([make_env])

        # Load normalization if available
        if self.vecnorm_path and os.path.exists(self.vecnorm_path):
            self.vec_env = VecNormalize.load(self.vecnorm_path, self.vec_env)
            self.vec_env.training = False  # Important: don't update stats during eval
            self.vec_env.norm_reward = False

        # Access the underlying env
        self.env = self.vec_env.envs[0]

        # Reset environment first
        obs = self.vec_env.reset()

        # Override with EXPERIMENTAL scenario data
        self.env.agent_pos = start_pos.copy()
        self.env.goal_pos = goal_pos.copy()
        self.env.start_pos = start_pos.copy()
        self.env.current_vel = np.zeros(2)
        self.env.heading = 0.0

        # Convert circular obstacles to rectangles (DRL was trained with rectangles)
        self.env.obstacles = []
        for obs_dict in obstacles:
            pos = np.array(obs_dict["position"])
            radius = obs_dict["radius"]
            # Bounding box: [x, y, width, height]
            self.env.obstacles.append(
                [
                    pos[0] - radius,  # left edge
                    pos[1] - radius,  # bottom edge
                    radius * 2,  # width
                    radius * 2,  # height
                ]
            )

        # CRITICAL: Update corridor bounds to fit the scenario
        # Calculate bounds from start, goal, and obstacles
        all_x = [start_pos[0], goal_pos[0]] + [o["position"][0] for o in obstacles]
        all_y = [start_pos[1], goal_pos[1]] + [o["position"][1] for o in obstacles]

        margin = 2.0  # Add margin around scenario
        self.env.x_min = min(all_x) - margin
        self.env.x_max = max(all_x) + margin
        self.env.y_min = min(all_y) - margin
        self.env.y_max = max(all_y) + margin

        # Re-compute observation with new setup
        raw_obs = self.env._get_observation()

        # Normalize if using VecNormalize
        if isinstance(self.vec_env, VecNormalize):
            obs = self.vec_env.normalize_obs(raw_obs.reshape(1, -1))[0]
        else:
            obs = raw_obs

        # Run episode
        positions = [start_pos.copy()]
        velocities = [np.zeros(2)]
        timestamps = [0.0]

        done = False
        step = 0
        success = False
        collision = False

        while not done and step < max_steps:
            # Predict action
            action, _ = self.model.predict(obs, deterministic=True)

            # Ensure action is properly shaped for vec_env
            # model.predict returns (n_actions,) but vec_env.step expects (n_envs, n_actions)
            if action.ndim == 1:
                action = action.reshape(1, -1)

            # Step environment
            # vec_env.step returns arrays for batch of envs
            obs, reward, terminated, info = self.vec_env.step(action)
            # Extract single env results
            reward = reward[0]
            terminated = terminated[0]
            info = info[0]

            # Record state (from underlying env)
            positions.append(self.env.agent_pos.copy())
            velocities.append(self.env.current_vel.copy())
            timestamps.append(step * self.env.DT)

            # Check success
            dist_to_goal = np.linalg.norm(self.env.agent_pos - self.env.goal_pos)
            if dist_to_goal < 0.5:  # Goal tolerance
                success = True
                done = True

            # Check collision (from info or manual check)
            if info.get("collision", False):
                collision = True
                done = True  # Stop on collision

            # Check if terminated by environment
            if terminated:
                # Could be success, collision, or timeout
                done = True
                # Check reason
                if dist_to_goal < 0.5:
                    success = True

            step += 1

        # Cleanup to avoid memory leaks
        self.vec_env.close()

        # Return result
        class Result:
            def __init__(self):
                self.success = success
                self.positions = np.array(positions)
                self.velocities = np.array(velocities)
                self.timestamps = np.array(timestamps)

        return Result()


# Test function
if __name__ == "__main__":
    checkpoint = "../checkpoints/ultimate_stage12_3837248_steps.zip"

    model = SimpleDRLModel(checkpoint)

    # Test scenario
    result = model.simulate(
        start_pos=np.array([2.0, 5.0]),
        goal_pos=np.array([8.0, 5.0]),
        obstacles=[{"position": [5.0, 5.0], "radius": 0.5}],
        max_steps=500,
    )

    print(f"Success: {result.success}")
    print(f"Steps: {len(result.positions)}")
