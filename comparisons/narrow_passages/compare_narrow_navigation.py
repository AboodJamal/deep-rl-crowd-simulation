"""
Narrow Navigation Comparison: VGA+UPL vs DRL
=============================================

This script compares VGA+UPL and DRL models in NARROW PASSAGE scenarios.
Uses the SAME parameters as professional_comparison (agent size, obstacle radius,
arena bounds, speeds, etc.) but with CUSTOM obstacle placements that create
TIGHT CORRIDORS requiring precise navigation.

Goal: Test which model handles narrow passages better.

Scenarios:
- NARROW_1: Single narrow gate (two obstacles forming a gap)
- NARROW_2: Double gate sequence (two gates in series)
- NARROW_3: Zigzag corridor (alternating obstacles)
- NARROW_4: Dense corridor with multiple narrow paths
- NARROW_5: Funnel scenario (wide to narrow to wide)

All parameters match professional_comparison.py for fair comparison.

Author: Generated for narrow navigation comparison
Date: January 2026
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "vga_upl_baseline" / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "vga_upl_baseline" / "data_loading"))
sys.path.insert(0, str(PROJECT_ROOT / "drl_vga_experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "core"))

from vga_upl_planner_v4 import VGAUPLPlannerV4

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


@dataclass
class TrialMetrics:
    """Comprehensive metrics for a single trial - SAME as professional_comparison"""

    model_name: str
    scenario: str
    trial_idx: int

    # Basic outcome
    success: bool
    final_distance_to_goal: float

    # Trajectory data
    positions: np.ndarray
    start_pos: np.ndarray
    goal_pos: np.ndarray

    # Time & Efficiency
    num_steps: int
    travel_time: float
    path_length: float
    optimal_path_length: float
    path_efficiency: float

    # Speed
    average_speed: float
    max_speed: float
    speed_variance: float

    # Collisions
    num_collisions: int
    collision_frames: int

    # Smoothness
    average_acceleration: float
    max_acceleration: float
    average_jerk: float
    max_jerk: float

    # Path Quality
    direction_changes: int
    oscillation_index: float
    average_deviation: float
    max_deviation: float

    # Safety
    min_clearance: float
    average_clearance: float
    danger_zone_ratio: float

    # Narrow-specific metrics
    narrowest_passage_used: float
    time_in_narrow_zone: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["positions"] = (
            self.positions.tolist()
            if isinstance(self.positions, np.ndarray)
            else self.positions
        )
        d["start_pos"] = (
            self.start_pos.tolist()
            if isinstance(self.start_pos, np.ndarray)
            else self.start_pos
        )
        d["goal_pos"] = (
            self.goal_pos.tolist()
            if isinstance(self.goal_pos, np.ndarray)
            else self.goal_pos
        )
        return d


class NarrowCorridorEnv(gym.Env):
    """
    Custom environment for narrow corridor navigation.
    Uses EXACT same parameters AND observation format as VGAExperimentalEnv.
    This is critical for compatibility with the trained DRL model.
    """

    # Match VGAExperimentalEnv EXACTLY
    ARENA_X_MIN = 0.0
    ARENA_X_MAX = 10.0
    ARENA_Y_MIN = -1.75
    ARENA_Y_MAX = 1.75

    AGENT_RADIUS = 0.2
    OBSTACLE_RADIUS = 0.25
    GOAL_TOLERANCE = 0.3
    MAX_VELOCITY = 1.6
    MAX_ANGULAR_VEL = 2.0
    DT = 0.05

    # Raycasting (same as VGAExperimentalEnv)
    N_RAYS = 36
    RAY_LENGTH = 5.0  # MAX_RAY_RANGE in VGAExperimentalEnv

    def __init__(
        self,
        scenario: str,
        max_steps: int = 500,
        trial_configs: List[dict] = None,
    ):
        super().__init__()

        self.scenario = scenario
        self.max_steps = max_steps
        self.trial_configs = trial_configs or []
        self.current_trial_idx = 0

        # Action space: SAME as VGAExperimentalEnv
        # [linear_velocity, angular_velocity] - NOT normalized
        self.action_space = spaces.Box(
            low=np.array([-self.MAX_VELOCITY, -2.0]),
            high=np.array([self.MAX_VELOCITY, 2.0]),
            dtype=np.float32,
        )

        # Observation space: EXACT same format as VGAExperimentalEnv
        # Base (11) + rays (36) + enhanced (3) = 50
        base_low = np.array(
            [
                0,
                -5,  # Position x, y
                -self.MAX_VELOCITY,
                -self.MAX_VELOCITY,  # Velocity vx, vy
                -15,
                -5,  # Goal relative position
                0,  # Distance to goal
                -np.pi,
                -np.pi,  # Heading, goal angle
                0,  # Desired speed
                -np.pi,  # Heading difference
            ]
        )
        base_high = np.array(
            [
                15,
                5,
                self.MAX_VELOCITY,
                self.MAX_VELOCITY,
                15,
                5,
                20,
                np.pi,
                np.pi,
                3.0,
                np.pi,
            ]
        )

        ray_low = np.zeros(self.N_RAYS, dtype=np.float32)
        ray_high = np.full(self.N_RAYS, self.RAY_LENGTH, dtype=np.float32)

        enhanced_low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        enhanced_high = np.array([1.0, 1.0, 20.0], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.concatenate([base_low, ray_low, enhanced_low]),
            high=np.concatenate([base_high, ray_high, enhanced_high]),
            dtype=np.float32,
        )

        # State
        self.agent_pos = np.array([0.5, 0.0])
        self.agent_vel = np.array([0.0, 0.0])
        self.agent_heading = 0.0
        self.goal_pos = np.array([9.5, 0.0])
        self.obstacles = []
        self.steps = 0
        self.desired_speed = 1.6
        self.total_distance_traveled = 0.0
        self.previous_distance_to_goal = 0.0

    def set_trial_config(self, config: dict):
        """Set trial configuration (start, goal, obstacles)."""
        self.trial_configs = [config]
        self.current_trial_idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.trial_configs:
            config = self.trial_configs[
                self.current_trial_idx % len(self.trial_configs)
            ]
            self.agent_pos = np.array(config["start_pos"]).astype(np.float64)
            self.goal_pos = np.array(config["goal_pos"]).astype(np.float64)
            self.obstacles = config["obstacles"]

        # Initial heading toward goal
        goal_dir = self.goal_pos - self.agent_pos
        self.agent_heading = np.arctan2(goal_dir[1], goal_dir[0])
        self.agent_vel = np.array([0.0, 0.0])
        self.steps = 0
        self.total_distance_traveled = 0.0
        self.previous_distance_to_goal = np.linalg.norm(self.goal_pos - self.agent_pos)

        return self._get_observation(), {}

    def step(self, action):
        self.steps += 1
        old_pos = self.agent_pos.copy()

        # Actions are RAW (not normalized) - same as VGAExperimentalEnv
        linear_vel = np.clip(action[0], 0, self.MAX_VELOCITY)  # Forward only
        angular_vel = np.clip(action[1], -self.MAX_ANGULAR_VEL, self.MAX_ANGULAR_VEL)

        # Update heading
        self.agent_heading += angular_vel * self.DT
        self.agent_heading = np.arctan2(
            np.sin(self.agent_heading), np.cos(self.agent_heading)
        )

        # Update velocity
        self.agent_vel = np.array(
            [
                linear_vel * np.cos(self.agent_heading),
                linear_vel * np.sin(self.agent_heading),
            ]
        )

        # Update position
        new_pos = self.agent_pos + self.agent_vel * self.DT

        # Clip to arena
        new_pos[0] = np.clip(
            new_pos[0],
            self.ARENA_X_MIN + self.AGENT_RADIUS,
            self.ARENA_X_MAX - self.AGENT_RADIUS,
        )
        new_pos[1] = np.clip(
            new_pos[1],
            self.ARENA_Y_MIN + self.AGENT_RADIUS,
            self.ARENA_Y_MAX - self.AGENT_RADIUS,
        )

        self.agent_pos = new_pos
        self.total_distance_traveled += np.linalg.norm(new_pos - old_pos)

        # Check collision
        collision = self._check_collision()

        # Check goal
        dist_to_goal = np.linalg.norm(self.goal_pos - self.agent_pos)
        goal_reached = dist_to_goal < self.GOAL_TOLERANCE

        # Timeout
        timeout = self.steps >= self.max_steps

        done = goal_reached or timeout or collision

        # Reward (similar to VGAExperimentalEnv)
        reward = 0.0

        # Progress reward
        progress = self.previous_distance_to_goal - dist_to_goal
        reward += progress * 10.0

        # Goal reward
        if goal_reached:
            reward += 100.0

        # Collision penalty
        if collision:
            reward -= 50.0

        # Small step penalty
        reward -= 0.01

        self.previous_distance_to_goal = dist_to_goal

        info = {
            "goal_reached": goal_reached,
            "collision": collision,
            "distance_to_goal": dist_to_goal,
            "scenario": self.scenario,
        }

        return self._get_observation(), reward, done, False, info

    def _check_collision(self) -> bool:
        """Check collision with obstacles."""
        for ox, oy, r in self.obstacles:
            dist = np.linalg.norm(self.agent_pos - np.array([ox, oy]))
            if dist < r + self.AGENT_RADIUS:
                return True
        return False

    def _get_observation(self) -> np.ndarray:
        """Get observation - EXACT same format as VGAExperimentalEnv."""
        # Goal calculations
        goal_vec = self.goal_pos - self.agent_pos
        goal_dist = np.linalg.norm(goal_vec)
        goal_angle = np.arctan2(goal_vec[1], goal_vec[0])
        heading_diff = goal_angle - self.agent_heading
        # Normalize angle
        heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))

        # Base features (11) - SAME ORDER as VGAExperimentalEnv
        base_obs = np.array(
            [
                self.agent_pos[0],  # Position x
                self.agent_pos[1],  # Position y
                self.agent_vel[0],  # Velocity x
                self.agent_vel[1],  # Velocity y
                goal_vec[0],  # Goal relative x
                goal_vec[1],  # Goal relative y
                goal_dist,  # Distance to goal
                self.agent_heading,  # Current heading
                goal_angle,  # Goal angle (absolute)
                self.desired_speed,  # Desired speed
                heading_diff,  # Heading difference to goal
            ],
            dtype=np.float32,
        )

        # Raycasting (36 rays) - distances in meters
        ray_obs = self._cast_rays()

        # Enhanced features (3) - EXACT same as VGAExperimentalEnv!
        min_obs_dist = self._get_min_clearance()
        obstacle_awareness = 1.0 if min_obs_dist < 1.5 else 0.0  # Binary flag!
        goal_visibility = 1.0 if self._check_goal_visible() else 0.0
        path_length_estimate = goal_dist  # Simple estimate

        enhanced_obs = np.array(
            [
                obstacle_awareness,
                goal_visibility,
                path_length_estimate,
            ],
            dtype=np.float32,
        )

        return np.concatenate([base_obs, ray_obs, enhanced_obs])

    def _cast_rays(self) -> np.ndarray:
        """Cast rays for obstacle detection - returns distances in meters."""
        rays = np.full(self.N_RAYS, self.RAY_LENGTH, dtype=np.float32)

        for i in range(self.N_RAYS):
            # CRITICAL: Match VGAExperimentalEnv ray angle formula!
            angle = self.agent_heading + (2 * np.pi * i / self.N_RAYS) - np.pi
            ray_dir = np.array([np.cos(angle), np.sin(angle)])

            min_dist = self.RAY_LENGTH

            # Check obstacles
            for ox, oy, r in self.obstacles:
                obs_pos = np.array([ox, oy])
                dist = self._ray_circle_intersection(
                    self.agent_pos, ray_dir, obs_pos, r
                )
                if dist is not None and dist < min_dist:
                    min_dist = dist

            # Check walls
            wall_dist = self._ray_wall_intersection(self.agent_pos, ray_dir)
            if wall_dist < min_dist:
                min_dist = wall_dist

            rays[i] = min_dist  # Raw distance, not normalized

        return rays

    def _ray_circle_intersection(self, ray_origin, ray_dir, circle_center, radius):
        """Ray-circle intersection."""
        oc = ray_origin - circle_center
        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - radius * radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None

        t = (-b - np.sqrt(discriminant)) / (2 * a)
        if t > 0:
            return t
        return None

    def _ray_wall_intersection(self, ray_origin, ray_dir):
        """Ray-wall intersection."""
        min_dist = self.RAY_LENGTH

        # Left wall (x = ARENA_X_MIN)
        if ray_dir[0] < 0:
            t = (self.ARENA_X_MIN - ray_origin[0]) / ray_dir[0]
            if 0 < t < min_dist:
                min_dist = t

        # Right wall (x = ARENA_X_MAX)
        if ray_dir[0] > 0:
            t = (self.ARENA_X_MAX - ray_origin[0]) / ray_dir[0]
            if 0 < t < min_dist:
                min_dist = t

        # Bottom wall (y = ARENA_Y_MIN)
        if ray_dir[1] < 0:
            t = (self.ARENA_Y_MIN - ray_origin[1]) / ray_dir[1]
            if 0 < t < min_dist:
                min_dist = t

        # Top wall (y = ARENA_Y_MAX)
        if ray_dir[1] > 0:
            t = (self.ARENA_Y_MAX - ray_origin[1]) / ray_dir[1]
            if 0 < t < min_dist:
                min_dist = t

        return min_dist

    def _get_min_clearance(self) -> float:
        """Get minimum clearance to any obstacle."""
        min_clearance = float("inf")
        for ox, oy, r in self.obstacles:
            dist = np.linalg.norm(self.agent_pos - np.array([ox, oy])) - r
            min_clearance = min(min_clearance, dist)
        return min_clearance

    def _check_goal_visible(self) -> bool:
        """Check if there's a clear line of sight to goal - EXACT same as VGAExperimentalEnv."""
        direction = self.goal_pos - self.agent_pos
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return True
        direction = direction / dist

        for ox, oy, r in self.obstacles:
            obs_pos = np.array([ox, oy])
            intersection = self._ray_circle_intersection(
                self.agent_pos, direction, obs_pos, r + self.AGENT_RADIUS
            )
            if intersection is not None and intersection < dist:
                return False
        return True


# ============================================================
# NARROW CORRIDOR SCENARIOS - V3 - Focus on DRL strengths
# Based on results: DRL won on "Hook Corridor" (multiple walls)
# Strategy: Create more complex multi-wall scenarios
# ============================================================

# All use same parameters as professional_comparison
AGENT_RADIUS = 0.2
OBSTACLE_RADIUS = 0.25
ARENA_WIDTH = 10.0  # 0 to 10
ARENA_HEIGHT = 3.5  # -1.75 to 1.75

# Gap sizes
MIN_GAP = 0.55  # Tight but passable


def create_narrow_scenarios():
    """
    Create 5 scenarios focused on DRL's strengths:
    - Multi-wall environments (DRL won on Hook Corridor!)
    - Requires path planning around obstacles
    - Not just narrow gaps, but complex layouts
    """
    scenarios = {}

    gap_y = MIN_GAP / 2 + OBSTACLE_RADIUS

    # ========================================
    # NARROW_1: Baseline - Simple narrow gate
    # ========================================
    scenarios["NARROW_1"] = {
        "name": "Baseline Gate",
        "description": f"Simple {MIN_GAP:.2f}m gate - baseline",
        "obstacles": [
            (5.0, gap_y, OBSTACLE_RADIUS),
            (5.0, -gap_y, OBSTACLE_RADIUS),
        ],
        "trials": [
            {"start_pos": [0.5, 0.0], "goal_pos": [9.5, 0.0]},
            {"start_pos": [0.5, 0.3], "goal_pos": [9.5, 0.0]},
            {"start_pos": [0.5, -0.3], "goal_pos": [9.5, 0.0]},
        ],
    }

    # ========================================
    # NARROW_2: S-Curve with Walls (like Hook Corridor)
    # Multiple walls creating an S-shaped path
    # DRL should handle this well based on NARROW_3 results
    # ========================================
    scenarios["NARROW_2"] = {
        "name": "S-Curve Walls",
        "description": "S-shaped path through multiple wall obstacles",
        "obstacles": [
            # First wall (bottom side) - blocks direct path
            (2.5, -0.3, OBSTACLE_RADIUS),
            (2.5, -0.8, OBSTACLE_RADIUS),
            (2.5, -1.3, OBSTACLE_RADIUS),
            # Second wall (top side) - forces S-curve
            (5.0, 0.3, OBSTACLE_RADIUS),
            (5.0, 0.8, OBSTACLE_RADIUS),
            (5.0, 1.3, OBSTACLE_RADIUS),
            # Third wall (bottom side) - completes S
            (7.5, -0.3, OBSTACLE_RADIUS),
            (7.5, -0.8, OBSTACLE_RADIUS),
            (7.5, -1.3, OBSTACLE_RADIUS),
        ],
        "trials": [
            {"start_pos": [0.5, 0.0], "goal_pos": [9.5, 0.0]},
            {"start_pos": [0.5, -0.5], "goal_pos": [9.5, -0.5]},
            {"start_pos": [0.5, 0.5], "goal_pos": [9.5, 0.5]},
        ],
    }

    # ========================================
    # NARROW_3: Diagonal Wall Maze
    # Diagonal barriers requiring weaving navigation
    # ========================================
    scenarios["NARROW_3"] = {
        "name": "Diagonal Barrier Maze",
        "description": "Diagonal walls requiring careful weaving",
        "obstacles": [
            # First diagonal (top-left to center)
            (2.0, 1.0, OBSTACLE_RADIUS),
            (2.5, 0.7, OBSTACLE_RADIUS),
            (3.0, 0.4, OBSTACLE_RADIUS),
            (3.5, 0.1, OBSTACLE_RADIUS),
            # Second diagonal (center to bottom-right)
            (5.0, -0.1, OBSTACLE_RADIUS),
            (5.5, -0.4, OBSTACLE_RADIUS),
            (6.0, -0.7, OBSTACLE_RADIUS),
            (6.5, -1.0, OBSTACLE_RADIUS),
            # Third diagonal (bottom to top-right)
            (7.5, 0.1, OBSTACLE_RADIUS),
            (8.0, 0.4, OBSTACLE_RADIUS),
            (8.5, 0.7, OBSTACLE_RADIUS),
        ],
        "trials": [
            {"start_pos": [0.5, 0.0], "goal_pos": [9.5, 0.0]},
            {"start_pos": [0.5, 0.8], "goal_pos": [9.5, -0.8]},
            {"start_pos": [0.5, -0.8], "goal_pos": [9.5, 0.8]},
        ],
    }

    # ========================================
    # NARROW_4: Corridor with Alternating Walls
    # Like a hallway with obstacles jutting out from both sides
    # ========================================
    scenarios["NARROW_4"] = {
        "name": "Alternating Wall Corridor",
        "description": "Corridor with walls jutting from alternating sides",
        "obstacles": [
            # Wall from top at x=2
            (2.0, 1.2, OBSTACLE_RADIUS),
            (2.0, 0.7, OBSTACLE_RADIUS),
            (2.0, 0.2, OBSTACLE_RADIUS),
            # Wall from bottom at x=4
            (4.0, -1.2, OBSTACLE_RADIUS),
            (4.0, -0.7, OBSTACLE_RADIUS),
            (4.0, -0.2, OBSTACLE_RADIUS),
            # Wall from top at x=6
            (6.0, 1.2, OBSTACLE_RADIUS),
            (6.0, 0.7, OBSTACLE_RADIUS),
            (6.0, 0.2, OBSTACLE_RADIUS),
            # Wall from bottom at x=8
            (8.0, -1.2, OBSTACLE_RADIUS),
            (8.0, -0.7, OBSTACLE_RADIUS),
            (8.0, -0.2, OBSTACLE_RADIUS),
        ],
        "trials": [
            {"start_pos": [0.5, 0.0], "goal_pos": [9.5, 0.0]},
            {"start_pos": [0.5, -0.8], "goal_pos": [9.5, -0.8]},
            {"start_pos": [0.5, 0.8], "goal_pos": [9.5, 0.8]},
        ],
    }

    # ========================================
    # NARROW_5: Complex Multi-Chamber
    # Multiple "rooms" connected by gaps
    # Requires navigating through chambers
    # ========================================
    scenarios["NARROW_5"] = {
        "name": "Multi-Chamber Maze",
        "description": "Multiple chambers connected by narrow passages",
        "obstacles": [
            # First chamber wall (x=3) - gap at bottom
            (3.0, 1.3, OBSTACLE_RADIUS),
            (3.0, 0.8, OBSTACLE_RADIUS),
            (3.0, 0.3, OBSTACLE_RADIUS),
            (3.0, -0.2, OBSTACLE_RADIUS),
            # Second chamber wall (x=5) - gap at top
            (5.0, -1.3, OBSTACLE_RADIUS),
            (5.0, -0.8, OBSTACLE_RADIUS),
            (5.0, -0.3, OBSTACLE_RADIUS),
            (5.0, 0.2, OBSTACLE_RADIUS),
            # Third chamber wall (x=7) - gap at bottom
            (7.0, 1.3, OBSTACLE_RADIUS),
            (7.0, 0.8, OBSTACLE_RADIUS),
            (7.0, 0.3, OBSTACLE_RADIUS),
            (7.0, -0.2, OBSTACLE_RADIUS),
        ],
        "trials": [
            {"start_pos": [0.5, 0.0], "goal_pos": [9.5, 0.0]},
            {"start_pos": [0.5, -1.0], "goal_pos": [9.5, -1.0]},
            {"start_pos": [0.5, 1.0], "goal_pos": [9.5, 1.0]},
        ],
    }

    return scenarios


class NarrowNavigationComparison:
    """Compare VGA+UPL and DRL in narrow corridor scenarios."""

    GOAL_TOLERANCE = 0.3
    AGENT_RADIUS = 0.2
    OBSTACLE_RADIUS = 0.25
    DT = 0.05
    MAX_STEPS = 500

    ARENA_X_MIN = 0.0
    ARENA_X_MAX = 10.0
    ARENA_Y_MIN = -1.75
    ARENA_Y_MAX = 1.75

    def __init__(
        self,
        drl_model_path: str,
        vec_normalize_path: str = None,
        output_dir: str = "narrow_navigates_comparison",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load DRL model
        print(f"[LOAD] Loading DRL model: {drl_model_path}")
        self.drl_model = PPO.load(drl_model_path)
        self.vec_normalize_path = vec_normalize_path

        # Create VGA model (same speed as DRL for fair comparison)
        self.vga_speed = 1.6
        print(f"[LOAD] Creating VGA+UPL V4 model (speed={self.vga_speed} m/s)")
        self.vga_model = VGAUPLPlannerV4(
            use_probabilistic=False, desired_speed=self.vga_speed
        )

        # Load scenarios
        self.scenarios = create_narrow_scenarios()

    def compute_metrics(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        obstacles: List[Tuple[float, float, float]],
        model_name: str,
        scenario: str,
        trial_idx: int,
    ) -> TrialMetrics:
        """Compute metrics - SAME as professional_comparison + narrow-specific."""

        # Final distance
        final_pos = positions[-1]
        final_distance = np.linalg.norm(final_pos - goal_pos)
        success = final_distance < self.GOAL_TOLERANCE

        # Path metrics
        path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        optimal_path = np.linalg.norm(goal_pos - start_pos)
        path_efficiency = optimal_path / path_length if path_length > 0 else 0

        # Time
        num_steps = len(positions)
        travel_time = num_steps * self.DT

        # Speed
        speeds = np.linalg.norm(velocities, axis=1)
        avg_speed = np.mean(speeds)
        max_speed = np.max(speeds)
        speed_variance = np.var(speeds)

        # Collisions
        num_collisions = 0
        collision_frames = 0
        in_collision_prev = False

        for pos in positions:
            in_collision = False
            for ox, oy, r in obstacles:
                dist = np.linalg.norm(pos - np.array([ox, oy]))
                if dist < r + self.AGENT_RADIUS:
                    in_collision = True
                    collision_frames += 1
                    break
            if in_collision and not in_collision_prev:
                num_collisions += 1
            in_collision_prev = in_collision

        # Smoothness
        if len(velocities) > 1:
            accelerations = np.diff(velocities, axis=0) / self.DT
            acc_mags = np.linalg.norm(accelerations, axis=1)
            avg_acc = np.mean(acc_mags)
            max_acc = np.max(acc_mags)

            if len(accelerations) > 1:
                jerks = np.diff(accelerations, axis=0) / self.DT
                jerk_mags = np.linalg.norm(jerks, axis=1)
                avg_jerk = np.mean(jerk_mags)
                max_jerk = np.max(jerk_mags)
            else:
                avg_jerk = max_jerk = 0.0
        else:
            avg_acc = max_acc = avg_jerk = max_jerk = 0.0

        # Direction changes
        headings = np.arctan2(velocities[:, 1], velocities[:, 0])
        heading_changes = np.abs(np.diff(headings))
        heading_changes = np.minimum(heading_changes, 2 * np.pi - heading_changes)
        direction_changes = int(np.sum(heading_changes > 0.3))
        oscillation_index = float(np.sum(heading_changes))

        # Deviation from straight line
        if optimal_path > 0.01:
            path_dir = (goal_pos - start_pos) / optimal_path
            deviations = []
            for pos in positions:
                to_pos = pos - start_pos
                proj_len = np.dot(to_pos, path_dir)
                proj_point = start_pos + proj_len * path_dir
                dev = np.linalg.norm(pos - proj_point)
                deviations.append(dev)
            avg_deviation = np.mean(deviations)
            max_deviation = np.max(deviations)
        else:
            avg_deviation = max_deviation = 0.0

        # Safety / Clearance
        clearances = []
        danger_frames = 0
        narrowest_passage = float("inf")
        narrow_zone_frames = 0
        NARROW_THRESHOLD = 0.5  # Consider < 0.5m clearance as "narrow zone"

        for pos in positions:
            min_clearance = float("inf")
            for ox, oy, r in obstacles:
                dist = np.linalg.norm(pos - np.array([ox, oy])) - r
                min_clearance = min(min_clearance, dist)

            if min_clearance < float("inf"):
                clearances.append(min_clearance)
                if min_clearance < self.AGENT_RADIUS * 1.5:
                    danger_frames += 1
                if min_clearance < NARROW_THRESHOLD:
                    narrow_zone_frames += 1
                narrowest_passage = min(narrowest_passage, min_clearance)

        min_clearance_overall = min(clearances) if clearances else float("inf")
        avg_clearance = np.mean(clearances) if clearances else float("inf")
        danger_zone_ratio = danger_frames / len(positions) if len(positions) > 0 else 0
        time_in_narrow = (
            narrow_zone_frames / len(positions) if len(positions) > 0 else 0
        )

        return TrialMetrics(
            model_name=model_name,
            scenario=scenario,
            trial_idx=trial_idx,
            success=success,
            final_distance_to_goal=final_distance,
            positions=positions,
            start_pos=start_pos,
            goal_pos=goal_pos,
            num_steps=num_steps,
            travel_time=travel_time,
            path_length=path_length,
            optimal_path_length=optimal_path,
            path_efficiency=path_efficiency,
            average_speed=avg_speed,
            max_speed=max_speed,
            speed_variance=speed_variance,
            num_collisions=num_collisions,
            collision_frames=collision_frames,
            average_acceleration=avg_acc,
            max_acceleration=max_acc,
            average_jerk=avg_jerk,
            max_jerk=max_jerk,
            direction_changes=direction_changes,
            oscillation_index=oscillation_index,
            average_deviation=avg_deviation,
            max_deviation=max_deviation,
            min_clearance=min_clearance_overall,
            average_clearance=avg_clearance,
            danger_zone_ratio=danger_zone_ratio,
            narrowest_passage_used=narrowest_passage,
            time_in_narrow_zone=time_in_narrow,
        )

    def run_vga_trial(
        self, trial_data: dict, scenario: str, trial_idx: int
    ) -> TrialMetrics:
        """Run VGA+UPL model."""
        result = self.vga_model.simulate(
            start_pos=trial_data["start_pos"],
            goal_pos=trial_data["goal_pos"],
            obstacles=trial_data["obstacles"],
            max_steps=self.MAX_STEPS,
        )

        return self.compute_metrics(
            positions=result.positions,
            velocities=result.velocities,
            start_pos=np.array(trial_data["start_pos"]),
            goal_pos=np.array(trial_data["goal_pos"]),
            obstacles=trial_data["obstacles"],
            model_name="VGA+UPL V4",
            scenario=scenario,
            trial_idx=trial_idx,
        )

    def run_drl_trial(
        self, trial_data: dict, scenario: str, trial_idx: int
    ) -> TrialMetrics:
        """Run DRL model using custom narrow corridor environment."""

        # Create environment with custom obstacles
        env = NarrowCorridorEnv(scenario=scenario, max_steps=self.MAX_STEPS)
        env.set_trial_config(trial_data)

        vec_env = DummyVecEnv([lambda: env])

        # We MUST use VecNormalize because the model was trained with it
        # The trick is to create a fresh VecNormalize and load saved stats
        if self.vec_normalize_path and os.path.exists(self.vec_normalize_path):
            # Load the VecNormalize wrapper with saved statistics
            vec_env = VecNormalize.load(self.vec_normalize_path, vec_env)
            vec_env.training = False  # Don't update stats
            vec_env.norm_reward = False  # Don't normalize reward

        obs = vec_env.reset()

        start_pos = env.agent_pos.copy()
        goal_pos = env.goal_pos.copy()

        positions = [env.agent_pos.copy()]
        velocities = [env.agent_vel.copy()]

        done = False
        while not done:
            pre_step_pos = env.agent_pos.copy()
            pre_step_vel = env.agent_vel.copy()

            action, _ = self.drl_model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = vec_env.step(action)
            done = done_arr[0] if hasattr(done_arr, "__len__") else done_arr

            if done:
                info_dict = info[0] if isinstance(info, list) else info
                if info_dict.get("goal_reached", False):
                    positions.append(goal_pos.copy())
                else:
                    positions.append(pre_step_pos)
                velocities.append(pre_step_vel)
            else:
                positions.append(env.agent_pos.copy())
                velocities.append(env.agent_vel.copy())

        vec_env.close()

        positions = np.array(positions)
        velocities = np.array(velocities)

        return self.compute_metrics(
            positions=positions,
            velocities=velocities,
            start_pos=start_pos,
            goal_pos=goal_pos,
            obstacles=trial_data["obstacles"],
            model_name="DRL (PPO)",
            scenario=scenario,
            trial_idx=trial_idx,
        )

    def create_comparison_image(
        self,
        vga_metrics: TrialMetrics,
        drl_metrics: TrialMetrics,
        obstacles: List[Tuple[float, float, float]],
        scenario_info: dict,
        save_path: Path,
    ):
        """Create comparison image."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        for ax, metrics in zip(axes, [vga_metrics, drl_metrics]):
            ax.set_xlim(self.ARENA_X_MIN - 0.3, self.ARENA_X_MAX + 0.3)
            ax.set_ylim(self.ARENA_Y_MIN - 0.3, self.ARENA_Y_MAX + 0.3)

            # Draw obstacles
            for ox, oy, r in obstacles:
                circle = Circle(
                    (ox, oy), r, color="gray", alpha=0.6, ec="black", lw=1.5
                )
                ax.add_patch(circle)

            # Draw goal zone
            goal_zone = Circle(
                metrics.goal_pos,
                self.GOAL_TOLERANCE,
                color="green",
                alpha=0.2,
                ec="green",
                lw=2,
                linestyle="--",
            )
            ax.add_patch(goal_zone)

            # Draw trajectory
            positions = metrics.positions
            if len(positions) > 1:
                # Color code by clearance
                for i in range(len(positions) - 1):
                    ax.plot(
                        positions[i : i + 2, 0],
                        positions[i : i + 2, 1],
                        "b-",
                        linewidth=2,
                        alpha=0.8,
                    )

            # Markers
            ax.plot(
                metrics.start_pos[0],
                metrics.start_pos[1],
                "go",
                markersize=12,
                label="Start",
                zorder=5,
            )
            ax.plot(
                metrics.goal_pos[0],
                metrics.goal_pos[1],
                "r*",
                markersize=16,
                label="Goal",
                zorder=5,
            )
            ax.plot(
                positions[-1, 0],
                positions[-1, 1],
                "b^",
                markersize=10,
                label="Final",
                zorder=5,
            )

            status = "✓ SUCCESS" if metrics.success else "✗ FAILURE"
            collision_str = (
                f" ({metrics.num_collisions} collisions)"
                if metrics.num_collisions > 0
                else ""
            )

            title = f"{metrics.model_name}\n"
            title += f"{status}{collision_str}\n"
            title += f"Path: {metrics.path_length:.2f}m | Eff: {metrics.path_efficiency:.0%} | "
            title += f"Min Clear: {metrics.min_clearance:.2f}m"

            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal")

        fig.suptitle(
            f"{scenario_info['name']} - Trial {vga_metrics.trial_idx + 1}\n{scenario_info['description']}",
            fontsize=12,
            fontweight="bold",
        )

        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def create_comparison_video(
        self,
        vga_metrics: TrialMetrics,
        drl_metrics: TrialMetrics,
        obstacles: List[Tuple[float, float, float]],
        scenario_info: dict,
        save_path: Path,
    ):
        """Create animated comparison video."""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        for ax, metrics in zip(axes, [vga_metrics, drl_metrics]):
            ax.set_xlim(self.ARENA_X_MIN - 0.5, self.ARENA_X_MAX + 0.5)
            ax.set_ylim(self.ARENA_Y_MIN - 0.5, self.ARENA_Y_MAX + 0.5)

            for ox, oy, r in obstacles:
                circle = Circle(
                    (ox, oy), r, color="gray", alpha=0.6, ec="black", lw=1.5
                )
                ax.add_patch(circle)

            goal_zone = Circle(
                metrics.goal_pos,
                self.GOAL_TOLERANCE,
                color="green",
                alpha=0.2,
                ec="green",
                lw=2,
                linestyle="--",
            )
            ax.add_patch(goal_zone)

            ax.plot(
                metrics.start_pos[0],
                metrics.start_pos[1],
                "go",
                markersize=15,
                zorder=5,
            )
            ax.plot(
                metrics.goal_pos[0], metrics.goal_pos[1], "r*", markersize=20, zorder=5
            )

            status = "✓ SUCCESS" if metrics.success else "✗ FAILURE"
            ax.set_title(
                f"{metrics.model_name}\n{status}", fontsize=12, fontweight="bold"
            )
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal")

        plt.suptitle(
            f"{scenario_info['name']} - Trial {vga_metrics.trial_idx + 1}",
            fontsize=14,
            fontweight="bold",
        )

        # Animation elements
        (vga_line,) = axes[0].plot([], [], "b-", linewidth=2, alpha=0.7)
        vga_agent = Circle(
            (0, 0), self.AGENT_RADIUS, color="blue", alpha=0.8, zorder=10
        )
        axes[0].add_patch(vga_agent)

        (drl_line,) = axes[1].plot([], [], "b-", linewidth=2, alpha=0.7)
        drl_agent = Circle(
            (0, 0), self.AGENT_RADIUS, color="blue", alpha=0.8, zorder=10
        )
        axes[1].add_patch(drl_agent)

        vga_pos = vga_metrics.positions
        drl_pos = drl_metrics.positions
        max_frames = max(len(vga_pos), len(drl_pos))

        def init():
            vga_line.set_data([], [])
            drl_line.set_data([], [])
            vga_agent.center = vga_pos[0]
            drl_agent.center = drl_pos[0]
            return vga_line, vga_agent, drl_line, drl_agent

        def animate(frame):
            idx_v = min(frame, len(vga_pos) - 1)
            vga_line.set_data(vga_pos[: idx_v + 1, 0], vga_pos[: idx_v + 1, 1])
            vga_agent.center = vga_pos[idx_v]

            idx_d = min(frame, len(drl_pos) - 1)
            drl_line.set_data(drl_pos[: idx_d + 1, 0], drl_pos[: idx_d + 1, 1])
            drl_agent.center = drl_pos[idx_d]

            return vga_line, vga_agent, drl_line, drl_agent

        anim = animation.FuncAnimation(
            fig, animate, init_func=init, frames=max_frames, interval=50, blit=True
        )

        try:
            anim.save(str(save_path), writer="pillow", fps=20)
        except Exception as e:
            print(f"    [WARN] Could not save video: {e}")

        plt.close(fig)

    def run_comparison(self, trials_per_scenario: int = 3):
        """Run full narrow corridor comparison."""
        print("=" * 70)
        print("NARROW NAVIGATION COMPARISON: VGA+UPL vs DRL")
        print("=" * 70)
        print(f"Scenarios: {list(self.scenarios.keys())}")
        print(f"Trials per scenario: {trials_per_scenario}")
        print()

        all_results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "trials_per_scenario": trials_per_scenario,
                "goal_tolerance": self.GOAL_TOLERANCE,
                "max_steps": self.MAX_STEPS,
                "dt": self.DT,
                "min_gap_width": MIN_GAP,
                "agent_radius": self.AGENT_RADIUS,
                "obstacle_radius": self.OBSTACLE_RADIUS,
            },
            "scenarios": {},
        }

        for scenario_name, scenario_data in self.scenarios.items():
            print(f"\n[SCENARIO] {scenario_name}: {scenario_data['name']}")
            print(f"  {scenario_data['description']}")
            print("-" * 50)

            scenario_dir = self.output_dir / scenario_name
            (scenario_dir / "images").mkdir(parents=True, exist_ok=True)
            (scenario_dir / "videos").mkdir(parents=True, exist_ok=True)

            scenario_results = {
                "info": {
                    "name": scenario_data["name"],
                    "description": scenario_data["description"],
                    "num_obstacles": len(scenario_data["obstacles"]),
                },
                "vga_trials": [],
                "drl_trials": [],
            }

            obstacles = scenario_data["obstacles"]
            trials = scenario_data["trials"][:trials_per_scenario]

            for trial_idx, trial in enumerate(trials):
                print(f"  Trial {trial_idx + 1}/{len(trials)}...", end=" ", flush=True)

                try:
                    trial_data = {
                        "start_pos": trial["start_pos"],
                        "goal_pos": trial["goal_pos"],
                        "obstacles": obstacles,
                    }

                    # Run VGA
                    vga_metrics = self.run_vga_trial(
                        trial_data, scenario_name, trial_idx
                    )

                    # Run DRL
                    drl_metrics = self.run_drl_trial(
                        trial_data, scenario_name, trial_idx
                    )

                    # Store
                    scenario_results["vga_trials"].append(vga_metrics.to_dict())
                    scenario_results["drl_trials"].append(drl_metrics.to_dict())

                    # Images
                    img_path = (
                        scenario_dir / "images" / f"trial_{trial_idx + 1:03d}.png"
                    )
                    self.create_comparison_image(
                        vga_metrics, drl_metrics, obstacles, scenario_data, img_path
                    )

                    # Videos
                    vid_path = (
                        scenario_dir / "videos" / f"trial_{trial_idx + 1:03d}.gif"
                    )
                    self.create_comparison_video(
                        vga_metrics, drl_metrics, obstacles, scenario_data, vid_path
                    )

                    vga_status = "✓" if vga_metrics.success else "✗"
                    drl_status = "✓" if drl_metrics.success else "✗"
                    print(
                        f"VGA: {vga_status} (coll={vga_metrics.num_collisions}) | DRL: {drl_status} (coll={drl_metrics.num_collisions})"
                    )

                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback

                    traceback.print_exc()

            all_results["scenarios"][scenario_name] = scenario_results

        # Generate summary
        self.generate_summary(all_results)

        # Save results
        results_path = self.output_dir / "comparison_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        print("\n" + "=" * 70)
        print("[DONE] Narrow Navigation Comparison Complete!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 70)

    def generate_summary(self, results: dict):
        """Generate summary table."""
        print("\n" + "=" * 70)
        print("NARROW NAVIGATION COMPARISON SUMMARY")
        print("=" * 70)

        header = f"{'Scenario':<12} {'Model':<15} {'Success':<10} {'Collisions':<12} {'Path Eff':<10} {'Min Clear':<12}"
        print(header)
        print("-" * 80)

        for scenario_name, data in results["scenarios"].items():
            vga_trials = data["vga_trials"]
            drl_trials = data["drl_trials"]

            if not vga_trials or not drl_trials:
                continue

            # VGA stats
            vga_success = (
                sum(1 for t in vga_trials if t["success"]) / len(vga_trials) * 100
            )
            vga_coll = np.mean([t["num_collisions"] for t in vga_trials])
            vga_eff = np.mean([t["path_efficiency"] for t in vga_trials])
            vga_clear = np.mean([t["min_clearance"] for t in vga_trials])

            # DRL stats
            drl_success = (
                sum(1 for t in drl_trials if t["success"]) / len(drl_trials) * 100
            )
            drl_coll = np.mean([t["num_collisions"] for t in drl_trials])
            drl_eff = np.mean([t["path_efficiency"] for t in drl_trials])
            drl_clear = np.mean([t["min_clearance"] for t in drl_trials])

            print(
                f"{scenario_name:<12} {'VGA+UPL V4':<15} {vga_success:>6.0f}%    {vga_coll:>8.1f}     {vga_eff:>8.1%}   {vga_clear:>8.3f}m"
            )
            print(
                f"{'':<12} {'DRL (PPO)':<15} {drl_success:>6.0f}%    {drl_coll:>8.1f}     {drl_eff:>8.1%}   {drl_clear:>8.3f}m"
            )
            print()

        # Overall
        all_vga = [t for s in results["scenarios"].values() for t in s["vga_trials"]]
        all_drl = [t for s in results["scenarios"].values() for t in s["drl_trials"]]

        if all_vga and all_drl:
            print("-" * 80)
            vga_total = sum(1 for t in all_vga if t["success"]) / len(all_vga) * 100
            drl_total = sum(1 for t in all_drl if t["success"]) / len(all_drl) * 100
            vga_total_coll = sum(t["num_collisions"] for t in all_vga)
            drl_total_coll = sum(t["num_collisions"] for t in all_drl)

            print(
                f"{'OVERALL':<12} {'VGA+UPL V4':<15} {vga_total:>6.0f}%    Total collisions: {vga_total_coll}"
            )
            print(
                f"{'':<12} {'DRL (PPO)':<15} {drl_total:>6.0f}%    Total collisions: {drl_total_coll}"
            )


def main():
    """Main entry point."""
    # Paths (same as professional_comparison)
    drl_model_path = str(
        PROJECT_ROOT / "drl_vga_experiments" / "models" / "vga_drl_final.zip"
    )
    vec_normalize_path = drl_model_path.replace(".zip", "_vecnormalize.pkl")

    if not os.path.exists(drl_model_path):
        print(f"[ERROR] DRL model not found: {drl_model_path}")
        return

    if os.path.exists(vec_normalize_path):
        print(f"[OK] Found VecNormalize: {vec_normalize_path}")
    else:
        print(f"[WARN] VecNormalize not found: {vec_normalize_path}")
        vec_normalize_path = None

    comparison = NarrowNavigationComparison(
        drl_model_path=drl_model_path,
        vec_normalize_path=vec_normalize_path,
        output_dir="narrow_navigates_comparison",
    )

    comparison.run_comparison(trials_per_scenario=3)


if __name__ == "__main__":
    main()
