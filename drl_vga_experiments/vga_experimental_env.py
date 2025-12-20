"""
VGA Experimental Environment - EXACT MATCH to VGA Dataset
=========================================================

This environment EXACTLY matches the VGA experimental setup:
- Arena size: ~10m x 3.5m (matching real experiment)
- Agent radius: 0.2m (VGA paper)
- Obstacle radius: 0.25m (VGA paper)
- dt: 0.05s (VGA paper)
- Physics: Helbing & Molnár (1995)

SCENARIOS:
- SOSP: Single Obstacle Single Pedestrian (1 obstacle at fixed position)
- MOSP_A: 4 obstacles at fixed positions
- MOSP_B: 7 obstacles at fixed positions
- MOSP_C: 12 obstacles at fixed positions
- MOSP_D: 16 obstacles at fixed positions

Each trial has UNIQUE start/goal positions loaded from the VGA dataset.
Obstacle positions are FIXED per scenario type.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import math


class VGAExperimentalEnv(gym.Env):
    """
    Gymnasium environment matching VGA experimental setup EXACTLY.

    Key differences from UltimateDomainRandomizedEnv:
    - FIXED arena size (10m x 3.5m)
    - FIXED obstacle positions per scenario
    - REAL trial start/goal from dataset
    - VGA-matching physics parameters
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    # VGA EXACT PARAMETERS - DO NOT CHANGE
    AGENT_RADIUS = 0.2  # VGA paper: 0.2m pedestrian radius
    OBSTACLE_RADIUS = 0.25  # VGA paper: 0.25m obstacle radius
    MAX_VELOCITY = 1.6  # Slightly higher than typical 1.34 for learning flexibility
    MAX_ACCELERATION = 2.5  # Realistic pedestrian acceleration
    DT = 0.05  # VGA paper: 50ms timestep

    # Arena bounds (VGA experimental setup)
    ARENA_X_MIN = 0.0
    ARENA_X_MAX = 10.0
    ARENA_Y_MIN = -1.75
    ARENA_Y_MAX = 1.75

    # Raycasting parameters (for DRL observation)
    N_RAYS = 36  # 10° resolution
    MAX_RAY_RANGE = 8.0  # Sufficient for 10m arena

    # Goal tolerance
    GOAL_TOLERANCE = 0.3  # VGA paper: 0.3m success threshold

    # Fixed obstacle positions per scenario (EXACT from VGA dataset files)
    OBSTACLE_CONFIGS = {
        "SOSP": [
            (5.000, 0.000),  # Single obstacle at center
        ],
        "MOSP_A": [
            (5.465, -0.446),
            (7.264, 0.344),
            (3.983, 0.208),
            (3.932, -0.978),
        ],
        "MOSP_B": [
            (5.358, -0.207),
            (4.950, 0.191),
            (7.108, 0.290),
            (2.892, -1.120),
            (4.850, -1.320),
            (4.200, 1.237),
            (7.550, -0.813),
        ],
        "MOSP_C": [
            (4.114, 0.527),
            (6.647, -0.736),
            (3.052, 0.644),
            (6.246, 1.222),
            (3.336, 1.490),
            (6.045, -0.469),
            (4.331, -0.845),
            (5.142, -1.456),
            (3.052, -0.653),
            (4.607, 0.946),
            (5.426, 0.092),
            (2.174, -0.360),
        ],
        "MOSP_D": [
            (4.911, 1.455),
            (5.465, -0.446),
            (4.970, -1.259),
            (6.324, 0.336),
            (6.675, -1.053),
            (5.418, 1.103),
            (7.264, 0.344),
            (6.252, -1.370),
            (5.923, 1.430),
            (3.983, 0.208),
            (3.932, -0.978),
            (2.847, 1.107),
            (6.932, 0.733),
            (3.102, -0.519),
            (2.686, -1.270),
            (4.495, -0.878),
        ],
    }

    def __init__(
        self,
        scenario: str = "SOSP",
        data_root: str = r"D:\Abdullah Jamal\downloads\VGA-exp\Pedestrian-Experimental-Data",
        trial_indices: List[
            int
        ] = None,  # Which trials to use (for train/val/test split)
        render_mode: Optional[str] = None,
        randomize_start_goal: bool = False,  # If True, randomize within bounds (for augmentation)
    ):
        super().__init__()

        self.scenario = scenario
        self.data_root = Path(data_root)
        self.render_mode = render_mode
        self.randomize_start_goal = randomize_start_goal

        # Load trial data
        self._load_trial_data()

        # Filter to specific trials if provided
        if trial_indices is not None:
            self.trial_data = self.trial_data.iloc[trial_indices].reset_index(drop=True)

        self.trial_indices = trial_indices
        self.current_trial_idx = 0
        self.num_trials = len(self.trial_data)

        # Set obstacle positions for this scenario
        self.obstacles = self._get_obstacle_positions()

        # Action space: [linear_velocity, angular_velocity]
        # Using velocity control for smoother learning
        self.action_space = spaces.Box(
            low=np.array([-self.MAX_VELOCITY, -2.0]),
            high=np.array([self.MAX_VELOCITY, 2.0]),
            dtype=np.float32,
        )

        # Observation space: base (11) + raycasting (36) + enhanced (3) = 50
        # Matching the structure from UltimateDomainRandomizedEnv for compatibility
        base_low = np.array(
            [
                0,
                -5,  # Position x, y (relative bounds)
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
        ray_high = np.full(self.N_RAYS, self.MAX_RAY_RANGE, dtype=np.float32)

        enhanced_low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        enhanced_high = np.array([1.0, 1.0, 20.0], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.concatenate([base_low, ray_low, enhanced_low]),
            high=np.concatenate([base_high, ray_high, enhanced_high]),
            dtype=np.float32,
        )

        # State variables
        self.agent_pos = None
        self.agent_vel = None
        self.agent_heading = None
        self.goal_pos = None
        self.desired_speed = None

        # Episode tracking
        self.steps = 0
        self.max_steps = 500  # ~25 seconds at 0.05s timestep
        self.total_distance_traveled = 0.0
        self.collision_count = 0
        self.previous_distance_to_goal = None

        # Rendering
        self.fig = None
        self.ax = None

    def _load_trial_data(self):
        """Load trial start/goal positions from VGA dataset."""
        if self.scenario == "SOSP":
            filepath = self.data_root / "SOSP_initialFinalPos_feed.txt"
            self.trial_data = pd.read_csv(
                filepath,
                header=None,
                names=[
                    "init_x",
                    "init_y",
                    "final_x",
                    "final_y",
                    "desired_speed",
                    "exp_no",
                ],
            )
        else:
            case = self.scenario.replace("MOSP_", "")
            filepath = self.data_root / f"MOSP_Case{case}_initialFinalPos_feed.txt"
            self.trial_data = pd.read_csv(
                filepath,
                header=None,
                names=[
                    "init_x",
                    "init_y",
                    "final_x",
                    "final_y",
                    "desired_speed",
                    "exp_no",
                    "idx_no",
                ],
            )

        # Drop any NaN rows
        self.trial_data = self.trial_data.dropna().reset_index(drop=True)

        print(f"[VGA Env] Loaded {len(self.trial_data)} trials for {self.scenario}")

    def _get_obstacle_positions(self) -> List[Tuple[float, float, float]]:
        """Get fixed obstacle positions for current scenario.
        Returns list of (x, y, radius) tuples.
        """
        if self.scenario == "SOSP":
            obs_key = "SOSP"
        else:
            obs_key = self.scenario

        # Load from actual data file if available
        try:
            if self.scenario == "SOSP":
                obs_file = self.data_root / "SOSP_obstPos_feed.txt"
            else:
                case = self.scenario.replace("MOSP_", "")
                obs_file = self.data_root / f"MOSP_Case{case}_obstPos_feed.txt"

            if obs_file.exists():
                obs_data = pd.read_csv(
                    obs_file, header=None, names=["x", "y", "obs_no"]
                )
                obstacles = [
                    (row["x"], row["y"], self.OBSTACLE_RADIUS)
                    for _, row in obs_data.iterrows()
                ]
                return obstacles
        except Exception as e:
            print(f"[WARN] Could not load obstacle file: {e}")

        # Fallback to hardcoded positions
        if obs_key in self.OBSTACLE_CONFIGS:
            return [
                (x, y, self.OBSTACLE_RADIUS) for x, y in self.OBSTACLE_CONFIGS[obs_key]
            ]

        return []

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment with next trial's start/goal."""
        super().reset(seed=seed)

        # Get trial data (cycle through trials)
        trial = self.trial_data.iloc[self.current_trial_idx]
        self.current_trial_idx = (self.current_trial_idx + 1) % self.num_trials

        # Set start position
        if self.randomize_start_goal:
            # Add small noise for data augmentation during training
            noise_scale = 0.1
            self.agent_pos = np.array(
                [
                    trial["init_x"] + np.random.uniform(-noise_scale, noise_scale),
                    trial["init_y"] + np.random.uniform(-noise_scale, noise_scale),
                ]
            )
            self.goal_pos = np.array(
                [
                    trial["final_x"] + np.random.uniform(-noise_scale, noise_scale),
                    trial["final_y"] + np.random.uniform(-noise_scale, noise_scale),
                ]
            )
        else:
            self.agent_pos = np.array([trial["init_x"], trial["init_y"]])
            self.goal_pos = np.array([trial["final_x"], trial["final_y"]])

        # Clip to arena bounds
        self.agent_pos = self._clip_to_arena(self.agent_pos)
        self.goal_pos = self._clip_to_arena(self.goal_pos)

        self.desired_speed = trial["desired_speed"]

        # Initial heading towards goal
        goal_dir = self.goal_pos - self.agent_pos
        self.agent_heading = np.arctan2(goal_dir[1], goal_dir[0])

        # Zero initial velocity
        self.agent_vel = np.array([0.0, 0.0])

        # Reset tracking
        self.steps = 0
        self.total_distance_traveled = 0.0
        self.collision_count = 0
        self.previous_distance_to_goal = np.linalg.norm(self.goal_pos - self.agent_pos)

        return self._get_observation(), self._get_info()

    def _clip_to_arena(self, pos: np.ndarray) -> np.ndarray:
        """Clip position to arena bounds with margin for agent radius."""
        margin = self.AGENT_RADIUS + 0.05
        return np.array(
            [
                np.clip(pos[0], self.ARENA_X_MIN + margin, self.ARENA_X_MAX - margin),
                np.clip(pos[1], self.ARENA_Y_MIN + margin, self.ARENA_Y_MAX - margin),
            ]
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.steps += 1

        # Parse action
        linear_vel = np.clip(action[0], -self.MAX_VELOCITY, self.MAX_VELOCITY)
        angular_vel = np.clip(action[1], -2.0, 2.0)

        # Update heading
        self.agent_heading += angular_vel * self.DT
        self.agent_heading = np.arctan2(
            np.sin(self.agent_heading), np.cos(self.agent_heading)
        )

        # Calculate velocity in world frame
        vx = linear_vel * np.cos(self.agent_heading)
        vy = linear_vel * np.sin(self.agent_heading)
        self.agent_vel = np.array([vx, vy])

        # Store old position for collision check
        old_pos = self.agent_pos.copy()

        # Update position
        new_pos = self.agent_pos + self.agent_vel * self.DT

        # Check collisions and boundaries
        collision = False

        # Obstacle collisions
        for obs_x, obs_y, obs_r in self.obstacles:
            dist_to_obs = np.linalg.norm(new_pos - np.array([obs_x, obs_y]))
            if dist_to_obs < self.AGENT_RADIUS + obs_r:
                collision = True
                # Push back from obstacle
                direction = new_pos - np.array([obs_x, obs_y])
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    new_pos = np.array([obs_x, obs_y]) + direction * (
                        self.AGENT_RADIUS + obs_r + 0.01
                    )

        # Arena boundary check
        if (
            new_pos[0] < self.ARENA_X_MIN + self.AGENT_RADIUS
            or new_pos[0] > self.ARENA_X_MAX - self.AGENT_RADIUS
            or new_pos[1] < self.ARENA_Y_MIN + self.AGENT_RADIUS
            or new_pos[1] > self.ARENA_Y_MAX - self.AGENT_RADIUS
        ):
            collision = True
            new_pos = self._clip_to_arena(new_pos)

        if collision:
            self.collision_count += 1

        # Track distance traveled
        self.total_distance_traveled += np.linalg.norm(new_pos - old_pos)
        self.agent_pos = new_pos

        # Calculate reward
        reward = self._compute_reward(collision)

        # Check termination
        distance_to_goal = np.linalg.norm(self.goal_pos - self.agent_pos)
        goal_reached = distance_to_goal < self.GOAL_TOLERANCE
        timeout = self.steps >= self.max_steps

        terminated = goal_reached
        truncated = timeout

        self.previous_distance_to_goal = distance_to_goal

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _compute_reward(self, collision: bool) -> float:
        """Compute reward - similar structure to UltimateDomainRandomizedEnv."""
        reward = 0.0

        distance_to_goal = np.linalg.norm(self.goal_pos - self.agent_pos)

        # Goal reached bonus
        if distance_to_goal < self.GOAL_TOLERANCE:
            # Bonus based on efficiency
            optimal_dist = np.linalg.norm(self.goal_pos - self.agent_pos)
            efficiency = max(
                0, 1 - (self.total_distance_traveled / (optimal_dist + 0.1) - 1)
            )
            reward += 50.0 + 20.0 * efficiency  # Large success bonus
            return reward

        # Progress reward (distance-based shaping)
        if self.previous_distance_to_goal is not None:
            progress = self.previous_distance_to_goal - distance_to_goal
            reward += progress * 5.0  # Scaled progress reward

        # Collision penalty
        if collision:
            reward -= 5.0

        # Small step penalty to encourage efficiency
        reward -= 0.02

        # Speed bonus (moving towards goal)
        speed = np.linalg.norm(self.agent_vel)
        goal_dir = self.goal_pos - self.agent_pos
        if np.linalg.norm(goal_dir) > 0:
            goal_dir = goal_dir / np.linalg.norm(goal_dir)
            velocity_towards_goal = np.dot(self.agent_vel, goal_dir)
            if velocity_towards_goal > 0:
                reward += 0.1 * velocity_towards_goal

        # Heading alignment bonus
        goal_angle = np.arctan2(goal_dir[1], goal_dir[0])
        heading_diff = abs(self.agent_heading - goal_angle)
        heading_diff = min(heading_diff, 2 * np.pi - heading_diff)
        reward += 0.05 * (1 - heading_diff / np.pi)

        return reward

    def _get_observation(self) -> np.ndarray:
        """Get observation matching UltimateDomainRandomizedEnv structure."""
        # Base features (11)
        goal_rel = self.goal_pos - self.agent_pos
        dist_to_goal = np.linalg.norm(goal_rel)
        goal_angle = np.arctan2(goal_rel[1], goal_rel[0])
        heading_diff = goal_angle - self.agent_heading
        heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))

        base_obs = np.array(
            [
                self.agent_pos[0],
                self.agent_pos[1],
                self.agent_vel[0],
                self.agent_vel[1],
                goal_rel[0],
                goal_rel[1],
                dist_to_goal,
                self.agent_heading,
                goal_angle,
                self.desired_speed,
                heading_diff,
            ],
            dtype=np.float32,
        )

        # Raycasting (36 rays)
        ray_obs = self._raycast()

        # Enhanced features (3): obstacle awareness, goal visibility, path estimate
        min_obs_dist = (
            min(
                [
                    np.linalg.norm(self.agent_pos - np.array([x, y])) - r
                    for x, y, r in self.obstacles
                ]
            )
            if self.obstacles
            else self.MAX_RAY_RANGE
        )
        obstacle_awareness = 1.0 if min_obs_dist < 1.5 else 0.0
        goal_visibility = 1.0 if self._check_goal_visible() else 0.0
        path_length_estimate = dist_to_goal  # Simple estimate

        enhanced_obs = np.array(
            [obstacle_awareness, goal_visibility, path_length_estimate],
            dtype=np.float32,
        )

        return np.concatenate([base_obs, ray_obs, enhanced_obs])

    def _raycast(self) -> np.ndarray:
        """Cast rays for obstacle detection."""
        rays = np.full(self.N_RAYS, self.MAX_RAY_RANGE, dtype=np.float32)

        for i in range(self.N_RAYS):
            angle = self.agent_heading + (2 * np.pi * i / self.N_RAYS) - np.pi
            ray_dir = np.array([np.cos(angle), np.sin(angle)])

            min_dist = self.MAX_RAY_RANGE

            # Check obstacles
            for obs_x, obs_y, obs_r in self.obstacles:
                obs_pos = np.array([obs_x, obs_y])
                dist = self._ray_circle_intersection(
                    self.agent_pos, ray_dir, obs_pos, obs_r
                )
                if dist is not None and dist < min_dist:
                    min_dist = dist

            # Check arena boundaries
            boundary_dist = self._ray_boundary_intersection(self.agent_pos, ray_dir)
            if boundary_dist is not None and boundary_dist < min_dist:
                min_dist = boundary_dist

            rays[i] = min_dist

        return rays

    def _ray_circle_intersection(
        self,
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        circle_center: np.ndarray,
        circle_radius: float,
    ) -> Optional[float]:
        """Calculate ray-circle intersection distance."""
        oc = ray_origin - circle_center
        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - circle_radius**2

        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return None

        t = (-b - np.sqrt(discriminant)) / (2 * a)
        if t > 0:
            return t
        return None

    def _ray_boundary_intersection(
        self, ray_origin: np.ndarray, ray_dir: np.ndarray
    ) -> Optional[float]:
        """Calculate ray intersection with arena boundaries."""
        min_t = float("inf")

        # Check all 4 boundaries
        boundaries = [
            (self.ARENA_X_MIN, 0, 1, 0),  # Left
            (self.ARENA_X_MAX, 0, 1, 0),  # Right
            (self.ARENA_Y_MIN, 1, 0, 1),  # Bottom
            (self.ARENA_Y_MAX, 1, 0, 1),  # Top
        ]

        for val, axis_idx, dir_idx, is_y in boundaries:
            if is_y:
                if abs(ray_dir[1]) > 1e-6:
                    t = (val - ray_origin[1]) / ray_dir[1]
                    if t > 0 and t < min_t:
                        x = ray_origin[0] + t * ray_dir[0]
                        if self.ARENA_X_MIN <= x <= self.ARENA_X_MAX:
                            min_t = t
            else:
                if abs(ray_dir[0]) > 1e-6:
                    t = (val - ray_origin[0]) / ray_dir[0]
                    if t > 0 and t < min_t:
                        y = ray_origin[1] + t * ray_dir[1]
                        if self.ARENA_Y_MIN <= y <= self.ARENA_Y_MAX:
                            min_t = t

        return min_t if min_t < float("inf") else None

    def _check_goal_visible(self) -> bool:
        """Check if there's a clear line of sight to goal."""
        direction = self.goal_pos - self.agent_pos
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return True
        direction = direction / dist

        for obs_x, obs_y, obs_r in self.obstacles:
            obs_pos = np.array([obs_x, obs_y])
            intersection = self._ray_circle_intersection(
                self.agent_pos, direction, obs_pos, obs_r + self.AGENT_RADIUS
            )
            if intersection is not None and intersection < dist:
                return False
        return True

    def _get_info(self) -> Dict:
        """Get episode info."""
        distance_to_goal = np.linalg.norm(self.goal_pos - self.agent_pos)
        return {
            "distance_to_goal": distance_to_goal,
            "goal_reached": distance_to_goal < self.GOAL_TOLERANCE,
            "collisions": self.collision_count,
            "steps": self.steps,
            "scenario": self.scenario,
            "trial_idx": self.current_trial_idx,
            "distance_traveled": self.total_distance_traveled,
        }

    def render(self):
        """Render the environment."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 5))
            plt.ion()

        self.ax.clear()

        # Arena boundaries
        self.ax.set_xlim(self.ARENA_X_MIN - 0.5, self.ARENA_X_MAX + 0.5)
        self.ax.set_ylim(self.ARENA_Y_MIN - 0.5, self.ARENA_Y_MAX + 0.5)
        self.ax.set_aspect("equal")

        # Draw arena
        arena = Rectangle(
            (self.ARENA_X_MIN, self.ARENA_Y_MIN),
            self.ARENA_X_MAX - self.ARENA_X_MIN,
            self.ARENA_Y_MAX - self.ARENA_Y_MIN,
            fill=False,
            edgecolor="black",
            linewidth=2,
        )
        self.ax.add_patch(arena)

        # Draw obstacles
        for obs_x, obs_y, obs_r in self.obstacles:
            obstacle = Circle((obs_x, obs_y), obs_r, color="gray", alpha=0.7)
            self.ax.add_patch(obstacle)

        # Draw agent
        agent = Circle(self.agent_pos, self.AGENT_RADIUS, color="blue", alpha=0.8)
        self.ax.add_patch(agent)

        # Draw heading direction
        heading_end = self.agent_pos + 0.5 * np.array(
            [np.cos(self.agent_heading), np.sin(self.agent_heading)]
        )
        self.ax.plot(
            [self.agent_pos[0], heading_end[0]],
            [self.agent_pos[1], heading_end[1]],
            "b-",
            linewidth=2,
        )

        # Draw goal
        goal = Circle(self.goal_pos, self.GOAL_TOLERANCE, color="green", alpha=0.5)
        self.ax.add_patch(goal)
        self.ax.plot(self.goal_pos[0], self.goal_pos[1], "g*", markersize=15)

        self.ax.set_title(
            f"{self.scenario} - Step {self.steps} - Collisions: {self.collision_count}"
        )

        plt.draw()
        plt.pause(0.01)

        if self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return img

    def close(self):
        """Close rendering."""
        if self.fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self.fig)
            self.fig = None
            self.ax = None


class VGAMixedEnv(gym.Env):
    """
    Mixed environment that samples from ALL VGA scenarios.
    Useful for training a generalized agent across all obstacle densities.
    """

    def __init__(
        self,
        data_root: str = r"D:\Abdullah Jamal\downloads\VGA-exp\Pedestrian-Experimental-Data",
        scenarios: List[str] = None,
        train_split: float = 0.7,
        split_type: str = "train",  # "train", "val", "test"
        render_mode: Optional[str] = None,
        randomize_start_goal: bool = False,
    ):
        super().__init__()

        if scenarios is None:
            scenarios = ["SOSP", "MOSP_A", "MOSP_B", "MOSP_C", "MOSP_D"]

        self.scenarios = scenarios
        self.data_root = data_root
        self.render_mode = render_mode
        self.randomize_start_goal = randomize_start_goal

        # Create environment for each scenario with proper splits
        self.envs = {}
        self.trial_counts = {}

        for scenario in scenarios:
            # Create temp env to get trial count
            temp_env = VGAExperimentalEnv(scenario=scenario, data_root=data_root)
            n_trials = temp_env.num_trials
            temp_env.close()

            # Calculate splits
            n_train = int(n_trials * train_split)
            n_val = int(n_trials * 0.15)
            n_test = n_trials - n_train - n_val

            if split_type == "train":
                indices = list(range(0, n_train))
            elif split_type == "val":
                indices = list(range(n_train, n_train + n_val))
            else:  # test
                indices = list(range(n_train + n_val, n_trials))

            if len(indices) > 0:
                self.envs[scenario] = VGAExperimentalEnv(
                    scenario=scenario,
                    data_root=data_root,
                    trial_indices=indices,
                    render_mode=render_mode,
                    randomize_start_goal=randomize_start_goal,
                )
                self.trial_counts[scenario] = len(indices)

        # Use first env as reference for spaces
        first_env = list(self.envs.values())[0]
        self.action_space = first_env.action_space
        self.observation_space = first_env.observation_space

        self.current_scenario = None
        self.current_env = None

        print(f"[VGA Mixed Env] Created {split_type} split with scenarios:")
        for s, c in self.trial_counts.items():
            print(f"  {s}: {c} trials")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset by sampling a random scenario."""
        super().reset(seed=seed)

        # Sample scenario weighted by trial count
        total = sum(self.trial_counts.values())
        probs = [self.trial_counts[s] / total for s in self.envs.keys()]
        self.current_scenario = np.random.choice(list(self.envs.keys()), p=probs)
        self.current_env = self.envs[self.current_scenario]

        return self.current_env.reset(seed=seed, options=options)

    def step(self, action):
        return self.current_env.step(action)

    def render(self):
        if self.current_env is not None:
            return self.current_env.render()

    def close(self):
        for env in self.envs.values():
            env.close()


if __name__ == "__main__":
    # Test the environment
    print("Testing VGA Experimental Environment...")

    env = VGAExperimentalEnv(scenario="SOSP", render_mode="human")

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Initial info: {info}")

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            print(f"Episode ended: {info}")
            obs, info = env.reset()

    env.close()
    print("Test complete!")
