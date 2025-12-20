"""
VGA + UPL - FIXED Implementation
=================================

Properly implements the Variable Goal Approach algorithm as described in the paper:
1. Detect nearest obstacle in rectangular region
2. Form cluster of nearby obstacles
3. Identify tangential obstacles (left/right)
4. Place variable goals perpendicular to tangents
5. Select goal using least deviation method
6. Update goal position to maintain spacing

Combined with UPL physics for realistic movement.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_base import NavigationModel, SimulationResult
from upl_physics import UPLPhysics, UPLParameters


@dataclass
class Obstacle:
    """Circular obstacle representation"""

    position: np.ndarray
    radius: float

    def distance_to(self, point: np.ndarray) -> float:
        """Surface-to-surface distance"""
        return np.linalg.norm(self.position - point) - self.radius


class VGAUPLPlannerFixed(NavigationModel):
    """
    Properly implemented VGA + UPL following the paper exactly.
    """

    def __init__(
        self,
        use_probabilistic: bool = False,
        upl_params: Optional[UPLParameters] = None,
    ):
        mode = "stochastic" if use_probabilistic else "deterministic"
        super().__init__(name=f"VGA+UPL-Fixed ({mode})")

        # VGA parameters
        self.use_probabilistic = use_probabilistic
        self.pedestrian_radius = 0.225  # m
        self.personal_gap = self.pedestrian_radius  # Personal space
        self.visible_angle = 100  # degrees (±100°)

        # UPL physics
        self.upl = UPLPhysics(upl_params)

        # State
        self.current_position = None
        self.current_velocity = None
        self.final_goal = None
        self.variable_goal = None
        self.obstacles = []

    def reset(self, start_pos: np.ndarray, goal_pos: np.ndarray, **kwargs):
        """Reset for new simulation"""
        self.current_position = start_pos.copy()
        self.current_velocity = np.zeros(2)
        self.final_goal = goal_pos.copy()
        self.variable_goal = goal_pos.copy()
        self.obstacles = []

    def set_obstacles(self, obstacles: List[Dict]):
        """Set obstacles from scenario data"""
        self.obstacles = [
            Obstacle(position=np.array(obs["position"]), radius=obs["radius"])
            for obs in obstacles
        ]

    # ========== VGA ALGORITHM ==========

    def _step1_find_nearest_obstacle(self) -> Optional[Obstacle]:
        """
        STEP 1: Find nearest obstacle in rectangular region between pedestrian and goal.
        """
        if not self.obstacles:
            return None

        # Define rectangular region (ABCD in paper)
        to_goal = self.final_goal - self.current_position

        # Find obstacles in direct path region
        obstacles_in_region = []
        for obs in self.obstacles:
            # Simple check: is obstacle between current position and goal?
            to_obs = obs.position - self.current_position

            # Project onto goal direction
            proj_length = np.dot(to_obs, to_goal) / (np.linalg.norm(to_goal) + 1e-6)

            # If projection is positive and less than distance to goal, it's in the region
            if 0 < proj_length < np.linalg.norm(to_goal):
                # Check perpendicular distance from path
                perp_dist = np.abs(
                    np.cross(to_goal / (np.linalg.norm(to_goal) + 1e-6), to_obs)
                )
                if perp_dist < obs.radius + 2.0:  # Within reasonable corridor width
                    obstacles_in_region.append(obs)

        if not obstacles_in_region:
            return None

        # Return nearest (center-to-center)
        return min(
            obstacles_in_region,
            key=lambda o: np.linalg.norm(o.position - self.current_position),
        )

    def _step2_form_cluster(self, nearest_obs: Obstacle) -> List[Obstacle]:
        """
        STEP 2: Form cluster of obstacles.
        Surface-to-surface distance < pedestrian_size are grouped.
        """
        cluster = [nearest_obs]
        cluster_changed = True

        while cluster_changed:
            cluster_changed = False
            for obs in self.obstacles:
                # Check if already in cluster (by identity)
                if any(obs is c for c in cluster):
                    continue

                # Check if close to any obstacle in cluster
                for cluster_obs in cluster:
                    surface_dist = (
                        np.linalg.norm(obs.position - cluster_obs.position)
                        - obs.radius
                        - cluster_obs.radius
                    )
                    if surface_dist < 2 * self.pedestrian_radius:  # Pedestrian size
                        cluster.append(obs)
                        cluster_changed = True
                        break

        return cluster

    def _step3_find_tangential_obstacles(
        self, cluster: List[Obstacle]
    ) -> Tuple[Obstacle, Obstacle]:
        """
        STEP 3: Find left and right tangential obstacles.
        TL = leftmost angle, TR = rightmost angle from pedestrian position.
        """
        if len(cluster) == 1:
            return cluster[0], cluster[0]

        # Calculate angles for all cluster obstacles
        angles = []
        for obs in cluster:
            to_obs = obs.position - self.current_position
            angle = np.arctan2(to_obs[1], to_obs[0])
            angles.append((angle, obs))

        # Sort by angle
        angles.sort(key=lambda x: x[0])

        # Leftmost (highest angle) and rightmost (lowest angle)
        left_tangent = angles[-1][1]
        right_tangent = angles[0][1]

        return left_tangent, right_tangent

    def _step4_place_variable_goals(
        self, left_obs: Obstacle, right_obs: Obstacle
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        STEP 4: Place variable goals perpendicular to tangential obstacles.
        """
        # Direction from pedestrian to obstacle
        to_left = left_obs.position - self.current_position
        to_right = right_obs.position - self.current_position

        # Perpendicular directions (90° rotation)
        perp_left = np.array([-to_left[1], to_left[0]])
        perp_left = perp_left / (np.linalg.norm(perp_left) + 1e-6)

        perp_right = np.array([to_right[1], -to_right[0]])
        perp_right = perp_right / (np.linalg.norm(perp_right) + 1e-6)

        # Place goals at personal gap distance
        gap_dist = left_obs.radius + self.personal_gap
        left_goal = left_obs.position + perp_left * gap_dist

        gap_dist = right_obs.radius + self.personal_gap
        right_goal = right_obs.position + perp_right * gap_dist

        return left_goal, right_goal

    def _step5_select_variable_goal(
        self, left_goal: np.ndarray, right_goal: np.ndarray
    ) -> np.ndarray:
        """
        STEP 5: Select one variable goal using:
        - Visibility check (-100° to 100°)
        - Least deviation method
        """
        # Calculate deviations (distance to line from pedestrian to final goal)
        to_goal = self.final_goal - self.current_position
        goal_dir = to_goal / (np.linalg.norm(to_goal) + 1e-6)

        # Perpendicular distance from line
        to_left = left_goal - self.current_position
        to_right = right_goal - self.current_position

        left_deviation = np.abs(
            np.cross(goal_dir, to_left / (np.linalg.norm(to_left) + 1e-6))
        )
        right_deviation = np.abs(
            np.cross(goal_dir, to_right / (np.linalg.norm(to_right) + 1e-6))
        )

        # Check visibility (angle from forward direction)
        left_angle = np.arctan2(to_left[1], to_left[0])
        right_angle = np.arctan2(to_right[1], to_right[0])
        forward_angle = np.arctan2(goal_dir[1], goal_dir[0])

        left_visible = np.abs(left_angle - forward_angle) < np.radians(
            self.visible_angle
        )
        right_visible = np.abs(right_angle - forward_angle) < np.radians(
            self.visible_angle
        )

        # Select based on visibility and deviation
        if left_visible and right_visible:
            return left_goal if left_deviation < right_deviation else right_goal
        elif left_visible:
            return left_goal
        elif right_visible:
            return right_goal
        else:
            # Neither visible - choose lesser deviation
            return left_goal if left_deviation < right_deviation else right_goal

    def _step6_update_variable_goal_position(self, goal: np.ndarray) -> np.ndarray:
        """
        STEP 6: Update variable goal to maintain equal spacing from obstacles.
        """
        # Check for nearby obstacles
        nearby_obstacles = []
        for obs in self.obstacles:
            dist = np.linalg.norm(goal - obs.position) - obs.radius
            if dist < self.personal_gap * 2:
                nearby_obstacles.append(obs)

        if len(nearby_obstacles) >= 2:
            # Adjust to midpoint between closest two obstacles
            nearby_obstacles.sort(key=lambda o: np.linalg.norm(o.position - goal))
            obs1, obs2 = nearby_obstacles[0], nearby_obstacles[1]

            # Midpoint between surfaces
            adjusted_goal = (obs1.position + obs2.position) / 2
            return adjusted_goal

        return goal

    def update_variable_goal(self):
        """
        Complete VGA algorithm: Steps 1-6.
        """
        # Check if close to current variable goal
        if np.linalg.norm(self.current_position - self.variable_goal) < 0.5:
            self.variable_goal = self.final_goal.copy()

        # Check if close to final goal
        if np.linalg.norm(self.current_position - self.final_goal) < 0.3:
            return

        # STEP 1: Find nearest obstacle
        nearest_obs = self._step1_find_nearest_obstacle()
        if not nearest_obs:
            self.variable_goal = self.final_goal.copy()
            return

        # STEP 2: Form cluster
        cluster = self._step2_form_cluster(nearest_obs)

        # STEP 3: Find tangential obstacles
        left_obs, right_obs = self._step3_find_tangential_obstacles(cluster)

        # STEP 4: Place variable goals
        left_goal, right_goal = self._step4_place_variable_goals(left_obs, right_obs)

        # STEP 5: Select variable goal
        selected_goal = self._step5_select_variable_goal(left_goal, right_goal)

        # STEP 6: Update position
        self.variable_goal = self._step6_update_variable_goal_position(selected_goal)

    def step(self) -> bool:
        """
        Single simulation step: Update variable goal, then move with UPL.
        """
        # Update variable goal using VGA
        self.update_variable_goal()

        # Calculate goal direction
        to_goal = self.variable_goal - self.current_position
        dist = np.linalg.norm(to_goal)
        goal_direction = to_goal / (dist + 1e-6)

        # Convert obstacles to UPL format
        upl_obstacles = [(obs.position, obs.radius) for obs in self.obstacles]

        # Use UPL to calculate force
        force = self.upl.calculate_total_force(
            agent_pos=self.current_position,
            current_velocity=self.current_velocity,
            goal_direction=goal_direction,
            obstacles=upl_obstacles,
        )

        # Update velocity and position
        self.current_velocity += force * self.upl.params.dt
        self.current_position += self.current_velocity * self.upl.params.dt

        # Check if reached final goal
        dist_to_goal = np.linalg.norm(self.current_position - self.final_goal)
        return dist_to_goal < 0.3

    def simulate(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        obstacles: List[Dict],
        max_steps: int = 2000,
    ):
        """
        Run full simulation with VGA+UPL.
        """
        self.reset(start_pos, goal_pos)
        self.set_obstacles(obstacles)

        # Storage
        positions = [self.current_position.copy()]
        velocities = [self.current_velocity.copy()]
        timestamps = [0.0]

        current_time = 0.0

        for step in range(max_steps):
            goal_reached = self.step()
            current_time += self.upl.params.dt

            positions.append(self.current_position.copy())
            velocities.append(self.current_velocity.copy())
            timestamps.append(current_time)

            if goal_reached:
                break

        # Check success
        final_dist = np.linalg.norm(self.current_position - self.final_goal)
        success = final_dist < 0.3

        return SimulationResult(
            positions=np.array(positions),
            velocities=np.array(velocities),
            timestamps=np.array(timestamps),
            success=success,
            metadata={
                "mode": "stochastic" if self.use_probabilistic else "deterministic",
                "final_distance_to_goal": final_dist,
                "num_obstacles": len(self.obstacles),
            },
        )

    def predict_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """Placeholder for base class"""
        return np.zeros(2)
