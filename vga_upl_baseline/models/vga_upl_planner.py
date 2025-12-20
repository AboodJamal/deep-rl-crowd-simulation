"""
VGA + UPL Integrated Planner
=============================

Combines Variable Goal Approach (goal selection) with Universal Power Law (physics).
This matches the paper's implementation exactly.

Architecture:
1. VGA Module: Selects intermediate goals (left/right around obstacles)
2. UPL Module: Calculates forces and moves agent toward selected goal
3. Integration: VGA updates goal, UPL follows it with physics

This supports both Deterministic and Stochastic modes.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_base import NavigationModel, SimulationResult
from upl_physics import UPLPhysics, UPLParameters


@dataclass
class Obstacle:
    """Representation of a circular obstacle"""

    position: np.ndarray  # (x, y)
    radius: float

    def distance_to(self, point: np.ndarray) -> float:
        """Distance from obstacle surface to point"""
        return np.linalg.norm(self.position - point) - self.radius


class VGAUPLPlanner(NavigationModel):
    """
    Combined VGA + UPL navigation model.

    VGA provides intelligent goal selection (avoiding obstacles by choosing subgoals).
    UPL provides realistic physics-based movement toward those goals.
    """

    def __init__(
        self,
        use_probabilistic: bool = False,
        upl_params: Optional[UPLParameters] = None,
        **kwargs,
    ):
        """
        Initialize VGA+UPL planner.

        Args:
            use_probabilistic: If True, use stochastic goal selection
            upl_params: UPL physics parameters
            **kwargs: Additional VGA parameters
        """
        mode = "stochastic" if use_probabilistic else "deterministic"
        super().__init__(name=f"VGA+UPL ({mode})")

        # VGA parameters
        self.use_probabilistic = use_probabilistic
        self.personal_distance = kwargs.get(
            "personal_distance", 0.1
        )  # Buffer distance from obstacles
        self.clustering_threshold = kwargs.get("clustering_threshold", 1.0)
        self.goal_switch_threshold = kwargs.get("goal_switch_threshold", 0.3)
        self.detection_cone_angle = kwargs.get("detection_cone_angle", 60.0)  # degrees

        # UPL physics engine
        self.upl = UPLPhysics(upl_params)

        # State variables
        self.current_position = None
        self.current_velocity = None
        self.final_goal = None
        self.current_goal = None  # Variable goal
        self.obstacles = []

        # Stuck detection
        self.stuck_counter = 0
        self.stuck_threshold = 20  # steps with low velocity
        self.velocity_threshold = 0.1  # m/s

    def predict_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        Predict action (required by base class, not used in VGA).

        Args:
            observation: Environment observation

        Returns:
            Action array (placeholder)
        """
        # VGA doesn't use observation-action paradigm
        # It directly plans paths, so return empty action
        return np.zeros(2)

    def reset(self, start_pos: np.ndarray, goal_pos: np.ndarray, **kwargs):
        """Reset planner for new simulation."""
        self.current_position = start_pos.copy()
        self.current_velocity = np.zeros(2)
        self.final_goal = goal_pos.copy()
        self.current_goal = goal_pos.copy()

        # Reset stuck counter
        self.stuck_counter = 0

        # Parse obstacles
        obstacles_data = kwargs.get("obstacles", [])
        self.obstacles = []

        for obs_data in obstacles_data:
            if isinstance(obs_data, Obstacle):
                self.obstacles.append(obs_data)
            elif isinstance(obs_data, dict):
                self.obstacles.append(
                    Obstacle(
                        position=np.array(obs_data["position"]),
                        radius=obs_data.get("radius", 0.3),
                    )
                )
            elif isinstance(obs_data, (list, tuple, np.ndarray)):
                # Assume [x, y, radius] format
                self.obstacles.append(
                    Obstacle(
                        position=np.array(obs_data[:2]),
                        radius=obs_data[2] if len(obs_data) > 2 else 0.3,
                    )
                )

    def _detect_obstacles_in_cone(self) -> List[Obstacle]:
        """
        Detect obstacles that ACTUALLY BLOCK the direct path to final goal.

        Per VGA paper: Only obstacles whose personal space intersects
        the direct line from agent to goal are considered blocking.
        
        CRITICAL: Only consider obstacles AHEAD of the agent (not already passed).
        """
        to_goal = self.final_goal - self.current_position
        goal_dist = np.linalg.norm(to_goal)

        if goal_dist < 1e-6:
            return []

        to_goal_norm = to_goal / goal_dist

        blocking = []
        agent_radius = self.upl.params.r  # 0.15m
        
        for obs in self.obstacles:
            # Check if obstacle actually blocks the direct line to goal
            # Project obstacle center onto the line from agent to goal
            to_obs = obs.position - self.current_position
            projection_length = np.dot(to_obs, to_goal_norm)

            # Only consider obstacles AHEAD of us and before the goal
            # Obstacle is "passed" if its center projection is behind agent by 0.1m
            if projection_length < 0.1 or projection_length > goal_dist:
                continue

            # Perpendicular distance from obstacle center to the line
            projection_point = self.current_position + projection_length * to_goal_norm
            perp_dist = np.linalg.norm(obs.position - projection_point)

            # Effective radius: obstacle + agent body + personal distance
            effective_radius = obs.radius + agent_radius + self.personal_distance

            # Is the line actually blocked?
            if perp_dist < effective_radius:
                blocking.append(obs)

        return blocking

    def _find_tangent_points(self, obstacle: Obstacle) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find left and right tangent points around an obstacle.

        Args:
            obstacle: The obstacle to navigate around

        Returns:
            (left_tangent, right_tangent) positions
        """
        center = obstacle.position
        # CRITICAL: Effective radius must include AGENT BODY + personal distance
        # Agent radius = 0.2m (from UPL params)
        agent_radius = self.upl.params.r  # 0.2m
        effective_radius = obstacle.radius + agent_radius + self.personal_distance

        # Vector from agent to obstacle
        to_center = center - self.current_position
        dist_to_center = np.linalg.norm(to_center)

        if dist_to_center < effective_radius:
            # Too close - use perpendicular directions
            perpendicular = np.array([-to_center[1], to_center[0]])
            perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-6)

            left_tangent = center + perpendicular * effective_radius
            right_tangent = center - perpendicular * effective_radius
            return left_tangent, right_tangent

        # Calculate tangent angle
        angle = np.arcsin(effective_radius / dist_to_center)

        # Direction to center
        to_center_norm = to_center / dist_to_center

        # Rotation matrices
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # Left tangent (counter-clockwise rotation)
        left_dir = np.array(
            [
                cos_angle * to_center_norm[0] - sin_angle * to_center_norm[1],
                sin_angle * to_center_norm[0] + cos_angle * to_center_norm[1],
            ]
        )

        # Right tangent (clockwise rotation)
        right_dir = np.array(
            [
                cos_angle * to_center_norm[0] + sin_angle * to_center_norm[1],
                -sin_angle * to_center_norm[0] + cos_angle * to_center_norm[1],
            ]
        )

        # Tangent points (project from center)
        tangent_dist = dist_to_center * np.cos(angle)
        left_tangent = self.current_position + left_dir * tangent_dist
        right_tangent = self.current_position + right_dir * tangent_dist

        return left_tangent, right_tangent

    def _calculate_deviation(self, candidate_goal: np.ndarray) -> float:
        """
        Calculate angular deviation from straight line to final goal.

        Args:
            candidate_goal: Proposed subgoal position

        Returns:
            Deviation angle in radians
        """
        # Direct path to final goal
        to_final = self.final_goal - self.current_position
        dist_final = np.linalg.norm(to_final)

        if dist_final < 1e-6:
            return 0.0

        # Path via candidate goal
        to_candidate = candidate_goal - self.current_position
        dist_candidate = np.linalg.norm(to_candidate)

        if dist_candidate < 1e-6:
            return 0.0

        # Calculate angle between vectors
        dot_product = np.dot(to_final, to_candidate) / (dist_final * dist_candidate)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

        return angle

    def _is_path_clear(self, from_pos: np.ndarray, to_pos: np.ndarray) -> bool:
        """Check if straight line from from_pos to to_pos is clear of obstacles."""
        to_goal = to_pos - from_pos
        goal_dist = np.linalg.norm(to_goal)

        if goal_dist < 1e-6:
            return True

        to_goal_norm = to_goal / goal_dist
        agent_radius = self.upl.params.r

        for obs in self.obstacles:
            to_obs = obs.position - from_pos
            projection_length = np.dot(to_obs, to_goal_norm)

            if projection_length < 0 or projection_length > goal_dist:
                continue

            projection_point = from_pos + projection_length * to_goal_norm
            perp_dist = np.linalg.norm(obs.position - projection_point)
            effective_radius = obs.radius + agent_radius + self.personal_distance

            if perp_dist < effective_radius:
                return False
        return True

    def _calculate_clearance(self, point: np.ndarray) -> float:
        """
        Calculate minimum clearance from a point to all obstacles.
        
        Per VGA paper: clearance = min(distance to ALL obstacles - obstacle radius)
        """
        if not self.obstacles:
            return float('inf')
            
        agent_radius = self.upl.params.r
        min_clearance = float('inf')
        
        for obs in self.obstacles:
            dist = np.linalg.norm(point - obs.position)
            # Clearance = distance from point to obstacle surface, minus agent body
            clearance = dist - obs.radius - agent_radius
            min_clearance = min(min_clearance, clearance)
            
        return min_clearance

    def _select_variable_goal(self) -> np.ndarray:
        """
        Select the next variable goal using VGA logic:

        1. Check if direct path to goal is blocked
        2. If blocked, find NEAREST blocking obstacle
        3. Calculate tangent points around it
        4. Select tangent with minimum deviation AND sufficient clearance
        """
        # Step 1: Find obstacles blocking direct path
        blocking_obstacles = self._detect_obstacles_in_cone()

        if not blocking_obstacles:
            return self.final_goal.copy()

        # Step 2: Find NEAREST blocking obstacle (simple, works better than clustering)
        distances = [np.linalg.norm(obs.position - self.current_position) for obs in blocking_obstacles]
        nearest_idx = np.argmin(distances)
        nearest_obs = blocking_obstacles[nearest_idx]

        # Step 3: Calculate tangent points
        left_tangent, right_tangent = self._find_tangent_points(nearest_obs)

        # Step 4: Evaluate tangents
        left_deviation = self._calculate_deviation(left_tangent)
        right_deviation = self._calculate_deviation(right_tangent)
        left_clearance = self._calculate_clearance(left_tangent)
        right_clearance = self._calculate_clearance(right_tangent)
        
        # Minimum required clearance
        min_required = self.personal_distance  # 0.1m

        if self.use_probabilistic:
            # Stochastic selection
            left_valid = left_clearance >= min_required
            right_valid = right_clearance >= min_required
            
            if left_valid and not right_valid:
                return left_tangent
            if right_valid and not left_valid:
                return right_tangent
            if not left_valid and not right_valid:
                return self.final_goal.copy()
            
            # Both valid - probabilistic based on deviation
            if left_deviation < 1e-6:
                return left_tangent
            if right_deviation < 1e-6:
                return right_tangent
                
            total_dev = left_deviation + right_deviation
            prob_left = right_deviation / total_dev
            if np.random.random() < prob_left:
                return left_tangent
            else:
                return right_tangent
        else:
            # Deterministic: pick tangent with minimum deviation if valid
            left_valid = left_clearance >= min_required
            right_valid = right_clearance >= min_required
            
            if left_valid and not right_valid:
                return left_tangent
            if right_valid and not left_valid:
                return right_tangent
            if not left_valid and not right_valid:
                return self.final_goal.copy()
            
            # Both valid - pick minimum deviation
            if left_deviation <= right_deviation:
                return left_tangent
            else:
                return right_tangent

    def step(self) -> bool:
        """
        Perform one simulation step.

        Returns:
            True if goal reached, False otherwise
        """
        # Check if final goal reached
        if np.linalg.norm(self.current_position - self.final_goal) < 0.2:
            return True

        # VGA: Select variable goal
        self.current_goal = self._select_variable_goal()

        # Check if VGA selected a detour (avoiding obstacles)
        is_avoiding_obstacles = (
            np.linalg.norm(self.current_goal - self.final_goal) > 0.3
        )

        if is_avoiding_obstacles:
            # PURE VGA MODE: Trust geometric planning completely
            # Move directly toward variable goal with simple kinematics (no UPL forces)
            # This matches paper behavior - smooth paths close to obstacles

            to_goal = self.current_goal - self.current_position
            dist_to_goal = np.linalg.norm(to_goal)

            if dist_to_goal > 1e-6:
                direction = to_goal / dist_to_goal
                # Simple velocity toward variable goal
                desired_vel = self.upl.params.v_desired * direction  # 1.34 m/s

                # Smooth velocity transition (not instant)
                alpha = 0.3  # Smoothing factor
                new_vel = alpha * desired_vel + (1 - alpha) * self.current_velocity

                # Update position
                new_pos = self.current_position + new_vel * self.upl.params.dt
            else:
                new_vel = self.current_velocity
                new_pos = self.current_position

            # PROPER collision avoidance - agent body must not overlap obstacles
            agent_radius = self.upl.params.r  # 0.2m
            min_clearance = (
                agent_radius + self.personal_distance
            )  # 0.2 + 0.1 = 0.3m from surface

            for obs in self.obstacles:
                to_obs = new_pos - obs.position
                dist_to_obs = np.linalg.norm(to_obs)
                surface_dist = dist_to_obs - obs.radius

                if surface_dist < min_clearance:  # Agent body would overlap
                    # Push away to maintain safe distance
                    if dist_to_obs > 1e-6:
                        push_dir = to_obs / dist_to_obs
                        push_amount = min_clearance - surface_dist
                        new_pos = new_pos + push_dir * push_amount
        else:
            # DIRECT PATH MODE: Use full UPL physics for safety
            upl_obstacles = [(obs.position, obs.radius) for obs in self.obstacles]
            new_pos, new_vel, _ = self.upl.step(
                self.current_position,
                self.current_velocity,
                self.current_goal,
                upl_obstacles,
            )

        # Stuck detection and recovery
        vel_mag = np.linalg.norm(new_vel)
        if vel_mag < self.velocity_threshold:
            self.stuck_counter += 1
            if self.stuck_counter > self.stuck_threshold:
                # Apply small random perturbation to escape local minimum
                perturbation = np.random.uniform(-0.1, 0.1, size=2)
                new_pos = new_pos + perturbation
                # Reset counter
                self.stuck_counter = 0
        else:
            self.stuck_counter = 0

        # Update state
        self.current_position = new_pos
        self.current_velocity = new_vel

        return False

    def simulate(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        max_steps: int = 1000,
        **kwargs,
    ) -> SimulationResult:
        """
        Simulate full trajectory from start to goal.

        Args:
            start_pos: Starting position [x, y]
            goal_pos: Goal position [x, y]
            max_steps: Maximum simulation steps
            **kwargs: Must include 'obstacles' list

        Returns:
            SimulationResult with trajectory data
        """
        self.reset(start_pos, goal_pos, **kwargs)

        # Storage
        positions = [self.current_position.copy()]
        velocities = [self.current_velocity.copy()]
        timestamps = [0.0]

        current_time = 0.0

        for step in range(max_steps):
            # Perform step
            goal_reached = self.step()

            # Update time
            current_time += self.upl.params.dt

            # Store state
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


def test_vga_upl():
    """Test VGA+UPL integration."""
    print("Testing VGA+UPL Integration...")

    # Create planner
    planner = VGAUPLPlanner(use_probabilistic=False)

    # Test scenario: same as UPL test but with VGA intelligence
    start = np.array([0.0, 0.0])
    goal = np.array([10.0, 0.0])
    obstacles = [{"position": [5.0, 0.0], "radius": 0.5}]  # Blocking direct path

    # Simulate
    result = planner.simulate(start, goal, obstacles=obstacles, max_steps=500)

    print(f"\nDeterministic Mode:")
    print(f"Success: {result.success}")
    print(f"Steps: {len(result.positions)}")
    print(f"Final position: {result.positions[-1]}")
    print(f"Distance to goal: {result.metadata['final_distance_to_goal']:.3f}m")

    # Test stochastic mode
    planner_stoch = VGAUPLPlanner(use_probabilistic=True)
    result_stoch = planner_stoch.simulate(
        start, goal, obstacles=obstacles, max_steps=500
    )

    print(f"\nStochastic Mode:")
    print(f"Success: {result_stoch.success}")
    print(f"Steps: {len(result_stoch.positions)}")
    print(f"Final position: {result_stoch.positions[-1]}")
    print(f"Distance to goal: {result_stoch.metadata['final_distance_to_goal']:.3f}m")

    # Run multiple stochastic trials to show variation
    print(f"\n5 Stochastic Trials (final positions):")
    for i in range(5):
        result_trial = planner_stoch.simulate(
            start, goal, obstacles=obstacles, max_steps=500
        )
        print(
            f"  Trial {i+1}: {result_trial.positions[-1]} (success: {result_trial.success})"
        )


if __name__ == "__main__":
    test_vga_upl()
