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
        **kwargs
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
        self.personal_distance = kwargs.get('personal_distance', 0.5)
        self.clustering_threshold = kwargs.get('clustering_threshold', 1.0)
        self.goal_switch_threshold = kwargs.get('goal_switch_threshold', 0.3)
        self.detection_cone_angle = kwargs.get('detection_cone_angle', 60.0)  # degrees
        
        # UPL physics engine
        self.upl = UPLPhysics(upl_params)
        
        # State variables
        self.current_position = None
        self.current_velocity = None
        self.final_goal = None
        self.current_goal = None  # Variable goal
        self.obstacles = []
    
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
        
    def reset(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        **kwargs
    ):
        """Reset planner for new simulation."""
        self.current_position = start_pos.copy()
        self.current_velocity = np.zeros(2)
        self.final_goal = goal_pos.copy()
        self.current_goal = goal_pos.copy()
        
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
                        radius=obs_data.get("radius", 0.3)
                    )
                )
            elif isinstance(obs_data, (list, tuple, np.ndarray)):
                # Assume [x, y, radius] format
                self.obstacles.append(
                    Obstacle(
                        position=np.array(obs_data[:2]),
                        radius=obs_data[2] if len(obs_data) > 2 else 0.3
                    )
                )
    
    def _detect_obstacles_in_cone(self) -> List[Obstacle]:
        """
        Detect obstacles in the detection cone toward current goal.
        
        Returns:
            List of obstacles that might block the path
        """
        # Direction to current goal
        to_goal = self.current_goal - self.current_position
        goal_dist = np.linalg.norm(to_goal)
        
        if goal_dist < 1e-6:
            return []
        
        to_goal_norm = to_goal / goal_dist
        
        # Cone angle in radians
        cone_angle_rad = np.radians(self.detection_cone_angle)
        
        detected = []
        for obs in self.obstacles:
            # Vector to obstacle
            to_obs = obs.position - self.current_position
            obs_dist = np.linalg.norm(to_obs)
            
            if obs_dist < 1e-6:
                continue
            
            to_obs_norm = to_obs / obs_dist
            
            # Check if obstacle is within cone angle
            dot_product = np.dot(to_goal_norm, to_obs_norm)
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
            
            # Check if obstacle is ahead and within cone
            if angle < cone_angle_rad and obs_dist < goal_dist + obs.radius:
                detected.append(obs)
        
        return detected
    
    def _find_tangent_points(
        self,
        obstacle: Obstacle
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find left and right tangent points around an obstacle.
        
        Args:
            obstacle: The obstacle to navigate around
            
        Returns:
            (left_tangent, right_tangent) positions
        """
        center = obstacle.position
        effective_radius = obstacle.radius + self.personal_distance
        
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
        left_dir = np.array([
            cos_angle * to_center_norm[0] - sin_angle * to_center_norm[1],
            sin_angle * to_center_norm[0] + cos_angle * to_center_norm[1]
        ])
        
        # Right tangent (clockwise rotation)
        right_dir = np.array([
            cos_angle * to_center_norm[0] + sin_angle * to_center_norm[1],
            -sin_angle * to_center_norm[0] + cos_angle * to_center_norm[1]
        ])
        
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
    
    def _select_variable_goal(self) -> np.ndarray:
        """
        Select the next variable goal using VGA logic.
        
        Returns:
            Selected goal position
        """
        # Check if we've reached current goal
        if np.linalg.norm(self.current_position - self.current_goal) < self.goal_switch_threshold:
            self.current_goal = self.final_goal.copy()
        
        # Detect obstacles blocking path
        blocking_obstacles = self._detect_obstacles_in_cone()
        
        if not blocking_obstacles:
            # No obstacles - aim for final goal
            return self.final_goal.copy()
        
        # Find nearest obstacle
        nearest_obs = min(
            blocking_obstacles,
            key=lambda obs: obs.distance_to(self.current_position)
        )
        
        # Get tangent points
        left_tangent, right_tangent = self._find_tangent_points(nearest_obs)
        
        # Calculate deviations
        left_deviation = self._calculate_deviation(left_tangent)
        right_deviation = self._calculate_deviation(right_tangent)
        
        # Goal selection
        if self.use_probabilistic:
            # Stochastic mode: probabilistic selection
            total_dev = left_deviation + right_deviation + 1e-6
            prob_right = left_deviation / total_dev  # Less deviation = higher probability
            
            if np.random.random() < prob_right:
                return right_tangent
            else:
                return left_tangent
        else:
            # Deterministic mode: choose minimum deviation
            if left_deviation < right_deviation:
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
        
        # UPL: Move toward current goal using physics
        # Convert obstacles to UPL format: list of (position, radius)
        upl_obstacles = [
            (obs.position, obs.radius)
            for obs in self.obstacles
        ]
        
        # Perform UPL physics step
        new_pos, new_vel, force = self.upl.step(
            self.current_position,
            self.current_velocity,
            self.current_goal,
            upl_obstacles
        )
        
        # Update state
        self.current_position = new_pos
        self.current_velocity = new_vel
        
        return False
    
    def simulate(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        max_steps: int = 1000,
        **kwargs
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
                'mode': 'stochastic' if self.use_probabilistic else 'deterministic',
                'final_distance_to_goal': final_dist,
                'num_obstacles': len(self.obstacles)
            }
        )


def test_vga_upl():
    """Test VGA+UPL integration."""
    print("Testing VGA+UPL Integration...")
    
    # Create planner
    planner = VGAUPLPlanner(use_probabilistic=False)
    
    # Test scenario: same as UPL test but with VGA intelligence
    start = np.array([0.0, 0.0])
    goal = np.array([10.0, 0.0])
    obstacles = [
        {'position': [5.0, 0.0], 'radius': 0.5}  # Blocking direct path
    ]
    
    # Simulate
    result = planner.simulate(start, goal, obstacles=obstacles, max_steps=500)
    
    print(f"\nDeterministic Mode:")
    print(f"Success: {result.success}")
    print(f"Steps: {len(result.positions)}")
    print(f"Final position: {result.positions[-1]}")
    print(f"Distance to goal: {result.metadata['final_distance_to_goal']:.3f}m")
    
    # Test stochastic mode
    planner_stoch = VGAUPLPlanner(use_probabilistic=True)
    result_stoch = planner_stoch.simulate(start, goal, obstacles=obstacles, max_steps=500)
    
    print(f"\nStochastic Mode:")
    print(f"Success: {result_stoch.success}")
    print(f"Steps: {len(result_stoch.positions)}")
    print(f"Final position: {result_stoch.positions[-1]}")
    print(f"Distance to goal: {result_stoch.metadata['final_distance_to_goal']:.3f}m")
    
    # Run multiple stochastic trials to show variation
    print(f"\n5 Stochastic Trials (final positions):")
    for i in range(5):
        result_trial = planner_stoch.simulate(start, goal, obstacles=obstacles, max_steps=500)
        print(f"  Trial {i+1}: {result_trial.positions[-1]} (success: {result_trial.success})")


if __name__ == "__main__":
    test_vga_upl()
