"""
Variable Goal Approach (VGA) Implementation

Implements the VGA algorithm from the paper:
"Variable Goal Approach (VGA): Incorporating Human Intelligence into
Microscopic Pedestrian Dynamics Models"

The VGA algorithm:
1. Detects nearest obstacles along path to goal
2. Forms obstacle clusters if obstacles are close together
3. Identifies left and right tangential obstacles
4. Places variable goals at personal distance from tangents
5. Selects goal based on deviation from straight path
6. Optionally uses probabilistic selection for stochasticity

Author: [Your Name]
Date: December 2025
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from model_base import NavigationModel, SimulationResult


@dataclass
class Obstacle:
    """Representation of a circular obstacle"""

    position: np.ndarray  # (x, y)
    radius: float

    def distance_to(self, point: np.ndarray) -> float:
        """Distance from obstacle surface to point"""
        return np.linalg.norm(self.position - point) - self.radius

    def contains_point(self, point: np.ndarray) -> bool:
        """Check if point is inside obstacle"""
        return np.linalg.norm(self.position - point) < self.radius


@dataclass
class ObstacleCluster:
    """Group of obstacles treated as one entity"""

    obstacles: List[Obstacle]

    def get_bounding_circle(self) -> Tuple[np.ndarray, float]:
        """
        Get minimum bounding circle for cluster

        Returns:
            (center, radius) of bounding circle
        """
        if len(self.obstacles) == 1:
            obs = self.obstacles[0]
            return obs.position, obs.radius

        # Simple implementation: center of mass with max radius
        positions = np.array([obs.position for obs in self.obstacles])
        center = np.mean(positions, axis=0)

        # Find max distance from center to any obstacle surface
        max_radius = 0.0
        for obs in self.obstacles:
            dist = np.linalg.norm(obs.position - center) + obs.radius
            max_radius = max(max_radius, dist)

        return center, max_radius

    def get_tangent_points(
        self, from_point: np.ndarray, personal_distance: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get left and right tangent points to cluster from observer's perspective

        Args:
            from_point: Observer position (pedestrian)
            personal_distance: Extra clearance distance

        Returns:
            (left_tangent, right_tangent) positions
        """
        center, radius = self.get_bounding_circle()

        # Add personal distance to radius
        effective_radius = radius + personal_distance

        # Vector from observer to cluster center
        to_center = center - from_point
        dist_to_center = np.linalg.norm(to_center)

        if dist_to_center < effective_radius:
            # Inside or very close to cluster - use simple left/right
            perpendicular = np.array([-to_center[1], to_center[0]])
            perpendicular = perpendicular / np.linalg.norm(perpendicular)

            left_tangent = center + perpendicular * effective_radius
            right_tangent = center - perpendicular * effective_radius
            return left_tangent, right_tangent

        # Calculate tangent angle
        # sin(theta) = effective_radius / dist_to_center
        sin_theta = effective_radius / dist_to_center
        cos_theta = np.sqrt(1 - sin_theta**2)

        # Direction to center (normalized)
        direction = to_center / dist_to_center

        # Rotation matrices for left and right tangents
        # Left tangent: rotate counterclockwise
        angle_left = np.arcsin(sin_theta)
        rot_left = np.array(
            [
                [np.cos(angle_left), -np.sin(angle_left)],
                [np.sin(angle_left), np.cos(angle_left)],
            ]
        )

        # Right tangent: rotate clockwise
        angle_right = -angle_left
        rot_right = np.array(
            [
                [np.cos(angle_right), -np.sin(angle_right)],
                [np.sin(angle_right), np.cos(angle_right)],
            ]
        )

        # Calculate tangent directions
        left_direction = rot_left @ direction
        right_direction = rot_right @ direction

        # Tangent points on cluster surface
        distance_to_tangent = np.sqrt(dist_to_center**2 - effective_radius**2)
        left_tangent = from_point + left_direction * distance_to_tangent
        right_tangent = from_point + right_direction * distance_to_tangent

        return left_tangent, right_tangent


class VGAPlanner(NavigationModel):
    """
    Variable Goal Approach (VGA) navigation planner

    Usage:
        planner = VGAPlanner(
            pedestrian_radius=0.25,
            personal_distance=0.5,
            max_speed=1.34,
            use_probabilistic=False
        )
        result = planner.simulate(start_pos, goal_pos, obstacles)
    """

    def __init__(
        self,
        pedestrian_radius: float = 0.25,
        personal_distance: float = 0.5,
        max_speed: float = 1.34,  # Average human walking speed (m/s)
        clustering_threshold: float = 0.5,  # Max distance between obstacles to cluster
        visibility_angle: float = 100.0,  # degrees (±100° = 200° total FOV)
        use_probabilistic: bool = False,  # Use probabilistic goal selection
        **kwargs,
    ):
        """
        Args:
            pedestrian_radius: Pedestrian body radius (meters)
            personal_distance: Desired clearance from obstacles (meters)
            max_speed: Maximum walking speed (m/s)
            clustering_threshold: Distance threshold for clustering obstacles
            visibility_angle: Half-angle of visibility cone (degrees)
            use_probabilistic: Whether to use probabilistic goal selection
        """
        super().__init__(name="VGA", **kwargs)

        self.pedestrian_radius = pedestrian_radius
        self.personal_distance = personal_distance
        self.max_speed = max_speed
        self.clustering_threshold = clustering_threshold
        self.visibility_angle = np.deg2rad(visibility_angle)
        self.use_probabilistic = use_probabilistic

        # Internal state
        self.current_position = None
        self.current_velocity = None
        self.current_goal = None  # Current variable goal (or final goal)
        self.final_goal = None
        self.obstacles = []
        self.obstacle_clusters = []

    def reset(self, start_pos: np.ndarray, goal_pos: np.ndarray, **kwargs):
        """
        Reset planner for new episode

        Args:
            start_pos: Starting position (x, y)
            goal_pos: Final goal position (x, y)
            **kwargs: Additional parameters
                - obstacles: List of Obstacle objects or dicts with 'position' and 'radius'
        """
        self.current_position = start_pos.copy()
        self.current_velocity = np.zeros(2)
        self.final_goal = goal_pos.copy()
        self.current_goal = goal_pos.copy()  # Start with final goal

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

        # Create initial clusters
        self._update_obstacle_clusters()

    def _update_obstacle_clusters(self):
        """
        Group nearby obstacles into clusters

        Obstacles closer than clustering_threshold are treated as one entity
        """
        if not self.obstacles:
            self.obstacle_clusters = []
            return

        # Simple greedy clustering
        unclustered = self.obstacles.copy()
        self.obstacle_clusters = []

        while unclustered:
            # Start new cluster with first obstacle
            cluster_obs = [unclustered.pop(0)]

            # Find all obstacles close to any in cluster
            changed = True
            while changed:
                changed = False
                for i in range(len(unclustered) - 1, -1, -1):
                    obs = unclustered[i]
                    # Check distance to any obstacle in cluster
                    for cluster_member in cluster_obs:
                        dist = np.linalg.norm(obs.position - cluster_member.position)
                        dist -= obs.radius + cluster_member.radius

                        if dist < self.clustering_threshold:
                            cluster_obs.append(unclustered.pop(i))
                            changed = True
                            break

            self.obstacle_clusters.append(ObstacleCluster(cluster_obs))

    def _is_visible(self, point: np.ndarray) -> bool:
        """
        Check if point is within visibility cone

        Args:
            point: Point to check

        Returns:
            True if point is visible from current position
        """
        to_point = point - self.current_position
        to_goal = self.final_goal - self.current_position

        # Angle between direction to point and direction to goal
        angle = np.arctan2(
            to_point[0] * to_goal[1] - to_point[1] * to_goal[0],
            to_point[0] * to_goal[0] + to_point[1] * to_goal[1],
        )

        return abs(angle) <= self.visibility_angle

    def _find_nearest_cluster_on_path(self) -> Optional[ObstacleCluster]:
        """
        Find nearest obstacle cluster that is on path to current goal

        Returns:
            Nearest cluster on path, or None if path is clear
        """
        if not self.obstacle_clusters:
            return None

        # Direction to current goal
        to_goal = self.current_goal - self.current_position
        dist_to_goal = np.linalg.norm(to_goal)

        if dist_to_goal < 0.01:
            return None

        to_goal_normalized = to_goal / dist_to_goal

        # Find cluster closest to line from current position to goal
        nearest_cluster = None
        min_distance = float("inf")

        for cluster in self.obstacle_clusters:
            center, radius = cluster.get_bounding_circle()

            # Check if cluster is ahead of us
            to_cluster = center - self.current_position
            projection = np.dot(to_cluster, to_goal_normalized)

            if projection < 0:
                continue  # Behind us

            if projection > dist_to_goal:
                continue  # Beyond goal

            # Distance from line to cluster center
            perpendicular_dist = np.linalg.norm(
                to_cluster - projection * to_goal_normalized
            )

            # Check if cluster blocks path (accounting for personal distance)
            effective_radius = radius + self.personal_distance + self.pedestrian_radius

            if perpendicular_dist < effective_radius:
                dist_to_cluster = np.linalg.norm(to_cluster) - radius
                if dist_to_cluster < min_distance:
                    min_distance = dist_to_cluster
                    nearest_cluster = cluster

        return nearest_cluster

    def _select_variable_goal(self, cluster: ObstacleCluster) -> np.ndarray:
        """
        Select left or right variable goal around cluster

        Args:
            cluster: Obstacle cluster to avoid

        Returns:
            Selected variable goal position
        """
        # Get tangent points
        left_tangent, right_tangent = cluster.get_tangent_points(
            self.current_position, self.personal_distance
        )

        # Check visibility
        left_visible = self._is_visible(left_tangent)
        right_visible = self._is_visible(right_tangent)

        # If only one is visible, choose it
        if left_visible and not right_visible:
            return left_tangent
        if right_visible and not left_visible:
            return right_tangent

        # Both visible (or both not visible) - choose based on deviation
        # Deviation = angle from current goal direction
        to_final_goal = self.final_goal - self.current_position
        to_final_goal /= np.linalg.norm(to_final_goal)

        to_left = left_tangent - self.current_position
        to_left /= np.linalg.norm(to_left)

        to_right = right_tangent - self.current_position
        to_right /= np.linalg.norm(to_right)

        # Calculate deviations (angles)
        deviation_left = np.arccos(np.clip(np.dot(to_left, to_final_goal), -1, 1))
        deviation_right = np.arccos(np.clip(np.dot(to_right, to_final_goal), -1, 1))

        # Select goal
        if self.use_probabilistic:
            # Probabilistic selection: less deviation = higher probability
            # P_left = deviation_right / (deviation_left + deviation_right)
            total_deviation = deviation_left + deviation_right
            if total_deviation < 1e-6:
                # Both deviations are zero - choose randomly
                return left_tangent if np.random.rand() < 0.5 else right_tangent

            p_left = deviation_right / total_deviation
            return left_tangent if np.random.rand() < p_left else right_tangent
        else:
            # Deterministic: choose minimum deviation
            return left_tangent if deviation_left < deviation_right else right_tangent

    def _update_variable_goal(self):
        """
        Update current variable goal based on VGA algorithm

        Checks if there's an obstacle on path to current goal,
        and if so, selects a new variable goal to avoid it
        """
        # Check if we reached current variable goal
        dist_to_current_goal = np.linalg.norm(self.current_goal - self.current_position)

        if dist_to_current_goal < 0.5:  # Within 0.5m of variable goal
            # Check if current goal is final goal
            if np.linalg.norm(self.current_goal - self.final_goal) < 0.1:
                return  # Already heading to final goal
            else:
                # Variable goal reached, switch to final goal or find next variable goal
                self.current_goal = self.final_goal.copy()

        # Find nearest cluster on path to current goal
        blocking_cluster = self._find_nearest_cluster_on_path()

        if blocking_cluster is not None:
            # Select new variable goal to avoid cluster
            new_goal = self._select_variable_goal(blocking_cluster)
            self.current_goal = new_goal

    def predict_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        Predict desired velocity toward current goal

        Args:
            observation: Dictionary with current state (not used - we maintain state internally)

        Returns:
            Desired velocity [vx, vy]
        """
        # Update variable goal based on VGA logic
        self._update_variable_goal()

        # Compute desired velocity toward current goal
        to_goal = self.current_goal - self.current_position
        distance = np.linalg.norm(to_goal)

        if distance < 0.01:
            return np.zeros(2)

        # Desired velocity: direction * max_speed
        desired_velocity = (to_goal / distance) * self.max_speed

        return desired_velocity

    def step(self, action: np.ndarray, dt: float) -> Dict[str, Any]:
        """
        Update position based on desired velocity

        Args:
            action: Desired velocity [vx, vy]
            dt: Time step

        Returns:
            Dictionary with updated state
        """
        # Simple integration: position += velocity * dt
        self.current_velocity = action
        self.current_position += action * dt

        # Check for collisions
        collision = False
        for obs in self.obstacles:
            if obs.distance_to(self.current_position) < self.pedestrian_radius:
                collision = True
                break

        return {
            "position": self.current_position.copy(),
            "velocity": self.current_velocity.copy(),
            "collision": collision,
            "current_goal": self.current_goal.copy(),
        }


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create VGA planner
    planner = VGAPlanner(
        pedestrian_radius=0.25,
        personal_distance=0.5,
        max_speed=1.34,
        use_probabilistic=False,
    )

    # Define scenario
    start_pos = np.array([0.0, 5.0])
    goal_pos = np.array([20.0, 5.0])

    # Create obstacles
    obstacles = [
        {"position": [10.0, 5.0], "radius": 0.5},
        {"position": [10.5, 5.0], "radius": 0.5},  # Close to first - will cluster
    ]

    # Run simulation
    print("\n=== VGA Planner Test ===\n")
    result = planner.simulate(
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacles=obstacles,
        max_steps=500,
        dt=0.1,
        goal_threshold=0.5,
    )

    # Print results
    print(f"Success: {result.success}")
    print(f"Travel time: {result.travel_time:.2f}s")
    print(f"Path length: {result.path_length:.2f}m")
    print(f"Collisions: {result.num_collisions}")

    # Plot trajectory
    plt.figure(figsize=(12, 6))

    # Plot trajectory
    plt.plot(
        result.positions[:, 0],
        result.positions[:, 1],
        "b-",
        linewidth=2,
        label="VGA Trajectory",
    )

    # Plot start and goal
    plt.plot(start_pos[0], start_pos[1], "go", markersize=10, label="Start")
    plt.plot(goal_pos[0], goal_pos[1], "r*", markersize=15, label="Goal")

    # Plot obstacles
    for obs in obstacles:
        circle = plt.Circle(obs["position"], obs["radius"], color="gray", alpha=0.5)
        plt.gca().add_patch(circle)

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("VGA Planner Test")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("vga_test.png", dpi=150)
    print("\nTrajectory plot saved to vga_test.png")
