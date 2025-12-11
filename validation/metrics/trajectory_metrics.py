"""
Trajectory Metrics Module

Computes core trajectory quality metrics for pedestrian navigation:
- Path smoothness (curvature variation)
- Speed deviation from desired walking speed
- Oscillation (lateral movement perpendicular to goal direction)
- Travel time and path efficiency
- Collision and safety metrics

These metrics are used to evaluate navigation quality and compare
models against human experimental data.

Author: Abdallah Jamal
Date: December 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


@dataclass
class TrajectoryMetrics:
    """Container for all trajectory metrics"""
    
    # Path Quality Metrics
    path_length: float  # Total distance traveled (m)
    straight_line_distance: float  # Direct distance start to goal (m)
    path_efficiency: float  # straight_line_distance / path_length (0-1)
    
    # Smoothness Metrics
    curvature_mean: float  # Average path curvature (1/m)
    curvature_std: float  # Standard deviation of curvature
    curvature_max: float  # Maximum curvature
    smoothness_index: float  # Lower is smoother (curvature_std)
    
    # Speed Metrics
    avg_speed: float  # Average speed (m/s)
    speed_std: float  # Standard deviation of speed
    speed_deviation_from_desired: float  # |avg_speed - desired_speed|
    desired_speed: float  # Target walking speed (typically 1.34 m/s)
    
    # Oscillation Metrics
    lateral_deviation_mean: float  # Mean lateral deviation from direct path (m)
    lateral_deviation_std: float  # Std of lateral deviation (m)
    oscillation_frequency: float  # Number of direction changes per meter
    
    # Time Metrics
    travel_time: float  # Total time from start to goal (s)
    time_to_first_progress: float  # Time until first movement toward goal (s)
    
    # Safety Metrics
    min_obstacle_clearance: float  # Closest approach to any obstacle (m)
    time_in_danger_zone: float  # Time spent < 0.5m from obstacles (s)
    num_collisions: int  # Number of obstacle contacts
    collision_severity: float  # Sum of penetration depths (m)
    
    # Success Metrics
    success: bool  # Did trajectory reach goal?
    stuck: bool  # Did trajectory get stuck (no progress for long time)?
    goal_distance_final: float  # Distance from goal at end (m)
    
    # Comfort Index (composite metric)
    comfort_index: float  # 0-1, higher is more comfortable


def compute_path_length(positions: np.ndarray) -> float:
    """
    Compute total path length
    
    Args:
        positions: (N, 2) array of positions
        
    Returns:
        Total distance traveled in meters
    """
    if len(positions) < 2:
        return 0.0
    
    deltas = np.diff(positions, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    return float(np.sum(distances))


def compute_curvature(positions: np.ndarray, smooth: bool = True) -> np.ndarray:
    """
    Compute path curvature at each point
    
    Curvature κ = |x' * y'' - y' * x''| / (x'^2 + y'^2)^(3/2)
    
    Args:
        positions: (N, 2) array of positions
        smooth: Whether to smooth positions before computing curvature
        
    Returns:
        (N-2,) array of curvature values (1/m)
    """
    if len(positions) < 3:
        return np.array([])
    
    pos = positions.copy()
    
    # Smooth trajectory to reduce noise
    if smooth and len(pos) >= 5:
        window_length = min(5, len(pos) if len(pos) % 2 == 1 else len(pos) - 1)
        if window_length >= 3:
            pos[:, 0] = savgol_filter(pos[:, 0], window_length, 2)
            pos[:, 1] = savgol_filter(pos[:, 1], window_length, 2)
    
    # First derivatives (velocity)
    dx = np.gradient(pos[:, 0])
    dy = np.gradient(pos[:, 1])
    
    # Second derivatives (acceleration)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Curvature formula
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx**2 + dy**2)**(3/2)
    
    # Avoid division by zero
    denominator = np.maximum(denominator, 1e-10)
    
    curvature = numerator / denominator
    
    return curvature


def compute_lateral_deviation(
    positions: np.ndarray, 
    start_pos: np.ndarray, 
    goal_pos: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Compute lateral deviation from straight line to goal
    
    Args:
        positions: (N, 2) array of positions
        start_pos: Starting position
        goal_pos: Goal position
        
    Returns:
        lateral_deviations: (N,) array of signed lateral deviations (m)
        oscillation_frequency: Number of sign changes per meter
    """
    if len(positions) < 2:
        return np.array([]), 0.0
    
    # Direction vector from start to goal
    to_goal = goal_pos - start_pos
    goal_distance = np.linalg.norm(to_goal)
    
    if goal_distance < 1e-6:
        return np.zeros(len(positions)), 0.0
    
    to_goal_normalized = to_goal / goal_distance
    
    # Perpendicular vector (rotate 90 degrees)
    perpendicular = np.array([-to_goal_normalized[1], to_goal_normalized[0]])
    
    # Compute lateral deviation for each position
    deviations_from_start = positions - start_pos
    lateral_deviations = np.dot(deviations_from_start, perpendicular)
    
    # Count direction changes (oscillations)
    sign_changes = np.sum(np.diff(np.sign(lateral_deviations)) != 0)
    path_length = compute_path_length(positions)
    oscillation_frequency = sign_changes / max(path_length, 1e-6)
    
    return lateral_deviations, oscillation_frequency


def compute_speed_profile(
    positions: np.ndarray, 
    timestamps: np.ndarray
) -> np.ndarray:
    """
    Compute instantaneous speed at each point
    
    Args:
        positions: (N, 2) array of positions
        timestamps: (N,) array of timestamps
        
    Returns:
        (N-1,) array of speeds (m/s)
    """
    if len(positions) < 2:
        return np.array([])
    
    deltas = np.diff(positions, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    
    dt = np.diff(timestamps)
    dt = np.maximum(dt, 1e-6)  # Avoid division by zero
    
    speeds = distances / dt
    
    return speeds


def compute_obstacle_clearances(
    positions: np.ndarray,
    obstacles: List[Dict],
    agent_radius: float = 0.225
) -> Tuple[float, np.ndarray]:
    """
    Compute minimum clearance to obstacles
    
    Args:
        positions: (N, 2) array of positions
        obstacles: List of obstacle dicts with 'position' and 'radius'
        agent_radius: Radius of agent body
        
    Returns:
        min_clearance: Minimum clearance to any obstacle (m)
        clearances: (N,) array of clearance at each position
    """
    if len(obstacles) == 0:
        return float('inf'), np.full(len(positions), float('inf'))
    
    clearances = []
    
    for pos in positions:
        min_dist = float('inf')
        
        for obs in obstacles:
            obs_pos = np.array(obs['position'])
            obs_radius = obs.get('radius', 0.3)
            
            # Distance from agent center to obstacle surface
            dist_to_surface = np.linalg.norm(pos - obs_pos) - obs_radius - agent_radius
            min_dist = min(min_dist, dist_to_surface)
        
        clearances.append(max(0.0, min_dist))  # Negative means collision
    
    clearances = np.array(clearances)
    min_clearance = float(np.min(clearances))
    
    return min_clearance, clearances


def compute_comfort_index(
    smoothness: float,
    speed_deviation: float,
    min_clearance: float,
    lateral_deviation_std: float,
    desired_clearance: float = 0.5,
    desired_speed: float = 1.34
) -> float:
    """
    Compute composite comfort index (0-1, higher is better)
    
    Combines multiple factors:
    - Path smoothness (low curvature variation)
    - Speed stability (close to desired speed)
    - Safety clearance (distance from obstacles)
    - Low lateral oscillation
    
    Args:
        smoothness: Curvature std (lower is better)
        speed_deviation: |avg_speed - desired_speed|
        min_clearance: Minimum obstacle clearance
        lateral_deviation_std: Lateral oscillation
        desired_clearance: Target clearance (default 0.5m)
        desired_speed: Target speed (default 1.34 m/s)
        
    Returns:
        Comfort index 0-1 (higher is more comfortable)
    """
    # Smoothness component (0-1, higher is better)
    # Assume smoothness < 0.1 is excellent, > 1.0 is poor
    smoothness_score = np.exp(-smoothness / 0.3)
    
    # Speed component (0-1, higher is better)
    # Penalize deviation from desired speed
    speed_score = np.exp(-speed_deviation / 0.3)
    
    # Clearance component (0-1, higher is better)
    # Reward maintaining safe distance
    if min_clearance < 0:  # Collision
        clearance_score = 0.0
    elif min_clearance >= desired_clearance:
        clearance_score = 1.0
    else:
        clearance_score = min_clearance / desired_clearance
    
    # Oscillation component (0-1, higher is better)
    # Lower lateral deviation is better
    oscillation_score = np.exp(-lateral_deviation_std / 0.2)
    
    # Weighted average
    comfort_index = (
        0.25 * smoothness_score +
        0.25 * speed_score +
        0.30 * clearance_score +
        0.20 * oscillation_score
    )
    
    return float(np.clip(comfort_index, 0.0, 1.0))


def compute_trajectory_metrics(
    positions: np.ndarray,
    timestamps: np.ndarray,
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    obstacles: Optional[List[Dict]] = None,
    desired_speed: float = 1.34,
    goal_threshold: float = 0.5,
    agent_radius: float = 0.225,
    stuck_threshold: float = 5.0,
    stuck_distance: float = 0.1
) -> TrajectoryMetrics:
    """
    Compute all trajectory metrics
    
    Args:
        positions: (N, 2) array of positions
        timestamps: (N,) array of timestamps
        start_pos: Starting position
        goal_pos: Goal position
        obstacles: List of obstacle dicts with 'position' and 'radius'
        desired_speed: Target walking speed (m/s)
        goal_threshold: Distance to consider goal reached (m)
        agent_radius: Radius of agent body (m)
        stuck_threshold: Time without progress to consider stuck (s)
        stuck_distance: Distance threshold for "no progress" (m)
        
    Returns:
        TrajectoryMetrics object with all computed metrics
    """
    if obstacles is None:
        obstacles = []
    
    N = len(positions)
    
    # Basic checks
    if N < 2:
        return TrajectoryMetrics(
            path_length=0.0,
            straight_line_distance=np.linalg.norm(goal_pos - start_pos),
            path_efficiency=0.0,
            curvature_mean=0.0,
            curvature_std=0.0,
            curvature_max=0.0,
            smoothness_index=0.0,
            avg_speed=0.0,
            speed_std=0.0,
            speed_deviation_from_desired=desired_speed,
            desired_speed=desired_speed,
            lateral_deviation_mean=0.0,
            lateral_deviation_std=0.0,
            oscillation_frequency=0.0,
            travel_time=timestamps[-1] - timestamps[0] if N > 0 else 0.0,
            time_to_first_progress=0.0,
            min_obstacle_clearance=float('inf'),
            time_in_danger_zone=0.0,
            num_collisions=0,
            collision_severity=0.0,
            success=False,
            stuck=True,
            goal_distance_final=np.linalg.norm(positions[-1] - goal_pos) if N > 0 else float('inf'),
            comfort_index=0.0
        )
    
    # Path metrics
    path_length = compute_path_length(positions)
    straight_line_distance = np.linalg.norm(goal_pos - start_pos)
    path_efficiency = straight_line_distance / max(path_length, 1e-6)
    
    # Curvature metrics
    curvature = compute_curvature(positions, smooth=True)
    if len(curvature) > 0:
        curvature_mean = float(np.mean(curvature))
        curvature_std = float(np.std(curvature))
        curvature_max = float(np.max(curvature))
        smoothness_index = curvature_std
    else:
        curvature_mean = curvature_std = curvature_max = smoothness_index = 0.0
    
    # Speed metrics
    speeds = compute_speed_profile(positions, timestamps)
    if len(speeds) > 0:
        avg_speed = float(np.mean(speeds))
        speed_std = float(np.std(speeds))
    else:
        avg_speed = speed_std = 0.0
    speed_deviation_from_desired = abs(avg_speed - desired_speed)
    
    # Lateral deviation and oscillation
    lateral_deviations, oscillation_freq = compute_lateral_deviation(
        positions, start_pos, goal_pos
    )
    if len(lateral_deviations) > 0:
        lateral_deviation_mean = float(np.mean(np.abs(lateral_deviations)))
        lateral_deviation_std = float(np.std(lateral_deviations))
    else:
        lateral_deviation_mean = lateral_deviation_std = 0.0
    
    # Time metrics
    travel_time = float(timestamps[-1] - timestamps[0])
    
    # Time to first progress (when agent first moves toward goal)
    distances_to_goal = np.linalg.norm(positions - goal_pos, axis=1)
    progress_made = distances_to_goal[0] - distances_to_goal
    first_progress_idx = np.where(progress_made > 0.1)[0]
    if len(first_progress_idx) > 0:
        time_to_first_progress = float(timestamps[first_progress_idx[0]] - timestamps[0])
    else:
        time_to_first_progress = travel_time
    
    # Obstacle clearance
    min_clearance, clearances = compute_obstacle_clearances(
        positions, obstacles, agent_radius
    )
    
    # Time in danger zone (< 0.5m from obstacles)
    danger_zone_threshold = 0.5
    in_danger = clearances < danger_zone_threshold
    time_in_danger = float(np.sum(in_danger) * np.mean(np.diff(timestamps)))
    
    # Collisions (clearance < 0)
    collision_mask = clearances < 0
    num_collisions = int(np.sum(collision_mask))
    collision_severity = float(np.sum(np.abs(clearances[collision_mask]))) if num_collisions > 0 else 0.0
    
    # Success and stuck detection
    goal_distance_final = float(np.linalg.norm(positions[-1] - goal_pos))
    success = goal_distance_final < goal_threshold
    
    # Check if stuck (no progress for stuck_threshold seconds)
    stuck = False
    if N > 10:
        window_size = int(stuck_threshold / np.mean(np.diff(timestamps)))
        for i in range(N - window_size):
            segment = positions[i:i+window_size]
            segment_travel = compute_path_length(segment)
            if segment_travel < stuck_distance:
                stuck = True
                break
    
    # Comfort index
    comfort_index = compute_comfort_index(
        smoothness=smoothness_index,
        speed_deviation=speed_deviation_from_desired,
        min_clearance=min_clearance,
        lateral_deviation_std=lateral_deviation_std,
        desired_clearance=0.5,
        desired_speed=desired_speed
    )
    
    return TrajectoryMetrics(
        path_length=path_length,
        straight_line_distance=straight_line_distance,
        path_efficiency=path_efficiency,
        curvature_mean=curvature_mean,
        curvature_std=curvature_std,
        curvature_max=curvature_max,
        smoothness_index=smoothness_index,
        avg_speed=avg_speed,
        speed_std=speed_std,
        speed_deviation_from_desired=speed_deviation_from_desired,
        desired_speed=desired_speed,
        lateral_deviation_mean=lateral_deviation_mean,
        lateral_deviation_std=lateral_deviation_std,
        oscillation_frequency=oscillation_freq,
        travel_time=travel_time,
        time_to_first_progress=time_to_first_progress,
        min_obstacle_clearance=min_clearance,
        time_in_danger_zone=time_in_danger,
        num_collisions=num_collisions,
        collision_severity=collision_severity,
        success=success,
        stuck=stuck,
        goal_distance_final=goal_distance_final,
        comfort_index=comfort_index
    )


def metrics_to_dict(metrics: TrajectoryMetrics) -> Dict:
    """Convert TrajectoryMetrics to dictionary"""
    return {
        'path_length': metrics.path_length,
        'straight_line_distance': metrics.straight_line_distance,
        'path_efficiency': metrics.path_efficiency,
        'curvature_mean': metrics.curvature_mean,
        'curvature_std': metrics.curvature_std,
        'curvature_max': metrics.curvature_max,
        'smoothness_index': metrics.smoothness_index,
        'avg_speed': metrics.avg_speed,
        'speed_std': metrics.speed_std,
        'speed_deviation_from_desired': metrics.speed_deviation_from_desired,
        'desired_speed': metrics.desired_speed,
        'lateral_deviation_mean': metrics.lateral_deviation_mean,
        'lateral_deviation_std': metrics.lateral_deviation_std,
        'oscillation_frequency': metrics.oscillation_frequency,
        'travel_time': metrics.travel_time,
        'time_to_first_progress': metrics.time_to_first_progress,
        'min_obstacle_clearance': metrics.min_obstacle_clearance,
        'time_in_danger_zone': metrics.time_in_danger_zone,
        'num_collisions': metrics.num_collisions,
        'collision_severity': metrics.collision_severity,
        'success': metrics.success,
        'stuck': metrics.stuck,
        'goal_distance_final': metrics.goal_distance_final,
        'comfort_index': metrics.comfort_index
    }


# Example usage
if __name__ == "__main__":
    # Synthetic trajectory for testing
    t = np.linspace(0, 10, 100)
    x = t
    y = 0.5 * np.sin(2 * np.pi * t / 5)  # Oscillating path
    
    positions = np.column_stack([x, y])
    timestamps = t
    start_pos = positions[0]
    goal_pos = np.array([10.0, 0.0])
    
    obstacles = [
        {'position': [5.0, 0.5], 'radius': 0.3},
        {'position': [5.0, -0.5], 'radius': 0.3}
    ]
    
    metrics = compute_trajectory_metrics(
        positions=positions,
        timestamps=timestamps,
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacles=obstacles
    )
    
    print("\n=== Trajectory Metrics Test ===\n")
    print(f"Path Length: {metrics.path_length:.2f}m")
    print(f"Path Efficiency: {metrics.path_efficiency:.2%}")
    print(f"Smoothness Index: {metrics.smoothness_index:.4f}")
    print(f"Avg Speed: {metrics.avg_speed:.2f} m/s")
    print(f"Lateral Deviation: {metrics.lateral_deviation_std:.3f}m")
    print(f"Comfort Index: {metrics.comfort_index:.2f}")
    print(f"Success: {metrics.success}")
