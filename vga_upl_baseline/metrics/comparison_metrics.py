"""
Comparison Metrics Module

Computes metrics for comparing model trajectories with human experimental data:
- Trajectory deviation (Hausdorff distance, Fréchet distance, DTW)
- Velocity profile similarity
- Spatial distribution metrics
- Human-likeness scores

These metrics quantify how closely a model reproduces human pedestrian behavior.

Author: Abdallah Jamal
Date: December 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.distance import directed_hausdorff
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, spearmanr, wasserstein_distance


@dataclass
class ComparisonMetrics:
    """Container for trajectory comparison metrics"""

    # Trajectory Similarity
    hausdorff_distance: float  # Max deviation between trajectories
    frechet_distance: float  # Continuous trajectory similarity
    dtw_distance: float  # Dynamic Time Warping distance

    # Path Similarity
    path_overlap: float  # Overlap between trajectory corridors (0-1)
    avg_point_distance: float  # Average closest point distance

    # Velocity Similarity
    velocity_correlation: float  # Pearson correlation of velocity profiles
    velocity_distribution_distance: float  # Wasserstein distance
    speed_profile_rmse: float  # RMSE of speed vs position

    # Spatial Distribution
    lateral_distribution_similarity: float  # KL divergence of lateral positions
    position_variance_ratio: float  # Ratio of position variances

    # Temporal Similarity
    travel_time_ratio: float  # Model time / Human time
    progress_correlation: float  # Correlation of distance-to-goal over time

    # Overall Human-Likeness Score
    human_likeness_score: float  # Composite score 0-1 (higher is more human-like)


def resample_trajectory(
    positions: np.ndarray, timestamps: np.ndarray, num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample trajectory to fixed number of points using interpolation

    Args:
        positions: (N, 2) array of positions
        timestamps: (N,) array of timestamps

    Returns:
        resampled_positions: (num_points, 2)
        resampled_timestamps: (num_points,)
    """
    if len(positions) < 2:
        return positions, timestamps

    # Create interpolation functions
    t_old = timestamps - timestamps[0]
    t_new = np.linspace(0, t_old[-1], num_points)

    interp_x = interp1d(t_old, positions[:, 0], kind="linear", fill_value="extrapolate")
    interp_y = interp1d(t_old, positions[:, 1], kind="linear", fill_value="extrapolate")

    new_positions = np.column_stack([interp_x(t_new), interp_y(t_new)])
    new_timestamps = t_new + timestamps[0]

    return new_positions, new_timestamps


def compute_hausdorff_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """
    Compute Hausdorff distance between two trajectories

    Hausdorff distance is the maximum of:
    - Max distance from any point in traj1 to closest point in traj2
    - Max distance from any point in traj2 to closest point in traj1

    Args:
        traj1: (N, 2) trajectory
        traj2: (M, 2) trajectory

    Returns:
        Hausdorff distance in meters
    """
    d1 = directed_hausdorff(traj1, traj2)[0]
    d2 = directed_hausdorff(traj2, traj1)[0]
    return float(max(d1, d2))


def compute_frechet_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """
    Compute discrete Fréchet distance between two trajectories

    Fréchet distance measures similarity of curves, considering ordering.
    It's like walking a dog on a leash - the leash length is the Fréchet distance.

    Args:
        traj1: (N, 2) trajectory
        traj2: (M, 2) trajectory

    Returns:
        Fréchet distance in meters
    """
    n, m = len(traj1), len(traj2)

    # Distance matrix
    dist_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dist_matrix[i, j] = np.linalg.norm(traj1[i] - traj2[j])

    # Dynamic programming to find minimum maximum distance
    ca = np.full((n, m), np.inf)
    ca[0, 0] = dist_matrix[0, 0]

    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], dist_matrix[i, 0])

    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], dist_matrix[0, j])

    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(
                min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]), dist_matrix[i, j]
            )

    return float(ca[n - 1, m - 1])


def compute_dtw_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """
    Compute Dynamic Time Warping distance between trajectories

    DTW finds optimal alignment between two sequences.
    Unlike Fréchet, it allows one-to-many mappings.

    Args:
        traj1: (N, 2) trajectory
        traj2: (M, 2) trajectory

    Returns:
        DTW distance
    """
    n, m = len(traj1), len(traj2)

    # Distance matrix
    dist_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dist_matrix[i, j] = np.linalg.norm(traj1[i] - traj2[j])

    # DTW dynamic programming
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_matrix[i - 1, j - 1]
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return float(dtw[n, m])


def compute_avg_point_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """
    Compute average minimum distance between trajectories

    For each point in traj1, find closest point in traj2, and average.

    Args:
        traj1: (N, 2) trajectory
        traj2: (M, 2) trajectory

    Returns:
        Average distance in meters
    """
    if len(traj1) == 0 or len(traj2) == 0:
        return float("inf")

    distances = []
    for p1 in traj1:
        dists_to_p1 = np.linalg.norm(traj2 - p1, axis=1)
        distances.append(np.min(dists_to_p1))

    return float(np.mean(distances))


def compute_path_overlap(
    traj1: np.ndarray, traj2: np.ndarray, corridor_width: float = 0.5
) -> float:
    """
    Compute overlap between trajectory corridors

    Creates a corridor around each trajectory and measures overlap.

    Args:
        traj1: (N, 2) trajectory
        traj2: (M, 2) trajectory
        corridor_width: Width of corridor around trajectory (m)

    Returns:
        Overlap ratio 0-1
    """
    # Simplified: count how many points of traj1 are within corridor of traj2
    if len(traj1) == 0 or len(traj2) == 0:
        return 0.0

    overlap_count = 0
    for p1 in traj1:
        dists = np.linalg.norm(traj2 - p1, axis=1)
        if np.min(dists) < corridor_width:
            overlap_count += 1

    overlap_ratio = overlap_count / len(traj1)
    return float(overlap_ratio)


def compute_velocity_correlation(
    velocities1: np.ndarray, velocities2: np.ndarray
) -> float:
    """
    Compute correlation between velocity profiles

    Args:
        velocities1: (N, 2) velocity vectors
        velocities2: (M, 2) velocity vectors

    Returns:
        Pearson correlation coefficient (-1 to 1)
    """
    if len(velocities1) < 2 or len(velocities2) < 2:
        return 0.0

    # Resample to same length
    n = min(len(velocities1), len(velocities2))

    if len(velocities1) != n:
        indices = np.linspace(0, len(velocities1) - 1, n).astype(int)
        velocities1 = velocities1[indices]

    if len(velocities2) != n:
        indices = np.linspace(0, len(velocities2) - 1, n).astype(int)
        velocities2 = velocities2[indices]

    # Compute speed (magnitude)
    speeds1 = np.linalg.norm(velocities1, axis=1)
    speeds2 = np.linalg.norm(velocities2, axis=1)

    # Pearson correlation
    if np.std(speeds1) < 1e-6 or np.std(speeds2) < 1e-6:
        return 0.0

    corr, _ = pearsonr(speeds1, speeds2)
    return float(corr)


def compute_speed_profile_rmse(
    positions1: np.ndarray,
    velocities1: np.ndarray,
    positions2: np.ndarray,
    velocities2: np.ndarray,
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    num_bins: int = 20,
) -> float:
    """
    Compute RMSE of speed vs. progress-to-goal profiles

    Similar to VGA paper Figure 2: v/v_desired vs. position along path

    Args:
        positions1, velocities1: Model trajectory
        positions2, velocities2: Human trajectory
        start_pos: Starting position
        goal_pos: Goal position
        num_bins: Number of spatial bins along path

    Returns:
        RMSE of speed profiles
    """

    def compute_progress_speed_profile(positions, velocities):
        # Distance to goal at each point
        distances_to_goal = np.linalg.norm(positions - goal_pos, axis=1)
        total_distance = np.linalg.norm(start_pos - goal_pos)

        # Progress: 0 at start, 1 at goal
        progress = 1 - (distances_to_goal / total_distance)
        progress = np.clip(progress, 0, 1)

        # Speed at each point
        speeds = np.linalg.norm(velocities, axis=1)

        # Bin by progress
        bins = np.linspace(0, 1, num_bins + 1)
        binned_speeds = []

        for i in range(num_bins):
            mask = (progress >= bins[i]) & (progress < bins[i + 1])
            if np.sum(mask) > 0:
                binned_speeds.append(np.mean(speeds[mask]))
            else:
                binned_speeds.append(np.nan)

        return np.array(binned_speeds)

    profile1 = compute_progress_speed_profile(positions1, velocities1)
    profile2 = compute_progress_speed_profile(positions2, velocities2)

    # Remove NaN bins
    valid_mask = ~(np.isnan(profile1) | np.isnan(profile2))
    if np.sum(valid_mask) < 2:
        return float("inf")

    rmse = np.sqrt(np.mean((profile1[valid_mask] - profile2[valid_mask]) ** 2))
    return float(rmse)


def compute_human_likeness_score(
    hausdorff: float,
    frechet: float,
    velocity_corr: float,
    path_overlap: float,
    speed_rmse: float,
    max_hausdorff: float = 2.0,
    max_frechet: float = 2.0,
    max_speed_rmse: float = 0.5,
) -> float:
    """
    Compute composite human-likeness score (0-1, higher is more human-like)

    Args:
        hausdorff: Hausdorff distance
        frechet: Fréchet distance
        velocity_corr: Velocity correlation (-1 to 1)
        path_overlap: Path overlap (0-1)
        speed_rmse: Speed profile RMSE
        max_*: Normalization constants

    Returns:
        Human-likeness score 0-1
    """
    # Normalize distances (lower is better)
    hausdorff_score = 1 - np.clip(hausdorff / max_hausdorff, 0, 1)
    frechet_score = 1 - np.clip(frechet / max_frechet, 0, 1)
    speed_score = 1 - np.clip(speed_rmse / max_speed_rmse, 0, 1)

    # Velocity correlation (higher is better, -1 to 1 → 0 to 1)
    velocity_score = (velocity_corr + 1) / 2

    # Path overlap already 0-1 (higher is better)
    overlap_score = path_overlap

    # Weighted average
    human_likeness = (
        0.25 * hausdorff_score
        + 0.25 * frechet_score
        + 0.20 * velocity_score
        + 0.15 * overlap_score
        + 0.15 * speed_score
    )

    return float(np.clip(human_likeness, 0.0, 1.0))


def compare_trajectories(
    model_positions: np.ndarray,
    model_velocities: np.ndarray,
    model_timestamps: np.ndarray,
    human_positions: np.ndarray,
    human_velocities: np.ndarray,
    human_timestamps: np.ndarray,
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    resample_points: int = 100,
) -> ComparisonMetrics:
    """
    Compute all comparison metrics between model and human trajectories

    Args:
        model_positions: (N, 2) model trajectory
        model_velocities: (N, 2) model velocities
        model_timestamps: (N,) model timestamps
        human_positions: (M, 2) human trajectory
        human_velocities: (M, 2) human velocities
        human_timestamps: (M,) human timestamps
        start_pos: Starting position
        goal_pos: Goal position
        resample_points: Number of points for resampling

    Returns:
        ComparisonMetrics object
    """
    # Resample to same length for fair comparison
    model_pos_resampled, model_t_resampled = resample_trajectory(
        model_positions, model_timestamps, resample_points
    )
    human_pos_resampled, human_t_resampled = resample_trajectory(
        human_positions, human_timestamps, resample_points
    )

    # Trajectory similarity
    hausdorff_dist = compute_hausdorff_distance(
        model_pos_resampled, human_pos_resampled
    )
    frechet_dist = compute_frechet_distance(model_pos_resampled, human_pos_resampled)
    dtw_dist = compute_dtw_distance(model_pos_resampled, human_pos_resampled)

    # Path similarity
    path_overlap_val = compute_path_overlap(
        model_pos_resampled, human_pos_resampled, corridor_width=0.5
    )
    avg_dist = compute_avg_point_distance(model_pos_resampled, human_pos_resampled)

    # Velocity similarity
    velocity_corr = compute_velocity_correlation(model_velocities, human_velocities)

    # Speed distributions
    model_speeds = np.linalg.norm(model_velocities, axis=1)
    human_speeds = np.linalg.norm(human_velocities, axis=1)
    velocity_dist = wasserstein_distance(model_speeds, human_speeds)

    # Speed profile RMSE
    speed_rmse = compute_speed_profile_rmse(
        model_positions,
        model_velocities,
        human_positions,
        human_velocities,
        start_pos,
        goal_pos,
    )

    # Spatial distribution (lateral positions)
    to_goal = goal_pos - start_pos
    to_goal_norm = to_goal / (np.linalg.norm(to_goal) + 1e-6)
    perpendicular = np.array([-to_goal_norm[1], to_goal_norm[0]])

    model_lateral = np.dot(model_positions - start_pos, perpendicular)
    human_lateral = np.dot(human_positions - start_pos, perpendicular)
    lateral_dist = wasserstein_distance(model_lateral, human_lateral)

    # Position variance ratio
    model_var = np.var(model_positions, axis=0).sum()
    human_var = np.var(human_positions, axis=0).sum()
    var_ratio = model_var / (human_var + 1e-6)

    # Temporal similarity
    model_travel_time = model_timestamps[-1] - model_timestamps[0]
    human_travel_time = human_timestamps[-1] - human_timestamps[0]
    travel_time_ratio = model_travel_time / (human_travel_time + 1e-6)

    # Progress correlation
    model_progress = 1 - np.linalg.norm(
        model_pos_resampled - goal_pos, axis=1
    ) / np.linalg.norm(start_pos - goal_pos)
    human_progress = 1 - np.linalg.norm(
        human_pos_resampled - goal_pos, axis=1
    ) / np.linalg.norm(start_pos - goal_pos)

    if len(model_progress) > 1 and len(human_progress) > 1:
        progress_corr, _ = pearsonr(model_progress, human_progress)
    else:
        progress_corr = 0.0

    # Human-likeness score
    human_likeness = compute_human_likeness_score(
        hausdorff=hausdorff_dist,
        frechet=frechet_dist,
        velocity_corr=velocity_corr,
        path_overlap=path_overlap_val,
        speed_rmse=speed_rmse,
    )

    return ComparisonMetrics(
        hausdorff_distance=hausdorff_dist,
        frechet_distance=frechet_dist,
        dtw_distance=dtw_dist,
        path_overlap=path_overlap_val,
        avg_point_distance=avg_dist,
        velocity_correlation=velocity_corr,
        velocity_distribution_distance=velocity_dist,
        speed_profile_rmse=speed_rmse,
        lateral_distribution_similarity=lateral_dist,
        position_variance_ratio=var_ratio,
        travel_time_ratio=travel_time_ratio,
        progress_correlation=progress_corr,
        human_likeness_score=human_likeness,
    )


def comparison_metrics_to_dict(metrics: ComparisonMetrics) -> Dict:
    """Convert ComparisonMetrics to dictionary"""
    return {
        "hausdorff_distance": metrics.hausdorff_distance,
        "frechet_distance": metrics.frechet_distance,
        "dtw_distance": metrics.dtw_distance,
        "path_overlap": metrics.path_overlap,
        "avg_point_distance": metrics.avg_point_distance,
        "velocity_correlation": metrics.velocity_correlation,
        "velocity_distribution_distance": metrics.velocity_distribution_distance,
        "speed_profile_rmse": metrics.speed_profile_rmse,
        "lateral_distribution_similarity": metrics.lateral_distribution_similarity,
        "position_variance_ratio": metrics.position_variance_ratio,
        "travel_time_ratio": metrics.travel_time_ratio,
        "progress_correlation": metrics.progress_correlation,
        "human_likeness_score": metrics.human_likeness_score,
    }


# Example usage
if __name__ == "__main__":
    # Synthetic trajectories for testing
    t = np.linspace(0, 10, 100)

    # Human trajectory (smooth)
    human_x = t
    human_y = 0.3 * np.sin(2 * np.pi * t / 10)
    human_pos = np.column_stack([human_x, human_y])
    human_vel = np.gradient(human_pos, axis=0)

    # Model trajectory (similar but with noise)
    model_x = t
    model_y = 0.3 * np.sin(2 * np.pi * t / 10) + 0.1 * np.random.randn(len(t))
    model_pos = np.column_stack([model_x, model_y])
    model_vel = np.gradient(model_pos, axis=0)

    start_pos = np.array([0.0, 0.0])
    goal_pos = np.array([10.0, 0.0])

    metrics = compare_trajectories(
        model_positions=model_pos,
        model_velocities=model_vel,
        model_timestamps=t,
        human_positions=human_pos,
        human_velocities=human_vel,
        human_timestamps=t,
        start_pos=start_pos,
        goal_pos=goal_pos,
    )

    print("\n=== Comparison Metrics Test ===\n")
    print(f"Hausdorff Distance: {metrics.hausdorff_distance:.3f}m")
    print(f"Fréchet Distance: {metrics.frechet_distance:.3f}m")
    print(f"Path Overlap: {metrics.path_overlap:.2%}")
    print(f"Velocity Correlation: {metrics.velocity_correlation:.3f}")
    print(f"Human-Likeness Score: {metrics.human_likeness_score:.2f}")
