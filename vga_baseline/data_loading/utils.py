"""
Common utilities for data loading and processing

Author: [Your Name]
Date: December 2025
"""

import numpy as np
from typing import Tuple, List
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def interpolate_trajectory(
    timestamps: np.ndarray,
    positions: np.ndarray,
    target_timestamps: np.ndarray,
) -> np.ndarray:
    """
    Interpolate trajectory to new time points

    Args:
        timestamps: Original timestamps (N,)
        positions: Original positions (N, 2)
        target_timestamps: Desired timestamps (M,)

    Returns:
        Interpolated positions (M, 2)
    """
    interp_x = interp1d(
        timestamps, positions[:, 0], kind="cubic", fill_value="extrapolate"
    )
    interp_y = interp1d(
        timestamps, positions[:, 1], kind="cubic", fill_value="extrapolate"
    )

    new_x = interp_x(target_timestamps)
    new_y = interp_y(target_timestamps)

    return np.column_stack([new_x, new_y])


def smooth_trajectory(
    positions: np.ndarray,
    window_length: int = 5,
    polyorder: int = 2,
) -> np.ndarray:
    """
    Smooth trajectory using Savitzky-Golay filter

    Args:
        positions: Position array (N, 2)
        window_length: Length of filter window (must be odd)
        polyorder: Order of polynomial fit

    Returns:
        Smoothed positions (N, 2)
    """
    if len(positions) < window_length:
        return positions

    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1

    smoothed = np.zeros_like(positions)
    smoothed[:, 0] = savgol_filter(positions[:, 0], window_length, polyorder)
    smoothed[:, 1] = savgol_filter(positions[:, 1], window_length, polyorder)

    return smoothed


def compute_velocities(
    timestamps: np.ndarray,
    positions: np.ndarray,
    smooth: bool = True,
) -> np.ndarray:
    """
    Compute velocities from positions via numerical differentiation

    Args:
        timestamps: Time points (N,)
        positions: Position array (N, 2)
        smooth: Whether to smooth velocities

    Returns:
        Velocity array (N, 2)
    """
    velocities = np.zeros_like(positions)

    if len(timestamps) > 1:
        dt = np.diff(timestamps)
        dx = np.diff(positions, axis=0)
        velocities[:-1] = dx / dt[:, np.newaxis]
        velocities[-1] = velocities[-2]  # Copy last velocity

    if smooth:
        velocities = smooth_trajectory(velocities, window_length=5, polyorder=2)

    return velocities


def resample_trajectory(
    timestamps: np.ndarray,
    positions: np.ndarray,
    dt: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample trajectory to uniform time step

    Args:
        timestamps: Original timestamps (N,)
        positions: Original positions (N, 2)
        dt: Desired time step

    Returns:
        (new_timestamps, new_positions)
    """
    t_start = timestamps[0]
    t_end = timestamps[-1]
    new_timestamps = np.arange(t_start, t_end, dt)

    new_positions = interpolate_trajectory(timestamps, positions, new_timestamps)

    return new_timestamps, new_positions


def compute_heading_angles(positions: np.ndarray) -> np.ndarray:
    """
    Compute heading angles from trajectory

    Args:
        positions: Position array (N, 2)

    Returns:
        Heading angles in radians (N,)
    """
    headings = np.zeros(len(positions))

    if len(positions) > 1:
        dx = np.diff(positions[:, 0])
        dy = np.diff(positions[:, 1])
        headings[:-1] = np.arctan2(dy, dx)
        headings[-1] = headings[-2]  # Copy last heading

    return headings


def align_trajectories_to_goal(
    positions: np.ndarray,
    goal_pos: np.ndarray,
) -> np.ndarray:
    """
    Rotate and translate trajectory so goal is at origin with heading 0

    Args:
        positions: Position array (N, 2)
        goal_pos: Goal position (2,)

    Returns:
        Aligned positions (N, 2)
    """
    # Translate so goal is at origin
    aligned = positions - goal_pos

    # TODO: Optionally rotate so average heading toward goal is 0
    # This would require computing the dominant direction

    return aligned


def downsample_trajectory(
    positions: np.ndarray,
    timestamps: np.ndarray,
    factor: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample trajectory by taking every Nth point

    Args:
        positions: Position array (N, 2)
        timestamps: Time array (N,)
        factor: Downsampling factor

    Returns:
        (downsampled_positions, downsampled_timestamps)
    """
    return positions[::factor], timestamps[::factor]


def convert_units(
    positions: np.ndarray,
    from_unit: str = "m",
    to_unit: str = "m",
) -> np.ndarray:
    """
    Convert position units (e.g., meters to centimeters)

    Args:
        positions: Position array (N, 2)
        from_unit: Original unit ('m', 'cm', 'mm')
        to_unit: Target unit ('m', 'cm', 'mm')

    Returns:
        Converted positions (N, 2)
    """
    conversions = {
        ("m", "cm"): 100.0,
        ("m", "mm"): 1000.0,
        ("cm", "m"): 0.01,
        ("cm", "mm"): 10.0,
        ("mm", "m"): 0.001,
        ("mm", "cm"): 0.1,
    }

    if from_unit == to_unit:
        return positions.copy()

    factor = conversions.get((from_unit, to_unit), 1.0)
    return positions * factor
