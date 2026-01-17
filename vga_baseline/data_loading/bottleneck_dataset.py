"""
Bottleneck Individuals Dataset Loader (Jülich)

Parses experimental data from Jülich bottleneck experiments:
https://ped.fz-juelich.de/da/doku.php?id=bottleneck_individuals

Paper: Boomers et al., 2024
"How Approaching Angle, Bottleneck Width and Walking Speed Affect
the Use of a Bottleneck by Individuals"

Author: [Your Name]
Date: December 2025
"""

import numpy as np
import pandas as pd
import h5py
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Motivation(Enum):
    """Participant motivation level"""

    NORMAL = "normal"
    HURRY = "hurry"


class GoalDirection(Enum):
    """Goal location relative to bottleneck"""

    STRAIGHT = "straight"
    LEFT_30 = "30_left"
    RIGHT_30 = "30_right"


@dataclass
class BottleneckConfig:
    """Configuration of bottleneck geometry"""

    width: float  # meters (0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
    length: float  # meters (0.2, 1.0, 2.0)
    height: float = 1.0  # meters (constant)

    # Bottleneck structure is 4m × 2m × 1m
    structure_length: float = 4.0
    structure_width: float = 2.0


@dataclass
class BottleneckTrajectory:
    """Container for a single bottleneck passage trajectory"""

    participant_id: str
    trial_id: str

    # Experimental conditions
    bottleneck_width: float
    bottleneck_length: float
    approach_angle: float  # degrees (-90 to +90)
    motivation: Motivation
    goal_direction: GoalDirection

    # Trajectory data
    timestamps: np.ndarray  # (N,) seconds
    positions: np.ndarray  # (N, 2) - (x, y) in meters
    velocities: np.ndarray  # (N, 2) - (vx, vy) in m/s

    # Computed from trajectory
    speeds: Optional[np.ndarray] = None  # (N,) - speed magnitude

    # Metadata
    age: Optional[int] = None
    gender: Optional[str] = None
    height: Optional[float] = None  # meters

    def __post_init__(self):
        """Compute derived quantities"""
        if self.speeds is None and self.velocities is not None:
            self.speeds = np.linalg.norm(self.velocities, axis=1)

    def duration(self) -> float:
        """Total passage time"""
        return self.timestamps[-1] - self.timestamps[0]

    def path_length(self) -> float:
        """Total distance traveled"""
        if len(self.positions) > 1:
            diffs = np.diff(self.positions, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            return np.sum(distances)
        return 0.0

    def mean_speed(self) -> float:
        """Average speed during passage"""
        return np.mean(self.speeds) if self.speeds is not None else 0.0

    def max_speed(self) -> float:
        """Maximum speed during passage"""
        return np.max(self.speeds) if self.speeds is not None else 0.0

    def get_bottleneck_entry_exit_times(
        self, bottleneck_x_range: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Find times when participant enters and exits bottleneck

        Args:
            bottleneck_x_range: (x_start, x_end) of bottleneck region

        Returns:
            (entry_time, exit_time) in seconds
        """
        x_positions = self.positions[:, 0]

        # Find first time entering bottleneck
        entry_idx = np.where(x_positions >= bottleneck_x_range[0])[0]
        entry_time = (
            self.timestamps[entry_idx[0]] if len(entry_idx) > 0 else self.timestamps[0]
        )

        # Find first time exiting bottleneck
        exit_idx = np.where(x_positions >= bottleneck_x_range[1])[0]
        exit_time = (
            self.timestamps[exit_idx[0]] if len(exit_idx) > 0 else self.timestamps[-1]
        )

        return entry_time, exit_time


class BottleneckDatasetLoader:
    """
    Loader for Jülich Bottleneck Individuals dataset

    Usage:
        loader = BottleneckDatasetLoader(data_dir="path/to/bottleneck/data")
        all_trajs = loader.load_all()
        filtered = loader.filter_trajectories(all_trajs, width=0.6, angle=0)
    """

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Path to bottleneck dataset directory
        """
        self.data_dir = Path(data_dir)

        # TODO: Update paths based on actual dataset structure
        self.metadata_dir = self.data_dir / "metadata"
        self.trajectories_dir = self.data_dir / "trajectories"
        self.hdf5_dir = self.data_dir / "hdf5"

    def load_metadata(self) -> pd.DataFrame:
        """
        Load experiment metadata (participant info, conditions, etc.)

        Returns:
            DataFrame with experiment metadata
        """
        metadata_file = self.metadata_dir / "experiment_metadata.json"

        if not metadata_file.exists():
            print(f"Warning: Metadata file not found at {metadata_file}")
            return pd.DataFrame()

        # TODO: Parse actual metadata JSON structure
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        return pd.DataFrame(metadata)

    def load_trajectory_hdf5(self, file_path: Path) -> BottleneckTrajectory:
        """
        Load a single trajectory from HDF5 file

        Args:
            file_path: Path to HDF5 trajectory file

        Returns:
            BottleneckTrajectory object
        """
        with h5py.File(file_path, "r") as f:
            # TODO: Update based on actual HDF5 structure
            # Expected groups: /trajectory, /metadata, /conditions

            # Load trajectory data
            timestamps = f["trajectory/time"][:]
            x_positions = f["trajectory/x"][:]
            y_positions = f["trajectory/y"][:]
            positions = np.column_stack([x_positions, y_positions])

            # Compute velocities (numerical derivative)
            velocities = np.zeros_like(positions)
            if len(timestamps) > 1:
                dt = np.diff(timestamps)
                dx = np.diff(positions, axis=0)
                velocities[:-1] = dx / dt[:, np.newaxis]
                velocities[-1] = velocities[-2]  # Copy last velocity

            # Load metadata
            participant_id = f["metadata/participant_id"][()].decode("utf-8")
            trial_id = f["metadata/trial_id"][()].decode("utf-8")

            # Load experimental conditions
            width = f["conditions/bottleneck_width"][()]
            length = f["conditions/bottleneck_length"][()]
            angle = f["conditions/approach_angle"][()]
            motivation = Motivation(f["conditions/motivation"][()].decode("utf-8"))
            goal_dir = GoalDirection(f["conditions/goal_direction"][()].decode("utf-8"))

            # Optional participant info
            age = f["metadata/age"][()] if "metadata/age" in f else None
            gender = (
                f["metadata/gender"][()].decode("utf-8")
                if "metadata/gender" in f
                else None
            )
            height = f["metadata/height"][()] if "metadata/height" in f else None

        return BottleneckTrajectory(
            participant_id=participant_id,
            trial_id=trial_id,
            bottleneck_width=width,
            bottleneck_length=length,
            approach_angle=angle,
            motivation=motivation,
            goal_direction=goal_dir,
            timestamps=timestamps,
            positions=positions,
            velocities=velocities,
            age=age,
            gender=gender,
            height=height,
        )

    def load_trajectory_txt(
        self, file_path: Path, metadata: Dict
    ) -> BottleneckTrajectory:
        """
        Load a single trajectory from TXT file

        Args:
            file_path: Path to TXT trajectory file
            metadata: Dictionary with experimental conditions

        Returns:
            BottleneckTrajectory object
        """
        # TODO: Parse TXT format
        # Expected format: time, x, y (space or comma separated)

        data = np.loadtxt(file_path)
        timestamps = data[:, 0]
        positions = data[:, 1:3]

        # Compute velocities
        velocities = np.zeros_like(positions)
        if len(timestamps) > 1:
            dt = np.diff(timestamps)
            dx = np.diff(positions, axis=0)
            velocities[:-1] = dx / dt[:, np.newaxis]
            velocities[-1] = velocities[-2]

        return BottleneckTrajectory(
            participant_id=metadata.get("participant_id", "unknown"),
            trial_id=metadata.get("trial_id", "unknown"),
            bottleneck_width=metadata["width"],
            bottleneck_length=metadata["length"],
            approach_angle=metadata["angle"],
            motivation=Motivation(metadata.get("motivation", "normal")),
            goal_direction=GoalDirection(metadata.get("goal_direction", "straight")),
            timestamps=timestamps,
            positions=positions,
            velocities=velocities,
        )

    def load_all(self) -> List[BottleneckTrajectory]:
        """
        Load all bottleneck trajectories

        Returns:
            List of BottleneckTrajectory objects
        """
        trajectories = []

        # Try HDF5 files first
        if self.hdf5_dir.exists():
            hdf5_files = list(self.hdf5_dir.glob("*.h5")) + list(
                self.hdf5_dir.glob("*.hdf5")
            )
            for hdf5_file in hdf5_files:
                try:
                    traj = self.load_trajectory_hdf5(hdf5_file)
                    trajectories.append(traj)
                except Exception as e:
                    print(f"Error loading {hdf5_file}: {e}")

        # Try TXT files if HDF5 not available
        elif self.trajectories_dir.exists():
            txt_files = list(self.trajectories_dir.glob("*.txt"))
            # TODO: Load corresponding metadata for each file
            print(f"Found {len(txt_files)} TXT trajectory files")
            print("TODO: Implement TXT loading with metadata mapping")

        print(f"Loaded {len(trajectories)} bottleneck trajectories")
        return trajectories

    def filter_trajectories(
        self,
        trajectories: List[BottleneckTrajectory],
        width: Optional[float] = None,
        length: Optional[float] = None,
        angle: Optional[float] = None,
        motivation: Optional[Motivation] = None,
        goal_direction: Optional[GoalDirection] = None,
    ) -> List[BottleneckTrajectory]:
        """
        Filter trajectories by experimental conditions

        Args:
            trajectories: List of all trajectories
            width: Bottleneck width filter (meters)
            length: Bottleneck length filter (meters)
            angle: Approach angle filter (degrees)
            motivation: Motivation level filter
            goal_direction: Goal direction filter

        Returns:
            Filtered list of trajectories
        """
        filtered = trajectories

        if width is not None:
            filtered = [
                t for t in filtered if np.isclose(t.bottleneck_width, width, atol=0.01)
            ]

        if length is not None:
            filtered = [
                t
                for t in filtered
                if np.isclose(t.bottleneck_length, length, atol=0.01)
            ]

        if angle is not None:
            filtered = [
                t for t in filtered if np.isclose(t.approach_angle, angle, atol=1.0)
            ]

        if motivation is not None:
            filtered = [t for t in filtered if t.motivation == motivation]

        if goal_direction is not None:
            filtered = [t for t in filtered if t.goal_direction == goal_direction]

        return filtered


def get_bottleneck_statistics(trajectories: List[BottleneckTrajectory]) -> pd.DataFrame:
    """
    Compute summary statistics for bottleneck trajectories

    Args:
        trajectories: List of BottleneckTrajectory objects

    Returns:
        DataFrame with summary statistics
    """
    stats = []

    for traj in trajectories:
        stats.append(
            {
                "width": traj.bottleneck_width,
                "length": traj.bottleneck_length,
                "angle": traj.approach_angle,
                "motivation": traj.motivation.value,
                "duration": traj.duration(),
                "path_length": traj.path_length(),
                "mean_speed": traj.mean_speed(),
                "max_speed": traj.max_speed(),
            }
        )

    return pd.DataFrame(stats)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python bottleneck_dataset.py <path_to_bottleneck_data>")
        sys.exit(1)

    data_dir = sys.argv[1]
    loader = BottleneckDatasetLoader(data_dir)

    # Load all trajectories
    print("\n=== Bottleneck Dataset Loading ===\n")
    all_trajs = loader.load_all()

    if all_trajs:
        # Display statistics
        stats = get_bottleneck_statistics(all_trajs)
        print("\nOverall Statistics:")
        print(stats.describe())

        # Show statistics by width
        print("\n\nStatistics by Bottleneck Width:")
        grouped = stats.groupby("width").agg(
            {
                "duration": ["mean", "std"],
                "mean_speed": ["mean", "std"],
            }
        )
        print(grouped)
    else:
        print("No trajectories loaded. Check data directory structure.")
