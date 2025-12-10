"""
VGA Dataset Loader

Parses experimental pedestrian trajectory data from the VGA GitHub repository:
https://github.com/kanika201293/Pedestrian-Experimental-Data

Supports scenarios:
- Head-On encounters
- Single Obstacle Single Pedestrian (SOSP)
- Multiple Obstacles Single Pedestrian (MOSP)
- Parallel Pedestrian

Author: [Your Name]
Date: December 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class VGATrajectory:
    """Container for a single pedestrian trajectory from VGA dataset"""
    pedestrian_id: str
    scenario: str  # 'head_on', 'sosp', 'mosp', 'parallel'
    trial_id: int
    case_id: str  # e.g., 'A', 'B', 'C', 'D'
    
    # Initial and final states
    start_pos: np.ndarray  # (x, y)
    goal_pos: np.ndarray   # (x, y)
    
    # Full trajectory (if available)
    positions: Optional[np.ndarray] = None  # (N, 2) array of (x, y) positions
    timestamps: Optional[np.ndarray] = None  # (N,) array of times
    velocities: Optional[np.ndarray] = None  # (N, 2) array of (vx, vy)
    
    # Obstacles (for SOSP/MOSP)
    obstacle_positions: Optional[List[np.ndarray]] = None
    obstacle_radii: Optional[List[float]] = None
    
    def duration(self) -> float:
        """Total time from start to goal"""
        if self.timestamps is not None:
            return self.timestamps[-1] - self.timestamps[0]
        return np.nan
    
    def path_length(self) -> float:
        """Total distance traveled along path"""
        if self.positions is not None and len(self.positions) > 1:
            diffs = np.diff(self.positions, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            return np.sum(distances)
        return np.nan
    
    def mean_speed(self) -> float:
        """Average speed along trajectory"""
        if self.velocities is not None:
            speeds = np.linalg.norm(self.velocities, axis=1)
            return np.mean(speeds)
        return np.nan


class VGADatasetLoader:
    """
    Loader for VGA experimental dataset
    
    Usage:
        loader = VGADatasetLoader(data_dir="path/to/vga/data")
        head_on_trajs = loader.load_head_on()
        sosp_trajs = loader.load_sosp()
    """
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Path to VGA dataset directory
        """
        self.data_dir = Path(data_dir)
        
        # TODO: Update these paths based on actual VGA repo structure
        self.head_on_file = self.data_dir / "Head_On_initialFinalPos_feed.txt"
        self.sosp_dir = self.data_dir / "SOSP"
        self.mosp_dir = self.data_dir / "MOSP"
        self.parallel_dir = self.data_dir / "Parallel_Ped"
        
    def load_head_on(self) -> List[VGATrajectory]:
        """
        Load Head-On scenario trajectories
        
        Returns:
            List of VGATrajectory objects for Head-On encounters
        """
        trajectories = []
        
        if not self.head_on_file.exists():
            print(f"Warning: Head-On data file not found at {self.head_on_file}")
            return trajectories
        
        # TODO: Parse Head_On_initialFinalPos_feed.txt
        # Expected format (example):
        # x_init_A, y_init_A, x_final_A, y_final_A, time_A, 
        # x_init_B, y_init_B, x_final_B, y_final_B, time_B, trial_id
        
        data = np.loadtxt(self.head_on_file, delimiter=',')
        
        for row in data:
            # Pedestrian A
            traj_a = VGATrajectory(
                pedestrian_id=f"A_{int(row[-1])}",
                scenario="head_on",
                trial_id=int(row[-1]),
                case_id="A",
                start_pos=np.array([row[0], row[1]]),
                goal_pos=np.array([row[2], row[3]]),
            )
            trajectories.append(traj_a)
            
            # Pedestrian B
            traj_b = VGATrajectory(
                pedestrian_id=f"B_{int(row[-1])}",
                scenario="head_on",
                trial_id=int(row[-1]),
                case_id="B",
                start_pos=np.array([row[5], row[6]]),
                goal_pos=np.array([row[7], row[8]]),
            )
            trajectories.append(traj_b)
        
        print(f"Loaded {len(trajectories)} Head-On trajectories")
        return trajectories
    
    def load_sosp(self) -> List[VGATrajectory]:
        """
        Load Single Obstacle Single Pedestrian (SOSP) trajectories
        
        Returns:
            List of VGATrajectory objects with obstacle information
        """
        trajectories = []
        
        if not self.sosp_dir.exists():
            print(f"Warning: SOSP directory not found at {self.sosp_dir}")
            return trajectories
        
        # TODO: Parse SOSP files
        # Expected to have:
        # - Initial/final positions
        # - Obstacle position and radius
        # - Full trajectory data (time, x, y)
        
        # Placeholder implementation
        print("TODO: Implement SOSP data loading")
        print(f"SOSP directory: {self.sosp_dir}")
        
        return trajectories
    
    def load_mosp(self) -> List[VGATrajectory]:
        """
        Load Multiple Obstacles Single Pedestrian (MOSP) trajectories
        
        Returns:
            List of VGATrajectory objects with multiple obstacle information
        """
        trajectories = []
        
        if not self.mosp_dir.exists():
            print(f"Warning: MOSP directory not found at {self.mosp_dir}")
            return trajectories
        
        # TODO: Parse MOSP files
        # Expected to have:
        # - Initial/final positions
        # - Multiple obstacle positions and radii
        # - Full trajectory data
        
        # Placeholder implementation
        print("TODO: Implement MOSP data loading")
        print(f"MOSP directory: {self.mosp_dir}")
        
        return trajectories
    
    def load_parallel_ped(self) -> List[VGATrajectory]:
        """
        Load Parallel Pedestrian (overtaking) trajectories
        
        Returns:
            List of VGATrajectory objects for parallel/overtaking scenarios
        """
        trajectories = []
        
        if not self.parallel_dir.exists():
            print(f"Warning: Parallel Ped directory not found at {self.parallel_dir}")
            return trajectories
        
        # TODO: Parse parallel pedestrian files
        print("TODO: Implement Parallel Ped data loading")
        
        return trajectories
    
    def load_all(self) -> Dict[str, List[VGATrajectory]]:
        """
        Load all VGA scenarios
        
        Returns:
            Dictionary mapping scenario name to list of trajectories
        """
        return {
            'head_on': self.load_head_on(),
            'sosp': self.load_sosp(),
            'mosp': self.load_mosp(),
            'parallel': self.load_parallel_ped(),
        }


def get_scenario_statistics(trajectories: List[VGATrajectory]) -> pd.DataFrame:
    """
    Compute summary statistics for a list of trajectories
    
    Args:
        trajectories: List of VGATrajectory objects
        
    Returns:
        DataFrame with summary statistics
    """
    stats = {
        'num_trajectories': len(trajectories),
        'mean_duration': np.mean([t.duration() for t in trajectories if not np.isnan(t.duration())]),
        'mean_path_length': np.mean([t.path_length() for t in trajectories if not np.isnan(t.path_length())]),
        'mean_speed': np.mean([t.mean_speed() for t in trajectories if not np.isnan(t.mean_speed())]),
    }
    
    return pd.DataFrame([stats])


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python vga_dataset.py <path_to_vga_data>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    loader = VGADatasetLoader(data_dir)
    
    # Load and display statistics
    print("\n=== VGA Dataset Statistics ===\n")
    
    all_data = loader.load_all()
    for scenario, trajs in all_data.items():
        print(f"\n{scenario.upper()}:")
        if trajs:
            stats = get_scenario_statistics(trajs)
            print(stats.to_string(index=False))
        else:
            print("  No data loaded")
