"""
VGA Experimental Dataset Loader
================================

Loads and parses experimental data from the VGA paper:
- SOSP (Single Obstacle Single Pedestrian)
- MOSP (Multiple Obstacles Single Pedestrian) - Cases A, B, C, D
- Head-On (two pedestrians)
- Parallel Pedestrian

Data format from GitHub: kanika201293/Pedestrian-Experimental-Data
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ExperimentalTrial:
    """Single experimental trial data"""
    
    scenario_type: str  # 'sosp', 'mosp_a', 'mosp_b', etc.
    trial_id: int
    
    # Pedestrian data
    initial_pos: np.ndarray  # [x, y]
    final_pos: np.ndarray    # [x, y]
    desired_speed: float     # m/s
    
    # Obstacle data
    obstacles: List[Dict[str, any]]  # List of {'position': [x, y], 'radius': r}
    
    # Metadata
    max_time: float  # Maximum simulation time for this scenario type
    

class VGADatasetLoader:
    """
    Loader for VGA experimental datasets.
    
    Parses text files containing initial/final positions and obstacle configurations.
    """
    
    def __init__(self, data_root: str, obstacle_radius: float = 0.25):
        """
        Initialize dataset loader.
        
        Args:
            data_root: Root directory containing experimental data files
            obstacle_radius: Radius to assign to obstacles (meters)
        """
        self.data_root = Path(data_root)
        self.obstacle_radius = obstacle_radius
        
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data directory not found: {data_root}")
    
    def load_sosp(self) -> List[ExperimentalTrial]:
        """
        Load SOSP (Single Obstacle Single Pedestrian) dataset.
        
        Returns:
            List of ExperimentalTrial objects
        """
        # File paths
        init_final_file = self.data_root / "SOSP_initialFinalPos_feed.txt"
        obstacle_file = self.data_root / "SOSP_obstPos_feed.txt"
        
        # Load obstacle position (same for all trials)
        obstacle_data = pd.read_csv(
            obstacle_file,
            header=None,
            names=['x', 'y', 'id'],
            skipinitialspace=True
        )
        
        obstacles = [{
            'position': [row['x'], row['y']],
            'radius': self.obstacle_radius
        } for _, row in obstacle_data.iterrows() if not pd.isna(row['x'])]
        
        # Load trial data
        trial_data = pd.read_csv(
            init_final_file,
            header=None,
            names=['init_x', 'init_y', 'final_x', 'final_y', 'desired_speed', 'exp_no'],
            skipinitialspace=True
        )
        
        # Create trial objects
        trials = []
        for _, row in trial_data.iterrows():
            if pd.isna(row['init_x']):
                continue
                
            trial = ExperimentalTrial(
                scenario_type='sosp',
                trial_id=int(row['exp_no']),
                initial_pos=np.array([row['init_x'], row['init_y']]),
                final_pos=np.array([row['final_x'], row['final_y']]),
                desired_speed=row['desired_speed'],
                obstacles=obstacles.copy(),
                max_time=6.833
            )
            trials.append(trial)
        
        return trials
    
    def load_mosp(self, case: str) -> List[ExperimentalTrial]:
        """
        Load MOSP (Multiple Obstacles Single Pedestrian) dataset.
        
        Args:
            case: 'A', 'B', 'C', or 'D'
        
        Returns:
            List of ExperimentalTrial objects
        """
        case = case.upper()
        if case not in ['A', 'B', 'C', 'D']:
            raise ValueError(f"Invalid MOSP case: {case}. Must be A, B, C, or D")
        
        # File paths
        init_final_file = self.data_root / f"MOSP_Case{case}_initialFinalPos_feed.txt"
        obstacle_file = self.data_root / f"MOSP_Case{case}_obstPos_feed.txt"
        
        # Load obstacle positions (same for all trials in this case)
        obstacle_data = pd.read_csv(
            obstacle_file,
            header=None,
            names=['x', 'y', 'id'],
            skipinitialspace=True
        )
        
        obstacles = [{
            'position': [row['x'], row['y']],
            'radius': self.obstacle_radius
        } for _, row in obstacle_data.iterrows() if not pd.isna(row['x'])]
        
        # Load trial data
        trial_data = pd.read_csv(
            init_final_file,
            header=None,
            names=['init_x', 'init_y', 'final_x', 'final_y', 'desired_speed', 'exp_no', 'idx'],
            skipinitialspace=True
        )
        
        # Max times for each case
        max_times = {
            'A': 10.866,
            'B': 11.266,
            'C': 11.466,
            'D': 12.833
        }
        
        # Create trial objects
        trials = []
        for _, row in trial_data.iterrows():
            if pd.isna(row['init_x']):
                continue
                
            trial = ExperimentalTrial(
                scenario_type=f'mosp_{case.lower()}',
                trial_id=int(row['exp_no']),
                initial_pos=np.array([row['init_x'], row['init_y']]),
                final_pos=np.array([row['final_x'], row['final_y']]),
                desired_speed=row['desired_speed'],
                obstacles=obstacles.copy(),
                max_time=max_times[case]
            )
            trials.append(trial)
        
        return trials
    
    def load_all_mosp(self) -> Dict[str, List[ExperimentalTrial]]:
        """
        Load all MOSP cases (A, B, C, D).
        
        Returns:
            Dictionary mapping case name to list of trials
        """
        all_mosp = {}
        for case in ['A', 'B', 'C', 'D']:
            try:
                trials = self.load_mosp(case)
                all_mosp[f'mosp_{case.lower()}'] = trials
                print(f"Loaded MOSP Case {case}: {len(trials)} trials")
            except Exception as e:
                print(f"Warning: Could not load MOSP Case {case}: {e}")
        
        return all_mosp
    
    def load_all(self) -> Dict[str, List[ExperimentalTrial]]:
        """
        Load all available datasets.
        
        Returns:
            Dictionary mapping scenario type to list of trials
        """
        all_data = {}
        
        # Load SOSP
        try:
            sosp_trials = self.load_sosp()
            all_data['sosp'] = sosp_trials
            print(f"Loaded SOSP: {len(sosp_trials)} trials")
        except Exception as e:
            print(f"Warning: Could not load SOSP: {e}")
        
        # Load all MOSP cases
        mosp_data = self.load_all_mosp()
        all_data.update(mosp_data)
        
        return all_data
    
    def get_summary(self, data: Dict[str, List[ExperimentalTrial]]) -> pd.DataFrame:
        """
        Get summary statistics for loaded data.
        
        Args:
            data: Dictionary of scenario -> trials
        
        Returns:
            DataFrame with summary statistics
        """
        summary = []
        
        for scenario_type, trials in data.items():
            if not trials:
                continue
            
            summary.append({
                'Scenario': scenario_type.upper(),
                'Num Trials': len(trials),
                'Num Obstacles': len(trials[0].obstacles),
                'Max Time (s)': trials[0].max_time,
                'Avg Desired Speed (m/s)': np.mean([t.desired_speed for t in trials]),
                'Avg Distance (m)': np.mean([
                    np.linalg.norm(t.final_pos - t.initial_pos) for t in trials
                ])
            })
        
        return pd.DataFrame(summary)


def test_loader():
    """Test the dataset loader"""
    print("Testing VGA Dataset Loader...")
    print("=" * 60)
    
    # Initialize loader - use project's data folder
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_root = os.path.join(project_root, "data", "VGA-Experimental-Data")
    loader = VGADatasetLoader(data_root, obstacle_radius=0.25)
    
    # Load all data
    all_data = loader.load_all()
    
    # Print summary
    print("\nDataset Summary:")
    print("=" * 60)
    summary = loader.get_summary(all_data)
    print(summary.to_string(index=False))
    
    # Show example trial
    print("\n\nExample Trial (SOSP #1):")
    print("=" * 60)
    if 'sosp' in all_data and all_data['sosp']:
        trial = all_data['sosp'][0]
        print(f"Scenario: {trial.scenario_type}")
        print(f"Trial ID: {trial.trial_id}")
        print(f"Initial Position: {trial.initial_pos}")
        print(f"Final Position: {trial.final_pos}")
        print(f"Desired Speed: {trial.desired_speed:.2f} m/s")
        print(f"Number of Obstacles: {len(trial.obstacles)}")
        for i, obs in enumerate(trial.obstacles):
            print(f"  Obstacle {i}: pos={obs['position']}, radius={obs['radius']}")
        print(f"Max Time: {trial.max_time} s")
    
    print("\n" + "=" * 60)
    print("Dataset loader test complete!")
    
    return all_data


if __name__ == "__main__":
    test_loader()
