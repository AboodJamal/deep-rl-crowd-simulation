"""
Experiment Runner for DRL vs VGA+UPL Comparison
================================================

Orchestrates comprehensive validation experiments comparing:
1. DRL (Deterministic mode)
2. DRL (Stochastic mode)
3. VGA+UPL (Deterministic mode)
4. VGA+UPL (Stochastic mode)

Runs experiments across multiple scenarios (SOSP, MOSP, Head-On, etc.)
and saves results for analysis.

Author: [Your Name]
Date: December 2025
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import time
from datetime import datetime
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data_loading'))

from models.vga_upl_planner import VGAUPLPlanner
from models.drl_policy_interface import DRLPolicyInterface
from models.model_base import SimulationResult


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run"""
    experiment_name: str
    output_dir: str
    
    # Scenarios to test
    scenarios: List[str]  # e.g., ['sosp', 'mosp', 'head_on']
    
    # Number of trials
    num_deterministic_trials: int = 1  # Usually 1 since deterministic
    num_stochastic_trials: int = 100   # Multiple for statistics
    
    # Simulation parameters
    max_steps: int = 1000
    goal_tolerance: float = 0.3
    
    # Model paths
    drl_checkpoint_path: Optional[str] = None
    
    # Random seed
    random_seed: int = 42


@dataclass
class ScenarioDefinition:
    """Definition of a test scenario"""
    name: str
    start_pos: np.ndarray
    goal_pos: np.ndarray
    obstacles: List[Dict[str, Any]]
    description: str
    scenario_type: str  # 'sosp', 'mosp', 'head_on', etc.


class ExperimentRunner:
    """
    Main experiment orchestrator.
    
    Runs DRL and VGA+UPL models across multiple scenarios and modes,
    collects results, and saves them for analysis.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.vga_det = VGAUPLPlanner(use_probabilistic=False)
        self.vga_stoch = VGAUPLPlanner(use_probabilistic=True)
        
        # DRL model (will be loaded if checkpoint provided)
        self.drl_model = None
        if config.drl_checkpoint_path:
            self.drl_model = self._load_drl_model(config.drl_checkpoint_path)
        
        # Results storage
        self.results = {
            'config': asdict(config),
            'timestamp': datetime.now().isoformat(),
            'scenarios': {},
            'summary': {}
        }
        
        # Set random seed
        np.random.seed(config.random_seed)
    
    def _load_drl_model(self, checkpoint_path: str) -> DRLPolicyInterface:
        """
        Load DRL model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            
        Returns:
            Loaded DRL model interface
        """
        try:
            print(f"Loading DRL model from: {checkpoint_path}")
            model = DRLPolicyInterface(
                checkpoint_path=checkpoint_path,
                device='cpu'
            )
            print("DRL model loaded successfully")
            return model
        except Exception as e:
            print(f"Warning: Could not load DRL model: {e}")
            print("Continuing with VGA+UPL only...")
            return None
    
    def create_sosp_scenarios(self) -> List[ScenarioDefinition]:
        """
        Create Single Obstacle Single Pedestrian scenarios.
        
        Returns:
            List of SOSP scenario definitions
        """
        scenarios = []
        
        # Basic SOSP - obstacle blocking direct path
        scenarios.append(ScenarioDefinition(
            name="sosp_center",
            start_pos=np.array([0.0, 0.0]),
            goal_pos=np.array([10.0, 0.0]),
            obstacles=[
                {'position': [5.0, 0.0], 'radius': 0.5}
            ],
            description="Single obstacle directly blocking path",
            scenario_type="sosp"
        ))
        
        # SOSP - obstacle slightly off-center
        scenarios.append(ScenarioDefinition(
            name="sosp_offset",
            start_pos=np.array([0.0, 0.0]),
            goal_pos=np.array([10.0, 0.0]),
            obstacles=[
                {'position': [5.0, 0.5], 'radius': 0.5}
            ],
            description="Single obstacle offset from direct path",
            scenario_type="sosp"
        ))
        
        # SOSP - larger obstacle
        scenarios.append(ScenarioDefinition(
            name="sosp_large",
            start_pos=np.array([0.0, 0.0]),
            goal_pos=np.array([10.0, 0.0]),
            obstacles=[
                {'position': [5.0, 0.0], 'radius': 1.0}
            ],
            description="Large obstacle blocking path",
            scenario_type="sosp"
        ))
        
        return scenarios
    
    def create_mosp_scenarios(self) -> List[ScenarioDefinition]:
        """
        Create Multi Obstacle Single Pedestrian scenarios (Cases A, B, C, D).
        
        Returns:
            List of MOSP scenario definitions
        """
        scenarios = []
        
        # Case A: Low density (2 obstacles)
        scenarios.append(ScenarioDefinition(
            name="mosp_case_a",
            start_pos=np.array([0.0, 0.0]),
            goal_pos=np.array([10.0, 0.0]),
            obstacles=[
                {'position': [3.0, 0.5], 'radius': 0.3},
                {'position': [7.0, -0.5], 'radius': 0.3}
            ],
            description="MOSP Case A: Low obstacle density",
            scenario_type="mosp"
        ))
        
        # Case B: Medium density (4 obstacles)
        scenarios.append(ScenarioDefinition(
            name="mosp_case_b",
            start_pos=np.array([0.0, 0.0]),
            goal_pos=np.array([10.0, 0.0]),
            obstacles=[
                {'position': [2.5, 0.7], 'radius': 0.3},
                {'position': [4.0, -0.5], 'radius': 0.3},
                {'position': [6.5, 0.5], 'radius': 0.3},
                {'position': [8.0, -0.7], 'radius': 0.3}
            ],
            description="MOSP Case B: Medium obstacle density",
            scenario_type="mosp"
        ))
        
        # Case C: High density (6 obstacles)
        scenarios.append(ScenarioDefinition(
            name="mosp_case_c",
            start_pos=np.array([0.0, 0.0]),
            goal_pos=np.array([10.0, 0.0]),
            obstacles=[
                {'position': [2.0, 0.8], 'radius': 0.35},
                {'position': [3.5, -0.6], 'radius': 0.35},
                {'position': [5.0, 0.7], 'radius': 0.35},
                {'position': [6.5, -0.5], 'radius': 0.35},
                {'position': [8.0, 0.6], 'radius': 0.35},
                {'position': [9.0, -0.8], 'radius': 0.35}
            ],
            description="MOSP Case C: High obstacle density",
            scenario_type="mosp"
        ))
        
        # Case D: Very high density (8 obstacles)
        scenarios.append(ScenarioDefinition(
            name="mosp_case_d",
            start_pos=np.array([0.0, 0.0]),
            goal_pos=np.array([10.0, 0.0]),
            obstacles=[
                {'position': [1.5, 0.9], 'radius': 0.4},
                {'position': [2.8, -0.7], 'radius': 0.4},
                {'position': [4.0, 0.6], 'radius': 0.4},
                {'position': [5.2, -0.8], 'radius': 0.4},
                {'position': [6.5, 0.7], 'radius': 0.4},
                {'position': [7.5, -0.6], 'radius': 0.4},
                {'position': [8.5, 0.8], 'radius': 0.4},
                {'position': [9.3, -0.9], 'radius': 0.4}
            ],
            description="MOSP Case D: Very high obstacle density",
            scenario_type="mosp"
        ))
        
        return scenarios
    
    def create_head_on_scenarios(self) -> List[ScenarioDefinition]:
        """
        Create head-on encounter scenarios.
        
        Returns:
            List of head-on scenario definitions
        """
        # Note: Head-on scenarios typically involve two agents
        # For now, we'll create scenarios with goal behind agent
        # This can be expanded when multi-agent support is added
        
        scenarios = []
        
        scenarios.append(ScenarioDefinition(
            name="head_on_narrow",
            start_pos=np.array([0.0, 0.0]),
            goal_pos=np.array([10.0, 0.0]),
            obstacles=[
                {'position': [5.0, 1.0], 'radius': 0.3},  # Walls creating narrow passage
                {'position': [5.0, -1.0], 'radius': 0.3}
            ],
            description="Narrow passage scenario",
            scenario_type="head_on"
        ))
        
        return scenarios
    
    def load_scenarios(self) -> List[ScenarioDefinition]:
        """
        Load all requested scenarios.
        
        Returns:
            List of scenario definitions
        """
        all_scenarios = []
        
        for scenario_type in self.config.scenarios:
            if scenario_type == 'sosp':
                all_scenarios.extend(self.create_sosp_scenarios())
            elif scenario_type == 'mosp':
                all_scenarios.extend(self.create_mosp_scenarios())
            elif scenario_type == 'head_on':
                all_scenarios.extend(self.create_head_on_scenarios())
            else:
                print(f"Warning: Unknown scenario type '{scenario_type}'")
        
        return all_scenarios
    
    def run_model_on_scenario(
        self,
        model,
        model_name: str,
        scenario: ScenarioDefinition,
        num_trials: int
    ) -> List[SimulationResult]:
        """
        Run a model multiple times on a scenario.
        
        Args:
            model: The model to run
            model_name: Name of the model for logging
            scenario: Scenario definition
            num_trials: Number of trials to run
            
        Returns:
            List of simulation results
        """
        results = []
        
        for trial in range(num_trials):
            try:
                # Run simulation
                result = model.simulate(
                    start_pos=scenario.start_pos,
                    goal_pos=scenario.goal_pos,
                    obstacles=scenario.obstacles,
                    max_steps=self.config.max_steps
                )
                
                results.append(result)
                
                if (trial + 1) % 10 == 0:
                    print(f"  {model_name}: {trial + 1}/{num_trials} trials complete")
                    
            except Exception as e:
                print(f"  Error in {model_name} trial {trial + 1}: {e}")
                continue
        
        return results
    
    def run_scenario(self, scenario: ScenarioDefinition) -> Dict[str, Any]:
        """
        Run all models on a single scenario.
        
        Args:
            scenario: Scenario definition
            
        Returns:
            Dictionary with results from all models
        """
        print(f"\nRunning scenario: {scenario.name}")
        print(f"  Description: {scenario.description}")
        
        scenario_results = {
            'definition': {
                'name': scenario.name,
                'description': scenario.description,
                'type': scenario.scenario_type,
                'start_pos': scenario.start_pos.tolist(),
                'goal_pos': scenario.goal_pos.tolist(),
                'num_obstacles': len(scenario.obstacles),
                'obstacles': scenario.obstacles  # Store obstacles for video generation
            },
            'models': {}
        }
        
        # VGA+UPL Deterministic
        print("Running VGA+UPL (Deterministic)...")
        vga_det_results = self.run_model_on_scenario(
            self.vga_det,
            "VGA+UPL (Det)",
            scenario,
            self.config.num_deterministic_trials
        )
        scenario_results['models']['vga_upl_deterministic'] = [
            self._serialize_result(r) for r in vga_det_results
        ]
        
        # VGA+UPL Stochastic
        print("Running VGA+UPL (Stochastic)...")
        vga_stoch_results = self.run_model_on_scenario(
            self.vga_stoch,
            "VGA+UPL (Stoch)",
            scenario,
            self.config.num_stochastic_trials
        )
        scenario_results['models']['vga_upl_stochastic'] = [
            self._serialize_result(r) for r in vga_stoch_results
        ]
        
        # DRL models (if available)
        if self.drl_model:
            # DRL Deterministic
            print("Running DRL (Deterministic)...")
            # TODO: Implement deterministic mode in DRL interface
            # drl_det_results = self.run_model_on_scenario(...)
            
            # DRL Stochastic
            print("Running DRL (Stochastic)...")
            # TODO: Implement stochastic mode in DRL interface
            # drl_stoch_results = self.run_model_on_scenario(...)
        
        return scenario_results
    
    def _serialize_result(self, result: SimulationResult) -> Dict[str, Any]:
        """
        Convert SimulationResult to JSON-serializable dictionary.
        
        Args:
            result: Simulation result
            
        Returns:
            Dictionary with serialized data
        """
        return {
            'positions': result.positions.tolist(),
            'velocities': result.velocities.tolist(),
            'timestamps': result.timestamps.tolist(),
            'success': bool(result.success),  # Convert numpy bool to Python bool
            'metadata': result.metadata
        }
    
    def calculate_summary_statistics(self):
        """Calculate summary statistics across all scenarios."""
        summary = {
            'total_scenarios': len(self.results['scenarios']),
            'models': {}
        }
        
        for scenario_name, scenario_data in self.results['scenarios'].items():
            for model_name, trials in scenario_data['models'].items():
                if model_name not in summary['models']:
                    summary['models'][model_name] = {
                        'total_trials': 0,
                        'successful_trials': 0,
                        'success_rate': 0.0,
                        'avg_path_length': 0.0,
                        'avg_time': 0.0
                    }
                
                model_summary = summary['models'][model_name]
                model_summary['total_trials'] += len(trials)
                
                for trial in trials:
                    if trial['success']:
                        model_summary['successful_trials'] += 1
                    
                    # Calculate path length
                    positions = np.array(trial['positions'])
                    if len(positions) > 1:
                        path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
                        model_summary['avg_path_length'] += path_length
                    
                    # Time
                    model_summary['avg_time'] += trial['timestamps'][-1]
        
        # Calculate averages
        for model_name, model_summary in summary['models'].items():
            total = model_summary['total_trials']
            if total > 0:
                model_summary['success_rate'] = model_summary['successful_trials'] / total
                model_summary['avg_path_length'] /= total
                model_summary['avg_time'] /= total
        
        self.results['summary'] = summary
    
    def save_results(self):
        """Save experiment results to JSON file."""
        output_file = self.output_dir / 'experiment_results.json'
        
        print(f"\nSaving results to: {output_file}")
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("Results saved successfully")
    
    def print_summary(self):
        """Print experiment summary to console."""
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        
        summary = self.results['summary']
        
        print(f"\nTotal Scenarios: {summary['total_scenarios']}")
        print(f"\nModel Performance:")
        print("-"*60)
        
        for model_name, stats in summary['models'].items():
            print(f"\n{model_name}:")
            print(f"  Total Trials: {stats['total_trials']}")
            print(f"  Success Rate: {stats['success_rate']:.1%}")
            print(f"  Avg Path Length: {stats['avg_path_length']:.2f}m")
            print(f"  Avg Time: {stats['avg_time']:.2f}s")
        
        print("\n" + "="*60)
    
    def run(self):
        """Run the complete experiment."""
        print("="*60)
        print("STARTING EXPERIMENT")
        print("="*60)
        print(f"Experiment: {self.config.experiment_name}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Random Seed: {self.config.random_seed}")
        
        start_time = time.time()
        
        # Load scenarios
        scenarios = self.load_scenarios()
        print(f"\nLoaded {len(scenarios)} scenarios")
        
        # Run each scenario
        for scenario in scenarios:
            scenario_results = self.run_scenario(scenario)
            self.results['scenarios'][scenario.name] = scenario_results
        
        # Calculate summary statistics
        self.calculate_summary_statistics()
        
        # Save results
        self.save_results()
        
        # Print summary
        elapsed_time = time.time() - start_time
        print(f"\nTotal experiment time: {elapsed_time:.1f} seconds")
        self.print_summary()


def main():
    """Example usage of experiment runner."""
    
    # Create configuration
    config = ExperimentConfig(
        experiment_name="vga_upl_validation",
        output_dir="validation/results/experiment_001",
        scenarios=['sosp', 'mosp'],
        num_deterministic_trials=1,
        num_stochastic_trials=100,
        max_steps=1000,
        random_seed=42
    )
    
    # Create and run experiment
    runner = ExperimentRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
