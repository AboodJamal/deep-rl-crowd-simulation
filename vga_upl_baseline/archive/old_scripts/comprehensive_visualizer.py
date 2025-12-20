"""
Comprehensive Visualization System for VGA Validation
======================================================

Generates extensive comparison plots for:
- VGA+UPL (Deterministic)
- VGA+UPL (Stochastic)
- DRL Model
- Experimental Data

Metrics from VGA Paper:
- Success rates
- Path lengths
- Velocity profiles (V/V_desired vs position)
- Stochastic path distributions
- Collision rates
- Travel times
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


class ComprehensiveVisualizer:
    """
    Creates comprehensive comparison visualizations.
    """
    
    def __init__(self, results_file: str, output_dir: str):
        """
        Initialize visualizer.
        
        Args:
            results_file: Path to comprehensive_results.json
            output_dir: Where to save plots
        """
        self.results_file = Path(results_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        print(f"Loading results from: {results_file}")
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        print(f"Loaded results for {len(self.results['scenarios'])} scenarios")
    
    def plot_success_rates(self):
        """Plot success rate comparison across all models and scenarios"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        scenarios = []
        models_data = {}
        
        # Extract data
        for scenario_type, stats in self.results['summary'].items():
            scenarios.append(scenario_type.upper())
            
            for model_name, model_stats in stats.items():
                if model_name not in models_data:
                    models_data[model_name] = []
                models_data[model_name].append(model_stats['success_rate'] * 100)
        
        # Plot grouped bars
        x = np.arange(len(scenarios))
        width = 0.2
        multiplier = 0
        
        for model_name, success_rates in models_data.items():
            offset = width * multiplier
            ax.bar(x + offset, success_rates, width, label=model_name.replace('_', ' ').title())
            multiplier += 1
        
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate Comparison Across Models and Scenarios')
        ax.set_xticks(x + width * (len(models_data) - 1) / 2)
        ax.set_xticks(x + width)
        ax.set_xticklabels(scenarios)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        # Add horizontal line at 90% (VGA paper target)
        ax.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='VGA Paper Target (90%)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'success_rates_comparison.png', bbox_inches='tight')
        print(f"Saved: success_rates_comparison.png")
        plt.close()
    
    def plot_path_length_distributions(self):
        """Plot path length distributions for each scenario"""
        scenarios = list(self.results['scenarios'].keys())
        
        for scenario_type in scenarios:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            scenario_data = self.results['scenarios'][scenario_type]
            
            for model_name, model_results in scenario_data['models'].items():
                # Extract successful path lengths
                path_lengths = [r['path_length'] for r in model_results if r['success']]
                
                if path_lengths:
                    ax.hist(path_lengths, bins=30, alpha=0.5, 
                           label=f"{model_name.replace('_', ' ').title()} (n={len(path_lengths)})")
            
            ax.set_xlabel('Path Length (m)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Path Length Distribution - {scenario_type.upper()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'path_length_dist_{scenario_type}.png', bbox_inches='tight')
            print(f"Saved: path_length_dist_{scenario_type}.png")
            plt.close()
    
    def plot_stochastic_trajectories(self, scenario_type: str, trial_id: int, max_trajectories: int = 100):
        """
        Plot stochastic trajectory overlay (like VGA Figure 7).
        
        Args:
            scenario_type: Which scenario to plot
            trial_id: Which trial to visualize
            max_trajectories: Maximum number of trajectories to overlay
        """
        if scenario_type not in self.results['scenarios']:
            print(f"Scenario {scenario_type} not found")
            return
        
        scenario_data = self.results['scenarios'][scenario_type]
        
        # Get stochastic results for this trial
        stoch_results = [r for r in scenario_data['models'].get('vga_upl_stochastic', [])
                        if r['trial_id'] == trial_id]
        
        if not stoch_results:
            print(f"No stochastic results found for trial {trial_id}")
            return
        
        # Limit number of trajectories
        stoch_results = stoch_results[:max_trajectories]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot all stochastic trajectories
        for result in stoch_results:
            positions = np.array(result['positions'])
            ax.plot(positions[:, 0], positions[:, 1], 'g-', alpha=0.1, linewidth=0.5)
        
        # Plot mean trajectory (if enough data)
        if len(stoch_results) > 10:
            # Interpolate to common length
            max_len = max(len(r['positions']) for r in stoch_results)
            interpolated = []
            for result in stoch_results:
                pos = np.array(result['positions'])
                if len(pos) > 1:
                    t_old = np.linspace(0, 1, len(pos))
                    t_new = np.linspace(0, 1, max_len)
                    x_new = np.interp(t_new, t_old, pos[:, 0])
                    y_new = np.interp(t_new, t_old, pos[:, 1])
                    interpolated.append(np.column_stack([x_new, y_new]))
            
            if interpolated:
                mean_traj = np.mean(interpolated, axis=0)
                ax.plot(mean_traj[:, 0], mean_traj[:, 1], 'darkgreen', linewidth=2, label='Mean Trajectory')
        
        # Plot start and goal
        start_pos = np.array(stoch_results[0]['positions'][0])
        goal_pos = np.array(stoch_results[0]['positions'][-1])
        
        ax.plot(start_pos[0], start_pos[1], 'bo', markersize=10, label='Start')
        ax.plot(goal_pos[0], goal_pos[1], 'r*', markersize=15, label='Goal')
        
        # Plot obstacles (get from first trial)
        trial_data = None
        for trial in self.get_experimental_trials(scenario_type):
            if trial['trial_id'] == trial_id:
                trial_data = trial
                break
        
        if trial_data and 'obstacles' in trial_data:
            for obs in trial_data['obstacles']:
                circle = plt.Circle(obs['position'], obs['radius'], 
                                   color='red', alpha=0.5, label='Obstacle')
                ax.add_patch(circle)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'Stochastic Path Distribution - {scenario_type.upper()} Trial {trial_id}\n'
                    f'({len(stoch_results)} trajectories)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'stochastic_paths_{scenario_type}_trial{trial_id}.png', 
                   bbox_inches='tight')
        print(f"Saved: stochastic_paths_{scenario_type}_trial{trial_id}.png")
        plt.close()
    
    def plot_velocity_profiles(self, scenario_type: str, trial_id: int):
        """
        Plot velocity profiles (V/V_desired vs position) like VGA Figure 2.
        
        Args:
            scenario_type: Which scenario
            trial_id: Which trial
        """
        if scenario_type not in self.results['scenarios']:
            return
        
        scenario_data = self.results['scenarios'][scenario_type]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model_name, model_results in scenario_data['models'].items():
            # Get this trial's result
            trial_results = [r for r in model_results if r['trial_id'] == trial_id]
            
            if not trial_results:
                continue
            
            # For stochastic, take first run
            result = trial_results[0]
            
            positions = np.array(result['positions'])
            velocities = np.array(result['velocities'])
            
            if len(positions) < 2:
                continue
            
            # Calculate distance along path
            diffs = np.diff(positions, axis=0)
            distances = np.cumsum(np.linalg.norm(diffs, axis=1))
            distances = np.insert(distances, 0, 0)
            
            # Calculate speeds
            speeds = np.linalg.norm(velocities, axis=1)
            
            # Normalize by desired speed (assume 1.34 m/s)
            v_desired = 1.34
            normalized_speeds = speeds / v_desired
            
            ax.plot(distances, normalized_speeds, label=model_name.replace('_', ' ').title(), 
                   alpha=0.7, linewidth=2)
        
        ax.set_xlabel('Distance Along Path (m)')
        ax.set_ylabel('V / V_desired')
        ax.set_title(f'Velocity Profile - {scenario_type.upper()} Trial {trial_id}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3, label='Desired Speed')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'velocity_profile_{scenario_type}_trial{trial_id}.png', 
                   bbox_inches='tight')
        print(f"Saved: velocity_profile_{scenario_type}_trial{trial_id}.png")
        plt.close()
    
    def plot_summary_statistics(self):
        """Create comprehensive summary statistics plot"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # 1. Success Rates
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_metric_comparison(ax1, 'success_rate', 'Success Rate (%)', multiply_by=100)
        
        # 2. Path Lengths
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_metric_comparison(ax2, 'mean_path_length', 'Mean Path Length (m)')
        
        # 3. Travel Times
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_metric_comparison(ax3, 'mean_travel_time', 'Mean Travel Time (s)')
        
        # 4. Collision Rates
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_metric_comparison(ax4, 'collision_rate', 'Collision Rate (%)', multiply_by=100)
        
        plt.suptitle('Comprehensive Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'summary_statistics.png', bbox_inches='tight')
        print(f"Saved: summary_statistics.png")
        plt.close()
    
    def _plot_metric_comparison(self, ax, metric_name: str, ylabel: str, multiply_by: float = 1.0):
        """Helper to plot a single metric comparison"""
        scenarios = []
        models_data = {}
        
        for scenario_type, stats in self.results['summary'].items():
            scenarios.append(scenario_type.upper())
            
            for model_name, model_stats in stats.items():
                if model_name not in models_data:
                    models_data[model_name] = []
                value = model_stats.get(metric_name, 0) * multiply_by
                models_data[model_name].append(value)
        
        x = np.arange(len(scenarios))
        width = 0.2
        multiplier = 0
        
        for model_name, values in models_data.items():
            offset = width * multiplier
            ax.bar(x + offset, values, width, label=model_name.replace('_', ' ').title())
            multiplier += 1
        
        ax.set_ylabel(ylabel)
        ax.set_xticks(x + width)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    def get_experimental_trials(self, scenario_type: str) -> List[Dict]:
        """Helper to get experimental trial data"""
        # This would need to reload from the original data
        # For now, return empty list
        return []
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("="*60)
        
        # 1. Success rates
        print("\n1. Generating success rate comparison...")
        self.plot_success_rates()
        
        # 2. Path length distributions
        print("\n2. Generating path length distributions...")
        self.plot_path_length_distributions()
        
        # 3. Summary statistics
        print("\n3. Generating summary statistics...")
        self.plot_summary_statistics()
        
        # 4. Stochastic trajectories (for first trial of each scenario)
        print("\n4. Generating stochastic trajectory overlays...")
        for scenario_type in self.results['scenarios'].keys():
            self.plot_stochastic_trajectories(scenario_type, trial_id=1, max_trajectories=100)
        
        # 5. Velocity profiles (for first trial of each scenario)
        print("\n5. Generating velocity profiles...")
        for scenario_type in self.results['scenarios'].keys():
            self.plot_velocity_profiles(scenario_type, trial_id=1)
        
        print("\n" + "="*60)
        print(f"All plots saved to: {self.output_dir}")
        print("="*60)


def main():
    """Main entry point"""
    results_file = "validation/results/vga_comparison_experiment/comprehensive_results.json"
    output_dir = "validation/results/vga_comparison_experiment/plots"
    
    visualizer = ComprehensiveVisualizer(results_file, output_dir)
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()
