"""
Re-run VGA+UPL experiment with FINAL optimized settings.
Saves results to validation/results/vga_final_experiment/
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.data_loading.vga_dataset import VGADatasetLoader
from validation.models.vga_upl_planner import VGAUPLPlanner
from validation.video_generator import (
    create_single_trajectory_video,
    create_stochastic_overlay_video,
)
from validation.plots.trajectory_plots import create_trajectory_comparison_plot


def run_vga_final_experiment():
    """Run comprehensive VGA+UPL experiment with final settings."""

    # Setup paths
    base_path = Path(__file__).parent
    results_dir = base_path / "results" / "vga_final_experiment"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (results_dir / "videos").mkdir(exist_ok=True)
    (results_dir / "plots").mkdir(exist_ok=True)

    print("=" * 80)
    print("VGA+UPL FINAL EXPERIMENT")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print()

    # Load dataset
    dataset_path = r"D:\Abdullah Jamal\downloads\VGA-exp\Pedestrian-Experimental-Data"
    loader = VGADatasetLoader(dataset_path, obstacle_radius=0.25)
    data = loader.load_all()

    # Create models with FINAL OPTIMIZED SETTINGS
    models = {
        "VGA_UPL_Det": VGAUPLPlanner(
            use_probabilistic=False,
            # VGA params - optimized after fixing clustering
            personal_distance=0.5,  # Optimal for balance
            clustering_threshold=1.0,  # Standard clustering
            detection_cone_angle=60.0,  # Standard cone
        ),
        "VGA_UPL_Stoch": VGAUPLPlanner(
            use_probabilistic=True,
            # Same VGA params
            personal_distance=0.5,
            clustering_threshold=1.0,
            detection_cone_angle=60.0,
        ),
    }

    print("Models:")
    for name in models:
        print(f"  - {name}")
    print()

    # Scenarios to test
    scenarios = {
        "sosp": ("SOSP", data["sosp"]),
        "mosp_a": ("MOSP_A", data["mosp_a"]),
        "mosp_b": ("MOSP_B", data["mosp_b"]),
        "mosp_c": ("MOSP_C", data["mosp_c"]),
        "mosp_d": ("MOSP_D", data["mosp_d"]),
    }

    # Results storage
    all_results = {}
    summary_stats = {}

    # Run experiments
    for scenario_key, (scenario_name, trials) in scenarios.items():
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario_name} ({len(trials)} trials)")
        print(f"{'='*80}")

        scenario_results = {}

        for model_name, model in models.items():
            print(f"\n  Model: {model_name}")
            print(f"  {'-'*76}")

            model_results = []
            successes = 0

            for i, trial in enumerate(trials, 1):
                trial_id = trial.trial_id

                # Run deterministic
                result = model.simulate(
                    start_pos=np.array(trial.initial_pos),
                    goal_pos=np.array(trial.final_pos),
                    obstacles=trial.obstacles,
                    max_steps=1000,
                )

                # For stochastic, run 100 times
                if model.use_probabilistic:
                    stochastic_runs = []
                    for run in range(100):
                        stoch_result = model.simulate(
                            start_pos=np.array(trial.initial_pos),
                            goal_pos=np.array(trial.final_pos),
                            obstacles=trial.obstacles,
                            max_steps=1000,
                        )
                        stochastic_runs.append(
                            {
                                "positions": stoch_result.positions.tolist(),
                                "velocities": stoch_result.velocities.tolist(),
                                "success": stoch_result.success,
                                "path_length": stoch_result.metadata.get(
                                    "path_length", 0
                                ),
                                "travel_time": stoch_result.metadata.get(
                                    "travel_time", 0
                                ),
                            }
                        )
                else:
                    stochastic_runs = None

                # Store result
                trial_result = {
                    "trial_id": trial_id,
                    "scenario": scenario_name,
                    "start_pos": trial.initial_pos,
                    "goal_pos": trial.final_pos,
                    "obstacles": trial.obstacles,
                    "deterministic": {
                        "positions": result.positions.tolist(),
                        "velocities": result.velocities.tolist(),
                        "success": result.success,
                        "path_length": result.metadata.get("path_length", 0),
                        "travel_time": result.metadata.get("travel_time", 0),
                        "final_distance_to_goal": result.metadata.get(
                            "final_distance_to_goal", 0
                        ),
                    },
                    "stochastic_runs": stochastic_runs,
                }

                model_results.append(trial_result)

                if result.success:
                    successes += 1

                # Progress
                if i % 10 == 0 or i == len(trials):
                    success_rate = (successes / i) * 100
                    print(
                        f"    Progress: {i}/{len(trials)} trials, Success rate: {success_rate:.1f}%"
                    )

            # Store model results for this scenario
            scenario_results[model_name] = model_results

            # Summary stats
            final_success_rate = (successes / len(trials)) * 100
            print(f"  Final Success Rate: {final_success_rate:.1f}%")

            if model_name not in summary_stats:
                summary_stats[model_name] = {}
            summary_stats[model_name][scenario_name] = {
                "total_trials": len(trials),
                "successes": successes,
                "success_rate": final_success_rate,
            }

        all_results[scenario_name] = scenario_results

    # Save comprehensive results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")

    results_file = results_dir / "comprehensive_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {results_file}")

    # Save summary statistics
    summary_file = results_dir / "summary_statistics.json"
    with open(summary_file, "w") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"Saved: {summary_file}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print()
    print(f"{'Scenario':<12} | {'VGA+UPL (Det)':<20} | {'VGA+UPL (Stoch)':<20}")
    print("-" * 80)
    for scenario_name in ["SOSP", "MOSP_A", "MOSP_B", "MOSP_C", "MOSP_D"]:
        det_stats = summary_stats["VGA_UPL_Det"][scenario_name]
        stoch_stats = summary_stats["VGA_UPL_Stoch"][scenario_name]

        det_str = f"{det_stats['successes']}/{det_stats['total_trials']} ({det_stats['success_rate']:.1f}%)"
        stoch_str = f"{stoch_stats['successes']}/{stoch_stats['total_trials']} ({stoch_stats['success_rate']:.1f}%)"

        print(f"{scenario_name:<12} | {det_str:<20} | {stoch_str:<20}")

    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}")

    # Generate sample videos and plots for each scenario (first 3 trials)
    for scenario_key, (scenario_name, trials) in scenarios.items():
        print(f"\n{scenario_name}:")

        for trial_idx in range(min(3, len(trials))):
            trial = trials[trial_idx]
            trial_id = trial.trial_id

            print(f"  Trial {trial_id}...")

            # Get results for this trial
            det_result = all_results[scenario_name]["VGA_UPL_Det"][trial_idx]
            stoch_result = all_results[scenario_name]["VGA_UPL_Stoch"][trial_idx]

            # Create deterministic video
            video_path = (
                results_dir / "videos" / f"{scenario_key}_trial{trial_id}_vga_det.mp4"
            )
            create_single_trajectory_video(
                positions=np.array(det_result["deterministic"]["positions"]),
                obstacles=trial.obstacles,
                start_pos=trial.initial_pos,
                goal_pos=trial.final_pos,
                output_path=str(video_path),
                title=f"{scenario_name} Trial {trial_id} - VGA+UPL (Det)",
            )

            # Create stochastic overlay video
            video_path_stoch = (
                results_dir / "videos" / f"{scenario_key}_trial{trial_id}_vga_stoch.mp4"
            )
            stoch_positions = [
                np.array(run["positions"]) for run in stoch_result["stochastic_runs"]
            ]
            create_stochastic_overlay_video(
                all_trajectories=stoch_positions,
                obstacles=trial.obstacles,
                start_pos=trial.initial_pos,
                goal_pos=trial.final_pos,
                output_path=str(video_path_stoch),
                title=f"{scenario_name} Trial {trial_id} - VGA+UPL (Stochastic)",
            )

            # Create comparison plot
            plot_path = (
                results_dir / "plots" / f"{scenario_key}_trial{trial_id}_comparison.png"
            )
            create_trajectory_comparison_plot(
                trajectories={
                    "VGA+UPL (Det)": np.array(det_result["deterministic"]["positions"]),
                    "VGA+UPL (Stoch - sample)": stoch_positions[
                        0
                    ],  # Show first stochastic run
                },
                obstacles=trial.obstacles,
                start_pos=trial.initial_pos,
                goal_pos=trial.final_pos,
                output_path=str(plot_path),
                title=f"{scenario_name} Trial {trial_id}",
            )

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {results_dir}")
    print(f"  - comprehensive_results.json: Full trajectory data")
    print(f"  - summary_statistics.json: Success rates per scenario")
    print(f"  - videos/: Sample trajectory videos")
    print(f"  - plots/: Trajectory comparison plots")


if __name__ == "__main__":
    run_vga_final_experiment()
