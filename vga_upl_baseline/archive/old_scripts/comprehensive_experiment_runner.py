"""
Comprehensive VGA Validation Experiment Runner
===============================================

Compares 4 models on real experimental scenarios:
1. VGA+UPL (Deterministic)
2. VGA+UPL (Stochastic)
3. DRL Model
4. Experimental Data (ground truth)

Generates extensive comparison metrics and visualizations.
"""

import numpy as np
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

# Add parent directories to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data_loading"))

from data_loading.vga_dataset import VGADatasetLoader, ExperimentalTrial
from models.vga_upl_planner import VGAUPLPlanner
from models.model_base import SimulationResult
from video_generator import TrajectoryVideoGenerator


@dataclass
class ModelResult:
    """Results from running one model on one trial"""

    model_name: str
    scenario_type: str
    trial_id: int

    success: bool
    path_length: float
    travel_time: float
    final_distance_to_goal: float

    # Trajectory data
    positions: List[List[float]]  # [[x, y], ...]
    velocities: List[List[float]]  # [[vx, vy], ...]
    timestamps: List[float]

    # Metrics
    mean_speed: float
    max_speed: float
    collision_occurred: bool


class ComprehensiveExperimentRunner:
    """
    Runs comprehensive validation experiments comparing all models.
    """

    def __init__(self, data_root: str, output_dir: str):
        """
        Initialize experiment runner.

        Args:
            data_root: Path to experimental data
            output_dir: Where to save results
        """
        self.data_root = data_root
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load experimental data
        print("Loading experimental data...")
        self.loader = VGADatasetLoader(data_root, obstacle_radius=0.25)
        self.experimental_data = self.loader.load_all()

        # Print summary
        summary = self.loader.get_summary(self.experimental_data)
        print("\nLoaded Experimental Data:")
        print(summary.to_string(index=False))

        # Initialize models
        print("\nInitializing models...")
        self.vga_det = VGAUPLPlanner(use_probabilistic=False)
        self.vga_stoch = VGAUPLPlanner(use_probabilistic=True)
        # Shared video generator (used for all per-model and comparison videos)
        self.video_generator = TrajectoryVideoGenerator(fps=20)

        # DRL model - Load using simple wrapper
        self.drl_model = None
        try:
            print("Loading DRL model...")
            from models.simple_drl_model import SimpleDRLModel

            # Try multiple possible locations for checkpoints
            base_dirs = [
                Path(os.path.dirname(__file__)).parent / "checkpoints",
                Path(os.path.dirname(__file__)).parent / "models",
                Path(os.path.dirname(__file__)).parent
                / "docs"
                / "archive"
                / "latestRun5-12(reviewed)"
                / "models",
            ]

            checkpoint_path = None
            vecnorm_path = None

            for base_dir in base_dirs:
                # Try stage12 first
                cp = base_dir / "ultimate_stage12_3837248_steps.zip"
                vn = base_dir / "ultimate_stage12_vecnormalize.pkl"
                if not cp.exists():
                    cp = base_dir / "ultimate_generalized_agent_stage12.zip"
                    vn = (
                        base_dir / "ultimate_generalized_agent_stage12_vecnormalize.pkl"
                    )

                if cp.exists():
                    checkpoint_path = str(cp)
                    if vn.exists():
                        vecnorm_path = str(vn)
                    break

            if checkpoint_path and Path(checkpoint_path).exists():
                self.drl_model = SimpleDRLModel(checkpoint_path, vecnorm_path)
                print("[OK] DRL model ready!")
            else:
                print(f"Warning: DRL checkpoint not found")
        except Exception as e:
            print(f"Warning: Could not load DRL model: {e}")
            import traceback

            traceback.print_exc()
            print("Continuing with VGA+UPL only...")

        # Results storage
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "data_root": data_root,
            "scenarios": {},
            "summary": {},
        }

    def generate_single_model_plot(
        self,
        scenario_type: str,
        model_name: str,
        model_key: str,
        trials: List[ExperimentalTrial],
        results: Dict[str, List[Dict]],
        output_path: Path,
    ):
        """Generate static plot for a SINGLE model vs Ground Truth"""
        plt.figure(figsize=(10, 6))

        colors = {
            "Ground Truth": "black",
            "VGA+UPL (Det)": "blue",
            "VGA+UPL (Stoch)": "red",
            "DRL": "green",
        }

        rep_trial = trials[0]

        ax = plt.gca()
        # Plot obstacles
        for obs in rep_trial.obstacles:
            circle = patches.Circle(
                obs["position"], obs["radius"], facecolor="gray", alpha=0.5, zorder=2
            )
            ax.add_patch(circle)

        # Plot Model Trajectories
        if model_key in results["models"]:
            model_results = results["models"][model_key]
            color = colors.get(model_name, "orange")

            for res in model_results:
                if "positions" in res and res["positions"]:
                    path = np.array(res["positions"])
                    if len(path) > 1:
                        alpha = 0.2 if model_key == "vga_upl_stochastic" else 0.6
                        plt.plot(
                            path[:, 0],
                            path[:, 1],
                            color=color,
                            alpha=alpha,
                            linewidth=2,
                            label=(
                                model_name
                                if model_name
                                not in plt.gca().get_legend_handles_labels()[1]
                                else ""
                            ),
                        )

        # Start/Goal
        plt.plot(
            rep_trial.initial_pos[0],
            rep_trial.initial_pos[1],
            "o",
            color="k",
            markersize=10,
            markerfacecolor="white",
            label="Start",
        )
        plt.plot(
            rep_trial.final_pos[0],
            rep_trial.final_pos[1],
            "*",
            color="gold",
            markersize=15,
            label="Goal",
        )

        plt.title(f"Trajectory: {scenario_type.upper()} - {model_name}")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.axis("equal")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Generated plot: {output_path}")

    def generate_scenario_plot(
        self,
        scenario_type: str,
        trials: List[ExperimentalTrial],
        results: Dict[str, List[Dict]],
        output_path: Path,
    ):
        """Generate static comparison plot for a scenario"""
        plt.figure(figsize=(10, 6))

        # Setup colors
        colors = {
            "Ground Truth": "black",
            "VGA+UPL (Det)": "blue",
            "VGA+UPL (Stoch)": "red",
            "DRL": "green",
        }

        # Plot obstacles (from first trial, assuming static for scenario type for visualization purpose,
        # or we plot for a specific trial. The user wants "in each scienario we have to see...".
        # Usually scenarios have similar layouts. I'll plot a representative trial, e.g., the first one.)
        rep_trial = trials[0]

        ax = plt.gca()
        for obs in rep_trial.obstacles:
            circle = patches.Circle(
                obs["position"], obs["radius"], facecolor="gray", alpha=0.5, zorder=2
            )
            ax.add_patch(circle)

        # Plot Trajectories
        # Ground Truth - (Trajectory data not available in dataset keys, only Start/End)
        # We will plot Start/Goal separately below.

        # Model trajectories
        # Identify model keys
        model_map = {
            "vga_upl_deterministic": "VGA+UPL (Det)",
            "vga_upl_stochastic": "VGA+UPL (Stoch)",
            "drl": "DRL",
        }

        for model_key, model_results in results["models"].items():
            label_name = model_map.get(model_key, model_key)
            color = colors.get(label_name, "orange")

            for res in model_results:
                # For stochastic, we might have many runs. Limit alpha.
                if "positions" in res and res["positions"]:
                    path = np.array(res["positions"])
                    if len(path) > 1:
                        alpha = 0.2 if model_key == "vga_upl_stochastic" else 0.6
                        plt.plot(
                            path[:, 0],
                            path[:, 1],
                            color=color,
                            alpha=alpha,
                            linewidth=2,
                            label=(
                                label_name
                                if label_name
                                not in plt.gca().get_legend_handles_labels()[1]
                                else ""
                            ),
                        )

        # Start/Goal
        plt.plot(
            rep_trial.initial_pos[0],
            rep_trial.initial_pos[1],
            "o",
            color="k",
            markersize=10,
            markerfacecolor="white",
            label="Start",
        )
        plt.plot(
            rep_trial.final_pos[0],
            rep_trial.final_pos[1],
            "*",
            color="gold",
            markersize=15,
            label="Goal",
        )

        plt.title(f"Trajectory Comparison: {scenario_type.upper()}")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.axis("equal")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Generated plot: {output_path}")

    def generate_video(
        self,
        scenario_type: str,
        trial: ExperimentalTrial,
        results_map: Dict[str, Any],
        output_path: Path,
    ):
        """
        Generate videos for a specific trial.

        This now creates:
        - A combined comparison video (all models side‑by‑side)
        - A separate video for each model:
          * VGA+UPL (Deterministic)
          * VGA+UPL (Stochastic overlay)
          * DRL (if available)
        """
        # Collect trajectories for this trial from stored results
        model_key_map = {
            "vga_upl_deterministic": "VGA+UPL (Det)",
            "vga_upl_stochastic": "VGA+UPL (Stoch)",
            "drl": "DRL",
        }

        trajectories_for_comparison: Dict[str, np.ndarray] = {}

        for model_key, model_label in model_key_map.items():
            if model_key not in results_map["models"]:
                continue

            # All results for this trial id
            trial_results = [
                r
                for r in results_map["models"][model_key]
                if r["trial_id"] == trial.trial_id
            ]
            if not trial_results:
                continue

            # First run is used for the comparison video
            first_run = trial_results[0]
            if "positions" in first_run and first_run["positions"]:
                path = np.array(first_run["positions"])
                if len(path) > 1:
                    trajectories_for_comparison[model_label] = path

        if not trajectories_for_comparison:
            return

        # Start / goal / obstacles from the experimental trial
        start_pos = np.array(trial.initial_pos)
        goal_pos = np.array(trial.final_pos)
        obstacles = trial.obstacles

        # 1) Combined comparison video (multi‑panel)
        try:
            self.video_generator.create_comparison_video(
                trajectories=trajectories_for_comparison,
                obstacles=obstacles,
                start_pos=start_pos,
                goal_pos=goal_pos,
                output_path=str(output_path),
                title=f"{scenario_type.upper()} - Trial {trial.trial_id} Comparison",
            )
        except Exception as e:
            print(f"Warning: comparison video generation failed for {output_path}: {e}")

        # 2) Per‑model videos (one folder of videos, each model gets its own file)
        video_dir = output_path.parent
        base_stem = output_path.stem  # e.g. "mosp_a_trial_1"

        # VGA+UPL Deterministic (single trajectory)
        if "vga_upl_deterministic" in results_map["models"]:
            det_results = [
                r
                for r in results_map["models"]["vga_upl_deterministic"]
                if r["trial_id"] == trial.trial_id and r.get("positions")
            ]
            if det_results:
                det_positions = np.array(det_results[0]["positions"])
                det_path = video_dir / f"{base_stem}_vga_upl_det.mp4"
                try:
                    self.video_generator.create_single_trajectory_video(
                        positions=det_positions,
                        obstacles=obstacles,
                        start_pos=start_pos,
                        goal_pos=goal_pos,
                        output_path=str(det_path),
                        title=f"{scenario_type.upper()} - Trial {trial.trial_id} VGA+UPL (Det)",
                    )
                except Exception as e:
                    print(
                        f"Warning: could not create VGA+UPL (Det) video for {det_path}: {e}"
                    )

        # VGA+UPL Stochastic (overlay of many runs)
        if "vga_upl_stochastic" in results_map["models"]:
            stoch_results = [
                r
                for r in results_map["models"]["vga_upl_stochastic"]
                if r["trial_id"] == trial.trial_id and r.get("positions")
            ]
            if stoch_results:
                # Limit to a reasonable number to keep videos light
                max_stoch = 20
                traj_list = [
                    np.array(r["positions"])
                    for r in stoch_results[:max_stoch]
                    if r.get("positions")
                ]
                if traj_list:
                    stoch_path = video_dir / f"{base_stem}_vga_upl_stoch_overlay.mp4"
                    try:
                        self.video_generator.create_stochastic_overlay_video(
                            all_trajectories=traj_list,
                            obstacles=obstacles,
                            start_pos=start_pos,
                            goal_pos=goal_pos,
                            output_path=str(stoch_path),
                            title=(
                                f"{scenario_type.upper()} - Trial {trial.trial_id} "
                                f"VGA+UPL (Stoch) Overlay ({len(traj_list)} runs)"
                            ),
                        )
                    except Exception as e:
                        print(
                            f"Warning: could not create VGA+UPL (Stoch) video for {stoch_path}: {e}"
                        )

        # DRL model (if available)
        if "drl" in results_map["models"]:
            drl_results = [
                r
                for r in results_map["models"]["drl"]
                if r["trial_id"] == trial.trial_id and r.get("positions")
            ]
            if drl_results:
                drl_positions = np.array(drl_results[0]["positions"])
                if len(drl_positions) > 1:
                    drl_path = video_dir / f"{base_stem}_drl.mp4"
                    try:
                        self.video_generator.create_single_trajectory_video(
                            positions=drl_positions,
                            obstacles=obstacles,
                            start_pos=start_pos,
                            goal_pos=goal_pos,
                            output_path=str(drl_path),
                            title=f"{scenario_type.upper()} - Trial {trial.trial_id} DRL",
                        )
                    except Exception as e:
                        print(
                            f"Warning: could not create DRL video for {drl_path}: {e}"
                        )

    def run_model_on_trial(self, model, model_name, trial, max_steps=1000):
        # ... (rest of method is same, but I need to make sure I don't delete it)
        # Actually I can just return here, the method is already defined in the class.
        # I only needed to replace generate_video.
        pass

    # ... (Re-implement run_all to include separate plots) ...
    # This replacement is tricky because I need to match the indentation and context.
    # I'll just replace `generate_video` fully, and then a separate call to update `run_all`.

    # Wait, I cannot define run_model_on_trial inside generate_video or mis-align blocks.
    # I will cancel this replacement and split it into two:
    # 1. Update generate_video
    # 2. Update run_all logic

    def run_model_on_trial(
        self, model, model_name: str, trial: ExperimentalTrial, max_steps: int = 1000
    ) -> ModelResult:
        """
        Run a single model on a single trial.

        Args:
            model: The model to run
            model_name: Name for logging
            trial: Experimental trial to simulate
            max_steps: Maximum simulation steps

        Returns:
            ModelResult with trajectory and metrics
        """
        # Run simulation
        result = model.simulate(
            start_pos=trial.initial_pos,
            goal_pos=trial.final_pos,
            obstacles=trial.obstacles,
            max_steps=max_steps,
        )

        # Calculate metrics
        positions = result.positions
        velocities = result.velocities
        timestamps = result.timestamps

        # Path length
        if len(positions) > 1:
            diffs = np.diff(positions, axis=0)
            path_length = np.sum(np.linalg.norm(diffs, axis=1))
        else:
            path_length = 0.0

        # Travel time
        travel_time = timestamps[-1] if len(timestamps) > 0 else 0.0

        # Final distance to goal
        final_dist = (
            np.linalg.norm(positions[-1] - trial.final_pos)
            if len(positions) > 0
            else float("inf")
        )

        # Speed metrics
        speeds = (
            np.linalg.norm(velocities, axis=1)
            if len(velocities) > 0
            else np.array([0.0])
        )
        mean_speed = np.mean(speeds)
        max_speed = np.max(speeds)

        # Check for collisions
        collision = False
        for pos in positions:
            for obs in trial.obstacles:
                obs_pos = np.array(obs["position"])
                obs_radius = obs["radius"]
                dist = np.linalg.norm(pos - obs_pos)
                if dist < obs_radius + 0.2:  # 0.2 = pedestrian radius
                    collision = True
                    break
            if collision:
                break

        return ModelResult(
            model_name=model_name,
            scenario_type=trial.scenario_type,
            trial_id=trial.trial_id,
            success=bool(result.success),  # Convert to Python bool
            path_length=float(path_length),
            travel_time=float(travel_time),
            final_distance_to_goal=float(final_dist),
            positions=positions.tolist(),
            velocities=velocities.tolist(),
            timestamps=timestamps.tolist(),
            mean_speed=float(mean_speed),
            max_speed=float(max_speed),
            collision_occurred=bool(collision),  # Convert to Python bool
        )

    def run_scenario(
        self,
        scenario_type: str,
        trials: List[ExperimentalTrial],
        num_stochastic_runs: int = 100,
    ) -> Dict[str, Any]:
        """
        Run all models on all trials in a scenario.

        Args:
            scenario_type: Name of scenario
            trials: List of experimental trials
            num_stochastic_runs: Number of runs for stochastic model

        Returns:
            Dictionary with all results
        """
        print(f"\n{'='*60}")
        print(f"Running Scenario: {scenario_type.upper()}")
        print(f"Number of trials: {len(trials)}")
        print(f"{'='*60}")

        scenario_results = {
            "scenario_type": scenario_type,
            "num_trials": len(trials),
            "models": {},
        }

        # Run VGA+UPL Deterministic
        print("\nRunning VGA+UPL (Deterministic)...")
        vga_det_results = []
        for i, trial in enumerate(trials):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(trials)}")
            result = self.run_model_on_trial(self.vga_det, "VGA+UPL (Det)", trial)
            vga_det_results.append(asdict(result))
        scenario_results["models"]["vga_upl_deterministic"] = vga_det_results

        # Run VGA+UPL Stochastic (multiple runs per trial)
        print(
            f"\nRunning VGA+UPL (Stochastic) - {num_stochastic_runs} runs per trial..."
        )
        vga_stoch_results = []
        for i, trial in enumerate(trials):
            if (i + 1) % 10 == 0:
                print(f"  Trial {i+1}/{len(trials)}")

            # Run multiple times for stochastic
            for run in range(num_stochastic_runs):
                result = self.run_model_on_trial(
                    self.vga_stoch, "VGA+UPL (Stoch)", trial
                )
                result_dict = asdict(result)
                result_dict["stochastic_run"] = run + 1
                vga_stoch_results.append(result_dict)

        scenario_results["models"]["vga_upl_stochastic"] = vga_stoch_results

        # DRL Model (if available)
        if self.drl_model:
            print("\nRunning DRL Model (Deterministic)...")
            drl_results = []
            for i, trial in enumerate(trials):
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i+1}/{len(trials)}")
                try:
                    result = self.run_model_on_trial(self.drl_model, "DRL", trial)
                    drl_results.append(asdict(result))
                except Exception as e:
                    print(f"  Error in DRL trial {i+1}: {e}")
                    continue
            scenario_results["models"]["drl"] = drl_results

        return scenario_results

    def calculate_summary_statistics(self):
        """Calculate summary statistics across all scenarios"""
        summary = {}

        for scenario_type, scenario_data in self.results["scenarios"].items():
            summary[scenario_type] = {}

            for model_name, model_results in scenario_data["models"].items():
                # Calculate statistics
                successes = [r["success"] for r in model_results]
                path_lengths = [r["path_length"] for r in model_results if r["success"]]
                travel_times = [r["travel_time"] for r in model_results if r["success"]]
                collisions = [r["collision_occurred"] for r in model_results]

                summary[scenario_type][model_name] = {
                    "total_runs": len(model_results),
                    "success_rate": np.mean(successes) if successes else 0.0,
                    "collision_rate": np.mean(collisions) if collisions else 0.0,
                    "mean_path_length": np.mean(path_lengths) if path_lengths else 0.0,
                    "std_path_length": np.std(path_lengths) if path_lengths else 0.0,
                    "mean_travel_time": np.mean(travel_times) if travel_times else 0.0,
                    "std_travel_time": np.std(travel_times) if travel_times else 0.0,
                }

        self.results["summary"] = summary

    def save_results(self):
        """Save results to JSON file"""
        output_file = self.output_dir / "comprehensive_results.json"

        print(f"\nSaving results to: {output_file}")

        # Custom JSON encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                return super().default(obj)

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, cls=NumpyEncoder)

        print("Results saved successfully!")

    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)

        for scenario_type, models in self.results["summary"].items():
            print(f"\n{scenario_type.upper()}:")
            print("-" * 80)

            for model_name, stats in models.items():
                print(f"\n  {model_name}:")
                print(f"    Success Rate: {stats['success_rate']:.1%}")
                print(f"    Collision Rate: {stats['collision_rate']:.1%}")
                print(
                    f"    Mean Path Length: {stats['mean_path_length']:.2f} ± {stats['std_path_length']:.2f} m"
                )
                print(
                    f"    Mean Travel Time: {stats['mean_travel_time']:.2f} ± {stats['std_travel_time']:.2f} s"
                )

        print("\n" + "=" * 80)

    def run_all(self, scenarios_to_run: List[str] = None, num_stochastic: int = 100):
        """
        Run complete validation experiment.

        Args:
            scenarios_to_run: List of scenario names to run (None = all)
            num_stochastic: Number of stochastic runs per trial
        """
        start_time = time.time()

        print("\n" + "=" * 80)
        print("STARTING COMPREHENSIVE VALIDATION EXPERIMENT")
        print("=" * 80)
        print(f"Output Directory: {self.output_dir}")
        print(f"Stochastic Runs per Trial: {num_stochastic}")

        # Determine which scenarios to run
        if scenarios_to_run is None:
            scenarios_to_run = list(self.experimental_data.keys())

        # Run each scenario
        for scenario_type in scenarios_to_run:
            if scenario_type not in self.experimental_data:
                print(f"Warning: Scenario '{scenario_type}' not found in data")
                continue

            trials = self.experimental_data[scenario_type]
            if not trials:
                print(f"Warning: No trials found for scenario '{scenario_type}'")
                continue

            scenario_results = self.run_scenario(scenario_type, trials, num_stochastic)
            self.results["scenarios"][scenario_type] = scenario_results

        # Calculate summary statistics
        print("\nCalculating summary statistics...")
        self.calculate_summary_statistics()

        # Save results
        self.save_results()

        # Generate Visualizations
        print("\nGenerating Visualizations...")
        for scenario_type, scenario_results in self.results["scenarios"].items():
            if scenario_type not in self.experimental_data:
                continue

            trials = self.experimental_data[scenario_type]

            # 1. Static Plots
            # A) Combined Comparison
            plot_path = self.output_dir / f"{scenario_type}_comparison.png"
            self.generate_scenario_plot(
                scenario_type, trials, scenario_results, plot_path
            )

            # B) Individual Model Plots (by scenario AND by model)
            #    - plots/<scenario_type>/<model_name>.png       (scenario‑centric)
            #    - plots_by_model/<model_name>/<scenario>.png  (model‑centric, what the user asked for)
            scenario_plots_dir = self.output_dir / "plots" / scenario_type
            scenario_plots_dir.mkdir(parents=True, exist_ok=True)

            model_plots_root = self.output_dir / "plots_by_model"
            model_plots_root.mkdir(parents=True, exist_ok=True)

            # Map of key -> nice name
            model_map = {
                "vga_upl_deterministic": "VGA+UPL (Det)",
                "vga_upl_stochastic": "VGA+UPL (Stoch)",
                "drl": "DRL",
            }

            for model_key, model_name in model_map.items():
                if model_key in scenario_results["models"]:
                    # Scenario‑centric file
                    scenario_filename = (
                        model_name.replace(" ", "_")
                        .replace("(", "")
                        .replace(")", "")
                        .replace("+", "")
                    ) + ".png"
                    scenario_plot_path = scenario_plots_dir / scenario_filename

                    # Model‑centric directory / file
                    model_folder_name = (
                        model_name.replace(" ", "_")
                        .replace("(", "")
                        .replace(")", "")
                        .replace("+", "")
                    )
                    model_dir = model_plots_root / model_folder_name
                    model_dir.mkdir(parents=True, exist_ok=True)
                    model_plot_path = model_dir / f"{scenario_type}.png"

                    # Generate and save both variants
                    self.generate_single_model_plot(
                        scenario_type,
                        model_name,
                        model_key,
                        trials,
                        scenario_results,
                        scenario_plot_path,
                    )
                    # Save a second copy into the per‑model folder
                    self.generate_single_model_plot(
                        scenario_type,
                        model_name,
                        model_key,
                        trials,
                        scenario_results,
                        model_plot_path,
                    )

            # 2. Videos (First 3 unique trials)
            # Find unique trial IDs
            trial_ids = sorted(list(set(t.trial_id for t in trials)))[
                :3
            ]  # Limit to first 3 for speed

            video_dir = self.output_dir / "videos"
            video_dir.mkdir(exist_ok=True)

            for tid in trial_ids:
                # Find the trial object
                trial = next((t for t in trials if t.trial_id == tid), None)
                if trial:
                    video_path = video_dir / f"{scenario_type}_trial_{tid}.mp4"
                    self.generate_video(
                        scenario_type, trial, scenario_results, video_path
                    )

        # Print summary
        self.print_summary()

        elapsed_time = time.time() - start_time
        print(f"\nTotal experiment time: {elapsed_time/60:.1f} minutes")
        print("\nExperiment complete!")


def main():
    """Main entry point"""
    # Configuration
    data_root = r"D:\Abdullah Jamal\downloads\VGA-exp\Pedestrian-Experimental-Data"
    output_dir = "validation/results/vga_comparison_experiment_with_drl"

    # Create experiment runner
    runner = ComprehensiveExperimentRunner(data_root, output_dir)

    # TEST: Run SOSP only first to verify DRL works
    # Then we'll expand to all scenarios
    runner.run_all(
        scenarios_to_run=["sosp"],  # Test with SOSP again
        num_stochastic=5,  # Reduced for validation
    )


if __name__ == "__main__":
    main()
