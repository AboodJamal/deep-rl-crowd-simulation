"""
Professional VGA+UPL vs DRL Comparison
======================================

This script creates a professional side-by-side comparison between:
- VGA+UPL V4 (physics-based model)
- DRL PPO (deep reinforcement learning model)

For academic/professor assessment with comprehensive metrics.

Author: Generated for research comparison
Date: December 2025
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for stability
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Go up to workspace root
sys.path.insert(0, str(PROJECT_ROOT / "vga_baseline" / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "vga_baseline" / "data_loading"))
sys.path.insert(0, str(PROJECT_ROOT / "drl_training"))
sys.path.insert(0, str(PROJECT_ROOT / "core"))

from vga_upl_planner_v4 import VGAUPLPlannerV4
from vga_dataset import VGADatasetLoader
from vga_experimental_env import VGAExperimentalEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


@dataclass
class TrialMetrics:
    """Comprehensive metrics for a single trial - same for both models"""

    model_name: str
    scenario: str
    trial_idx: int

    # Basic outcome
    success: bool
    final_distance_to_goal: float

    # Trajectory data
    positions: np.ndarray  # Will be converted to list for JSON
    start_pos: np.ndarray
    goal_pos: np.ndarray

    # Time & Efficiency
    num_steps: int
    travel_time: float
    path_length: float
    optimal_path_length: float
    path_efficiency: float  # optimal / actual

    # Speed
    average_speed: float
    max_speed: float
    speed_variance: float

    # Collisions
    num_collisions: int
    collision_frames: int

    # Smoothness
    average_acceleration: float
    max_acceleration: float
    average_jerk: float
    max_jerk: float

    # Path Quality
    direction_changes: int
    oscillation_index: float
    average_deviation: float
    max_deviation: float

    # Safety
    min_clearance: float
    average_clearance: float
    danger_zone_ratio: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        d = asdict(self)
        d["positions"] = (
            self.positions.tolist()
            if isinstance(self.positions, np.ndarray)
            else self.positions
        )
        d["start_pos"] = (
            self.start_pos.tolist()
            if isinstance(self.start_pos, np.ndarray)
            else self.start_pos
        )
        d["goal_pos"] = (
            self.goal_pos.tolist()
            if isinstance(self.goal_pos, np.ndarray)
            else self.goal_pos
        )
        return d


class ProfessionalComparison:
    """Professional comparison between VGA+UPL and DRL models."""

    # Constants matching VGA paper
    GOAL_TOLERANCE = 0.3  # meters
    AGENT_RADIUS = 0.2
    OBSTACLE_RADIUS = 0.25
    DT = 0.05  # 50ms timestep
    MAX_STEPS = 500  # Maximum steps per trial

    # Arena bounds (same for both models)
    ARENA_X_MIN = 0.0
    ARENA_X_MAX = 10.0
    ARENA_Y_MIN = -1.75
    ARENA_Y_MAX = 1.75

    def __init__(
        self,
        drl_model_path: str,
        vec_normalize_path: str = None,
        data_root: str = None,
        output_dir: str = "professional_comparison",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data path
        if data_root is None:
            data_root = str(PROJECT_ROOT / "data" / "VGA-Experimental-Data")
        self.data_root = data_root

        # Load DRL model
        print(f"[LOAD] Loading DRL model: {drl_model_path}")
        self.drl_model = PPO.load(drl_model_path)
        self.vec_normalize_path = vec_normalize_path

        # Create VGA model with matching speed (1.6 m/s to match DRL's MAX_VELOCITY)
        # This ensures fair comparison - both models have same speed capability
        self.vga_speed = 1.6  # Match DRL's MAX_VELOCITY for fair comparison
        print(
            f"[LOAD] Creating VGA+UPL V4 model (speed={self.vga_speed} m/s - matched to DRL)"
        )
        self.vga_model = VGAUPLPlannerV4(
            use_probabilistic=False, desired_speed=self.vga_speed
        )

        # Load dataset
        print(f"[LOAD] Loading VGA dataset from: {data_root}")
        self.dataset_loader = VGADatasetLoader(data_root=data_root)

        self.scenarios = ["SOSP", "MOSP_A", "MOSP_B", "MOSP_C", "MOSP_D"]

    def get_trial_data(self, scenario: str, trial_idx: int) -> dict:
        """Get trial data (start, goal, obstacles) for a specific scenario/trial."""
        if scenario == "SOSP":
            trials = self.dataset_loader.load_sosp()
        else:
            case = scenario.split("_")[1]
            trials = self.dataset_loader.load_mosp(case)

        trial = trials[trial_idx]

        # Convert obstacles to tuple format (x, y, r)
        obstacles = [
            (obs["position"][0], obs["position"][1], obs["radius"])
            for obs in trial.obstacles
        ]

        return {
            "start_pos": trial.initial_pos.copy(),
            "goal_pos": trial.final_pos.copy(),
            "obstacles": obstacles,
        }

    def compute_metrics(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        obstacles: List[Tuple[float, float, float]],
        model_name: str,
        scenario: str,
        trial_idx: int,
    ) -> TrialMetrics:
        """Compute comprehensive metrics from trajectory data."""

        # Final distance
        final_pos = positions[-1]
        final_distance = np.linalg.norm(final_pos - goal_pos)
        success = final_distance < self.GOAL_TOLERANCE

        # Path metrics
        path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        optimal_path = np.linalg.norm(goal_pos - start_pos)
        path_efficiency = optimal_path / path_length if path_length > 0 else 0

        # Time
        num_steps = len(positions)
        travel_time = num_steps * self.DT

        # Speed metrics
        speeds = np.linalg.norm(velocities, axis=1)
        avg_speed = np.mean(speeds)
        max_speed = np.max(speeds)
        speed_variance = np.var(speeds)

        # Collision detection
        num_collisions = 0
        collision_frames = 0
        in_collision_prev = False

        for pos in positions:
            in_collision = False
            for ox, oy, r in obstacles:
                dist = np.linalg.norm(pos - np.array([ox, oy]))
                if dist < r + self.AGENT_RADIUS:
                    in_collision = True
                    collision_frames += 1
                    break
            if in_collision and not in_collision_prev:
                num_collisions += 1
            in_collision_prev = in_collision

        # Smoothness - acceleration and jerk
        if len(velocities) > 1:
            accelerations = np.diff(velocities, axis=0) / self.DT
            acc_mags = np.linalg.norm(accelerations, axis=1)
            avg_acc = np.mean(acc_mags)
            max_acc = np.max(acc_mags)

            if len(accelerations) > 1:
                jerks = np.diff(accelerations, axis=0) / self.DT
                jerk_mags = np.linalg.norm(jerks, axis=1)
                avg_jerk = np.mean(jerk_mags)
                max_jerk = np.max(jerk_mags)
            else:
                avg_jerk = max_jerk = 0.0
        else:
            avg_acc = max_acc = avg_jerk = max_jerk = 0.0

        # Direction changes and oscillation
        headings = np.arctan2(velocities[:, 1], velocities[:, 0])
        heading_changes = np.abs(np.diff(headings))
        heading_changes = np.minimum(heading_changes, 2 * np.pi - heading_changes)
        direction_changes = int(np.sum(heading_changes > 0.3))  # > ~17 degrees
        oscillation_index = float(np.sum(heading_changes))

        # Deviation from straight line
        if optimal_path > 0.01:
            path_dir = (goal_pos - start_pos) / optimal_path
            deviations = []
            for pos in positions:
                to_pos = pos - start_pos
                proj_len = np.dot(to_pos, path_dir)
                proj_point = start_pos + proj_len * path_dir
                dev = np.linalg.norm(pos - proj_point)
                deviations.append(dev)
            avg_deviation = np.mean(deviations)
            max_deviation = np.max(deviations)
        else:
            avg_deviation = max_deviation = 0.0

        # Safety / Clearance
        clearances = []
        danger_frames = 0
        for pos in positions:
            min_clearance = float("inf")
            for ox, oy, r in obstacles:
                dist = np.linalg.norm(pos - np.array([ox, oy])) - r
                min_clearance = min(min_clearance, dist)
            if min_clearance < float("inf"):
                clearances.append(min_clearance)
                if min_clearance < self.AGENT_RADIUS * 1.5:
                    danger_frames += 1

        min_clearance_overall = min(clearances) if clearances else float("inf")
        avg_clearance = np.mean(clearances) if clearances else float("inf")
        danger_zone_ratio = danger_frames / len(positions) if positions.size > 0 else 0

        return TrialMetrics(
            model_name=model_name,
            scenario=scenario,
            trial_idx=trial_idx,
            success=success,
            final_distance_to_goal=final_distance,
            positions=positions,
            start_pos=start_pos,
            goal_pos=goal_pos,
            num_steps=num_steps,
            travel_time=travel_time,
            path_length=path_length,
            optimal_path_length=optimal_path,
            path_efficiency=path_efficiency,
            average_speed=avg_speed,
            max_speed=max_speed,
            speed_variance=speed_variance,
            num_collisions=num_collisions,
            collision_frames=collision_frames,
            average_acceleration=avg_acc,
            max_acceleration=max_acc,
            average_jerk=avg_jerk,
            max_jerk=max_jerk,
            direction_changes=direction_changes,
            oscillation_index=oscillation_index,
            average_deviation=avg_deviation,
            max_deviation=max_deviation,
            min_clearance=min_clearance_overall,
            average_clearance=avg_clearance,
            danger_zone_ratio=danger_zone_ratio,
        )

    def run_vga_trial(
        self, trial_data: dict, scenario: str, trial_idx: int
    ) -> TrialMetrics:
        """Run VGA+UPL model on a trial."""
        result = self.vga_model.simulate(
            start_pos=trial_data["start_pos"],
            goal_pos=trial_data["goal_pos"],
            obstacles=trial_data["obstacles"],
            max_steps=self.MAX_STEPS,
        )

        return self.compute_metrics(
            positions=result.positions,
            velocities=result.velocities,
            start_pos=trial_data["start_pos"],
            goal_pos=trial_data["goal_pos"],
            obstacles=trial_data["obstacles"],
            model_name="VGA+UPL V4",
            scenario=scenario,
            trial_idx=trial_idx,
        )

    def run_drl_trial(
        self, trial_data: dict, scenario: str, trial_idx: int
    ) -> TrialMetrics:
        """
        Run DRL model on a trial - EXACTLY like evaluate_vga_drl.py does it!

        Key insight: DON'T override trial data. The env loads the same data file as VGA.
        Just navigate to the correct trial_idx and run normally.
        """

        # Create base environment FIRST (same pattern as evaluate_vga_drl.py line 86-91)
        env = VGAExperimentalEnv(
            scenario=scenario,
            data_root=self.data_root,
            randomize_start_goal=False,  # CRITICAL: no randomization
        )

        # Wrap with DummyVecEnv keeping reference to env (line 96)
        vec_env = DummyVecEnv([lambda: env])

        # Load normalization if available (lines 97-100)
        if self.vec_normalize_path and os.path.exists(self.vec_normalize_path):
            vec_env = VecNormalize.load(self.vec_normalize_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False

        # Navigate to correct trial (env cycles through trials on each reset)
        # Reset trial_idx+1 times to land on trial_idx (since first reset goes to trial 0)
        for _ in range(trial_idx + 1):
            obs = vec_env.reset()

        # NOW env has the correct trial data loaded
        # Store start/goal from env (should match trial_data, but use env's values for consistency)
        start_pos = env.agent_pos.copy()
        goal_pos = env.goal_pos.copy()
        obstacles = [(x, y, r) for x, y, r in env.obstacles]

        # Run episode (exact pattern from evaluate_vga_drl.py lines 113-162)
        positions = [env.agent_pos.copy()]
        velocities = [env.agent_vel.copy()]

        done = False
        while not done:
            # Record position BEFORE step (critical for terminal step!)
            pre_step_pos = env.agent_pos.copy()
            pre_step_vel = env.agent_vel.copy()

            action, _ = self.drl_model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = vec_env.step(action)
            done = done_arr[0] if hasattr(done_arr, "__len__") else done_arr

            if done:
                # IMPORTANT: After done=True, VecEnv auto-resets!
                # env.agent_pos is now the NEXT trial's start position (WRONG!)
                # Use info to determine outcome
                info_dict = info[0] if isinstance(info, list) else info

                if info_dict.get("goal_reached", False):
                    # Success - agent reached goal
                    positions.append(goal_pos.copy())
                    velocities.append(pre_step_vel)
                else:
                    # Failure - use pre-step position
                    positions.append(pre_step_pos)
                    velocities.append(pre_step_vel)
            else:
                # Not done - env hasn't reset, current position is valid
                positions.append(env.agent_pos.copy())
                velocities.append(env.agent_vel.copy())

        vec_env.close()

        positions = np.array(positions)
        velocities = np.array(velocities)

        return self.compute_metrics(
            positions=positions,
            velocities=velocities,
            start_pos=start_pos,  # Use env's values, not trial_data (they should match)
            goal_pos=goal_pos,
            obstacles=obstacles,
            model_name="DRL (PPO)",
            scenario=scenario,
            trial_idx=trial_idx,
        )

    def create_comparison_image(
        self,
        vga_metrics: TrialMetrics,
        drl_metrics: TrialMetrics,
        obstacles: List[Tuple[float, float, float]],
        save_path: Path,
    ):
        """Create professional side-by-side comparison image."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))

            for ax, metrics in zip(axes, [vga_metrics, drl_metrics]):
                # Set identical bounds
                ax.set_xlim(self.ARENA_X_MIN - 0.3, self.ARENA_X_MAX + 0.3)
                ax.set_ylim(self.ARENA_Y_MIN - 0.3, self.ARENA_Y_MAX + 0.3)

                # Draw obstacles - dark red with black borders for visibility
                for ox, oy, r in obstacles:
                    circle = Circle(
                        (ox, oy),
                        r,
                        facecolor="darkred",
                        alpha=0.8,
                        ec="black",
                        lw=2,
                        zorder=1,
                    )
                    ax.add_patch(circle)

                # Draw goal zone
                goal_zone = Circle(
                    metrics.goal_pos,
                    self.GOAL_TOLERANCE,
                    color="green",
                    alpha=0.2,
                    ec="green",
                    lw=2,
                    linestyle="--",
                )
                ax.add_patch(goal_zone)

                # Draw trajectory
                positions = metrics.positions
                if len(positions) > 1:
                    ax.plot(
                        positions[:, 0], positions[:, 1], "b-", linewidth=2, alpha=0.8
                    )

                # Start and goal markers
                ax.plot(
                    metrics.start_pos[0],
                    metrics.start_pos[1],
                    "go",
                    markersize=12,
                    label="Start",
                    zorder=5,
                )
                ax.plot(
                    metrics.goal_pos[0],
                    metrics.goal_pos[1],
                    "r*",
                    markersize=16,
                    label="Goal",
                    zorder=5,
                )

                # Final position marker
                final_pos = positions[-1]
                ax.plot(
                    final_pos[0],
                    final_pos[1],
                    "b^",
                    markersize=10,
                    label="Final",
                    zorder=5,
                )

                # Status
                status = "SUCCESS" if metrics.success else "FAILURE"

                title = f"{metrics.model_name}\n"
                title += f"{status} | Path: {metrics.path_length:.2f}m | "
                title += f"Time: {metrics.travel_time:.1f}s | Eff: {metrics.path_efficiency:.0%}"

                ax.set_title(title, fontsize=10, fontweight="bold")
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_aspect("equal")

            fig.suptitle(
                f"{vga_metrics.scenario} - Trial {vga_metrics.trial_idx + 1}",
                fontsize=12,
                fontweight="bold",
            )

            plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        except Exception as e:
            print(f"    [WARN] Image save error: {e}")
            plt.close("all")

    def create_comparison_video(
        self,
        vga_metrics: TrialMetrics,
        drl_metrics: TrialMetrics,
        obstacles: List[Tuple[float, float, float]],
        save_path: Path,
    ):
        """Create side-by-side animated comparison video."""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Setup static elements
        for ax, metrics in zip(axes, [vga_metrics, drl_metrics]):
            ax.set_xlim(self.ARENA_X_MIN - 0.5, self.ARENA_X_MAX + 0.5)
            ax.set_ylim(self.ARENA_Y_MIN - 0.5, self.ARENA_Y_MAX + 0.5)

            # Obstacles - dark red with black borders for visibility
            for ox, oy, r in obstacles:
                circle = Circle(
                    (ox, oy),
                    r,
                    facecolor="darkred",
                    alpha=0.8,
                    ec="black",
                    lw=2,
                    zorder=1,
                )
                ax.add_patch(circle)

            # Goal zone
            goal_zone = Circle(
                metrics.goal_pos,
                self.GOAL_TOLERANCE,
                color="green",
                alpha=0.2,
                ec="green",
                lw=2,
                linestyle="--",
            )
            ax.add_patch(goal_zone)

            # Start and goal
            ax.plot(
                metrics.start_pos[0],
                metrics.start_pos[1],
                "go",
                markersize=15,
                zorder=5,
            )
            ax.plot(
                metrics.goal_pos[0], metrics.goal_pos[1], "r*", markersize=20, zorder=5
            )

            status = "✓ SUCCESS" if metrics.success else "✗ FAILURE"
            ax.set_title(
                f"{metrics.model_name}\n{status}", fontsize=12, fontweight="bold"
            )
            ax.set_xlabel("X (meters)")
            ax.set_ylabel("Y (meters)")
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal")

        plt.suptitle(
            f"{vga_metrics.scenario} - Trial {vga_metrics.trial_idx + 1}",
            fontsize=14,
            fontweight="bold",
        )

        # Animation elements
        (vga_line,) = axes[0].plot([], [], "b-", linewidth=2, alpha=0.7)
        vga_agent = Circle(
            (0, 0), self.AGENT_RADIUS, color="blue", alpha=0.8, zorder=10
        )
        axes[0].add_patch(vga_agent)

        (drl_line,) = axes[1].plot([], [], "b-", linewidth=2, alpha=0.7)
        drl_agent = Circle(
            (0, 0), self.AGENT_RADIUS, color="blue", alpha=0.8, zorder=10
        )
        axes[1].add_patch(drl_agent)

        vga_pos = vga_metrics.positions
        drl_pos = drl_metrics.positions
        max_frames = max(len(vga_pos), len(drl_pos))

        def init():
            vga_line.set_data([], [])
            drl_line.set_data([], [])
            vga_agent.center = vga_pos[0]
            drl_agent.center = drl_pos[0]
            return vga_line, vga_agent, drl_line, drl_agent

        def animate(frame):
            # VGA
            idx = min(frame, len(vga_pos) - 1)
            vga_line.set_data(vga_pos[: idx + 1, 0], vga_pos[: idx + 1, 1])
            vga_agent.center = vga_pos[idx]

            # DRL
            idx = min(frame, len(drl_pos) - 1)
            drl_line.set_data(drl_pos[: idx + 1, 0], drl_pos[: idx + 1, 1])
            drl_agent.center = drl_pos[idx]

            return vga_line, vga_agent, drl_line, drl_agent

        anim = animation.FuncAnimation(
            fig, animate, init_func=init, frames=max_frames, interval=50, blit=True
        )

        # Save as MP4 with ffmpeg (more reliable for many frames)
        try:
            Writer = animation.writers["ffmpeg"]
            writer = Writer(
                fps=20, metadata=dict(artist="VGA Comparison"), bitrate=1800
            )
            anim.save(str(save_path), writer=writer)
        except Exception as e:
            print(f"    [WARN] Could not save video: {e}")

        plt.close(fig)

    def run_comparison(self, trials_per_scenario: int = None):
        """Run full professional comparison.

        Args:
            trials_per_scenario: Number of trials per scenario.
                                If None, uses all available trials in dataset.
        """
        print("=" * 70)
        print("PROFESSIONAL VGA+UPL vs DRL COMPARISON")
        print("=" * 70)

        # Determine trial counts per scenario
        if trials_per_scenario is None:
            # Use all available trials
            trial_counts = {
                "SOSP": len(self.dataset_loader.load_sosp()),
                "MOSP_A": len(self.dataset_loader.load_mosp("A")),
                "MOSP_B": len(self.dataset_loader.load_mosp("B")),
                "MOSP_C": len(self.dataset_loader.load_mosp("C")),
                "MOSP_D": len(self.dataset_loader.load_mosp("D")),
            }
            print("Running on ALL available trials:")
            for scenario, count in trial_counts.items():
                print(f"  {scenario}: {count} trials")
            total_trials = sum(trial_counts.values())
            print(f"  TOTAL: {total_trials} trials")
        else:
            # Use specified number
            trial_counts = {s: trials_per_scenario for s in self.scenarios}
            print(f"Trials per scenario: {trials_per_scenario}")

        print(f"Scenarios: {', '.join(self.scenarios)}")
        print()

        all_results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "trial_counts": trial_counts,
                "goal_tolerance": self.GOAL_TOLERANCE,
                "max_steps": self.MAX_STEPS,
                "dt": self.DT,
            },
            "scenarios": {},
        }

        for scenario in self.scenarios:
            print(f"\n[SCENARIO] {scenario}")
            print("-" * 50)

            # Create output directories
            scenario_dir = self.output_dir / scenario
            (scenario_dir / "images").mkdir(parents=True, exist_ok=True)
            (scenario_dir / "videos").mkdir(parents=True, exist_ok=True)

            scenario_results = {
                "vga_trials": [],
                "drl_trials": [],
            }

            # Get trial count for this scenario
            num_trials = trial_counts[scenario]

            # Determine which trials to generate visualizations for (10 trials max)
            # Spread them evenly across the dataset
            if num_trials <= 10:
                vis_trials = list(range(num_trials))
            else:
                step = num_trials // 10
                vis_trials = [i * step for i in range(10)]

            for trial_idx in range(num_trials):
                print(
                    f"  Trial {trial_idx + 1}/{num_trials}...",
                    end=" ",
                    flush=True,
                )

                generate_viz = trial_idx in vis_trials

                try:
                    # Get IDENTICAL trial data for both models
                    trial_data = self.get_trial_data(scenario, trial_idx)

                    # Run VGA
                    vga_metrics = self.run_vga_trial(trial_data, scenario, trial_idx)

                    # Run DRL (with same start/goal/obstacles)
                    drl_metrics = self.run_drl_trial(trial_data, scenario, trial_idx)

                    # Store results
                    scenario_results["vga_trials"].append(vga_metrics.to_dict())
                    scenario_results["drl_trials"].append(drl_metrics.to_dict())

                    # Generate visualizations only for selected trials
                    if generate_viz:
                        # Generate image
                        img_path = (
                            scenario_dir / "images" / f"trial_{trial_idx + 1:03d}.png"
                        )
                        self.create_comparison_image(
                            vga_metrics, drl_metrics, trial_data["obstacles"], img_path
                        )

                        # Generate video
                        vid_path = (
                            scenario_dir / "videos" / f"trial_{trial_idx + 1:03d}.mp4"
                        )
                        self.create_comparison_video(
                            vga_metrics, drl_metrics, trial_data["obstacles"], vid_path
                        )

                    vga_status = "OK" if vga_metrics.success else "FAIL"
                    drl_status = "OK" if drl_metrics.success else "FAIL"
                    viz_marker = " [VIZ]" if generate_viz else ""
                    print(f"VGA: {vga_status} | DRL: {drl_status}{viz_marker}")

                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback

                    traceback.print_exc()

            all_results["scenarios"][scenario] = scenario_results

        # Generate summary
        self.generate_summary(all_results)

        # Save results
        results_path = self.output_dir / "comparison_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        print("\n" + "=" * 70)
        print("[DONE] Comparison complete!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 70)

    def generate_summary(self, results: dict):
        """Generate comparison summary table."""
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)

        header = f"{'Scenario':<10} {'Model':<15} {'Success':<10} {'Avg Path':<12} {'Avg Time':<10} {'Efficiency':<12} {'Collisions':<10}"
        print(header)
        print("-" * 80)

        summary_data = []

        for scenario, data in results["scenarios"].items():
            vga_trials = data["vga_trials"]
            drl_trials = data["drl_trials"]

            if not vga_trials or not drl_trials:
                continue

            # VGA stats
            vga_success = (
                sum(1 for t in vga_trials if t["success"]) / len(vga_trials) * 100
            )
            vga_path = np.mean([t["path_length"] for t in vga_trials])
            vga_time = np.mean([t["travel_time"] for t in vga_trials])
            vga_eff = np.mean([t["path_efficiency"] for t in vga_trials])
            vga_coll = np.mean([t["num_collisions"] for t in vga_trials])

            # DRL stats
            drl_success = (
                sum(1 for t in drl_trials if t["success"]) / len(drl_trials) * 100
            )
            drl_path = np.mean([t["path_length"] for t in drl_trials])
            drl_time = np.mean([t["travel_time"] for t in drl_trials])
            drl_eff = np.mean([t["path_efficiency"] for t in drl_trials])
            drl_coll = np.mean([t["num_collisions"] for t in drl_trials])

            print(
                f"{scenario:<10} {'VGA+UPL V4':<15} {vga_success:>6.1f}%   {vga_path:>10.2f}m {vga_time:>8.2f}s  {vga_eff:>10.1%}    {vga_coll:>8.1f}"
            )
            print(
                f"{'':<10} {'DRL (PPO)':<15} {drl_success:>6.1f}%   {drl_path:>10.2f}m {drl_time:>8.2f}s  {drl_eff:>10.1%}    {drl_coll:>8.1f}"
            )
            print()

            summary_data.append(
                {
                    "scenario": scenario,
                    "vga": {
                        "success": vga_success,
                        "path": vga_path,
                        "time": vga_time,
                        "efficiency": vga_eff,
                        "collisions": vga_coll,
                    },
                    "drl": {
                        "success": drl_success,
                        "path": drl_path,
                        "time": drl_time,
                        "efficiency": drl_eff,
                        "collisions": drl_coll,
                    },
                }
            )

        # Overall stats
        all_vga = [t for s in results["scenarios"].values() for t in s["vga_trials"]]
        all_drl = [t for s in results["scenarios"].values() for t in s["drl_trials"]]

        if all_vga and all_drl:
            print("-" * 80)
            vga_total_success = (
                sum(1 for t in all_vga if t["success"]) / len(all_vga) * 100
            )
            drl_total_success = (
                sum(1 for t in all_drl if t["success"]) / len(all_drl) * 100
            )

            print(f"{'OVERALL':<10} {'VGA+UPL V4':<15} {vga_total_success:>6.1f}%")
            print(f"{'':<10} {'DRL (PPO)':<15} {drl_total_success:>6.1f}%")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Professional VGA vs DRL comparison")
    parser.add_argument("--trials", type=int, default=20, help="Trials per scenario")
    args = parser.parse_args()

    # Paths
    drl_model_path = str(PROJECT_ROOT / "drl_training" / "models" / "vga_drl_final.zip")
    # VecNormalize auto-detect (same pattern as evaluate_vga_drl.py)
    vec_normalize_path = drl_model_path.replace(".zip", "_vecnormalize.pkl")

    if not os.path.exists(drl_model_path):
        print(f"[ERROR] DRL model not found: {drl_model_path}")
        return

    # Check VecNormalize
    if os.path.exists(vec_normalize_path):
        print(f"[OK] Found VecNormalize: {vec_normalize_path}")
    else:
        print(f"[WARN] VecNormalize not found: {vec_normalize_path}")
        vec_normalize_path = None

    comparison = ProfessionalComparison(
        drl_model_path=drl_model_path,
        vec_normalize_path=vec_normalize_path,
        output_dir="comparisons/static_obstacles_fullRun",
    )

    # Run full comparison on ALL trials in dataset
    # SOSP: 54, MOSP_A: 239, MOSP_B: 188, MOSP_C: 184, MOSP_D: 276
    # Total: 941 trials
    comparison.run_comparison(
        trials_per_scenario=None
    )  # None = use all available trials


if __name__ == "__main__":
    main()
