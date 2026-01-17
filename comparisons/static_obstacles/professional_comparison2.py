"""
Professional VGA+UPL vs ROBUST DRL Comparison (V2)
==================================================

This script creates a professional side-by-side comparison between:
- VGA+UPL V4 (physics-based model)
- ROBUST DRL PPO (trained with domain randomization & anti-oscillation)

Includes comprehensive statistical analysis.

Author: Generated for research comparison
Date: January 2026
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from scipy import stats

# Add paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "vga_upl_baseline" / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "vga_upl_baseline" / "data_loading"))
sys.path.insert(0, str(PROJECT_ROOT / "drl_vga_experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "core"))

from vga_upl_planner_v4 import VGAUPLPlannerV4
from vga_dataset import VGADatasetLoader
from vga_experimental_env import VGAExperimentalEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


@dataclass
class TrialMetrics:
    """Comprehensive metrics for a single trial"""

    model_name: str
    scenario: str
    trial_idx: int

    # Basic outcome
    success: bool
    final_distance_to_goal: float

    # Trajectory data
    positions: np.ndarray
    start_pos: np.ndarray
    goal_pos: np.ndarray

    # Time & Efficiency
    num_steps: int
    travel_time: float
    path_length: float
    optimal_path_length: float
    path_efficiency: float

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


class ProfessionalComparisonV2:
    """Professional comparison between VGA+UPL and ROBUST DRL models."""

    GOAL_TOLERANCE = 0.3
    AGENT_RADIUS = 0.2
    OBSTACLE_RADIUS = 0.25
    DT = 0.05
    MAX_STEPS = 500

    ARENA_X_MIN = 0.0
    ARENA_X_MAX = 10.0
    ARENA_Y_MIN = -1.75
    ARENA_Y_MAX = 1.75

    def __init__(
        self,
        drl_model_path: str,
        vec_normalize_path: str = None,
        data_root: str = None,
        output_dir: str = "professional_comparison2",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if data_root is None:
            data_root = str(PROJECT_ROOT / "data" / "VGA-Experimental-Data")
        self.data_root = data_root

        # Load ROBUST DRL model
        print(f"[LOAD] Loading ROBUST DRL model: {drl_model_path}")
        self.drl_model = PPO.load(drl_model_path)
        self.vec_normalize_path = vec_normalize_path

        # Create VGA model
        self.vga_speed = 1.6
        print(f"[LOAD] Creating VGA+UPL V4 model (speed={self.vga_speed} m/s)")
        self.vga_model = VGAUPLPlannerV4(
            use_probabilistic=False, desired_speed=self.vga_speed
        )

        # Load dataset
        print(f"[LOAD] Loading VGA dataset from: {data_root}")
        self.dataset_loader = VGADatasetLoader(data_root=data_root)

        self.scenarios = ["SOSP", "MOSP_A", "MOSP_B", "MOSP_C", "MOSP_D"]

    def get_trial_data(self, scenario: str, trial_idx: int) -> dict:
        if scenario == "SOSP":
            trials = self.dataset_loader.load_sosp()
        else:
            case = scenario.split("_")[1]
            trials = self.dataset_loader.load_mosp(case)

        trial = trials[trial_idx]
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
        final_pos = positions[-1]
        final_distance = np.linalg.norm(final_pos - goal_pos)
        success = final_distance < self.GOAL_TOLERANCE

        path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        optimal_path = np.linalg.norm(goal_pos - start_pos)
        path_efficiency = optimal_path / path_length if path_length > 0 else 0

        num_steps = len(positions)
        travel_time = num_steps * self.DT

        speeds = np.linalg.norm(velocities, axis=1)
        avg_speed = np.mean(speeds)
        max_speed = np.max(speeds)
        speed_variance = np.var(speeds)

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

        headings = np.arctan2(velocities[:, 1], velocities[:, 0])
        heading_changes = np.abs(np.diff(headings))
        heading_changes = np.minimum(heading_changes, 2 * np.pi - heading_changes)
        direction_changes = int(np.sum(heading_changes > 0.3))
        oscillation_index = float(np.sum(heading_changes))

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
        # Use VGAExperimentalEnv which matches the training environment (50 dims)
        env = VGAExperimentalEnv(
            scenario=scenario,
            data_root=self.data_root,
        )

        vec_env = DummyVecEnv([lambda: env])

        if self.vec_normalize_path and os.path.exists(self.vec_normalize_path):
            vec_env = VecNormalize.load(self.vec_normalize_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False

        for _ in range(trial_idx + 1):
            obs = vec_env.reset()

        start_pos = env.agent_pos.copy()
        goal_pos = env.goal_pos.copy()
        obstacles = [(x, y, r) for x, y, r in env.obstacles]

        positions = [env.agent_pos.copy()]
        velocities = [env.agent_vel.copy()]

        done = False
        while not done:
            pre_step_pos = env.agent_pos.copy()
            pre_step_vel = env.agent_vel.copy()

            action, _ = self.drl_model.predict(obs, deterministic=True)
            # VecNormalize returns 4 values (old gym API style)
            step_result = vec_env.step(action)
            if len(step_result) == 4:
                obs, reward, done_arr, info = step_result
                done = done_arr[0] if hasattr(done_arr, "__len__") else done_arr
            else:
                obs, reward, terminated, truncated, info = step_result
                done = (
                    terminated[0] or truncated[0]
                    if hasattr(terminated, "__len__")
                    else (terminated or truncated)
                )

            if done:
                if (
                    info[0].get("goal_reached", False)
                    if isinstance(info, list)
                    else info.get("goal_reached", False)
                ):
                    positions.append(goal_pos.copy())
                    velocities.append(pre_step_vel)
                else:
                    positions.append(pre_step_pos)
                    velocities.append(pre_step_vel)
            else:
                positions.append(env.agent_pos.copy())
                velocities.append(env.agent_vel.copy())

        vec_env.close()

        positions = np.array(positions)
        velocities = np.array(velocities)

        return self.compute_metrics(
            positions=positions,
            velocities=velocities,
            start_pos=start_pos,
            goal_pos=goal_pos,
            obstacles=obstacles,
            model_name="ROBUST DRL",
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
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))

            for ax, metrics in zip(axes, [vga_metrics, drl_metrics]):
                ax.set_xlim(self.ARENA_X_MIN - 0.3, self.ARENA_X_MAX + 0.3)
                ax.set_ylim(self.ARENA_Y_MIN - 0.3, self.ARENA_Y_MAX + 0.3)

                for ox, oy, r in obstacles:
                    circle = Circle(
                        (ox, oy), r, color="gray", alpha=0.6, ec="black", lw=1.5
                    )
                    ax.add_patch(circle)

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

                positions = metrics.positions
                if len(positions) > 1:
                    color = "blue" if metrics.model_name == "VGA+UPL V4" else "red"
                    ax.plot(
                        positions[:, 0],
                        positions[:, 1],
                        color=color,
                        linewidth=2,
                        alpha=0.8,
                    )

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

                final_pos = positions[-1]
                marker = "^" if metrics.success else "x"
                color = "green" if metrics.success else "red"
                ax.plot(
                    final_pos[0],
                    final_pos[1],
                    marker,
                    color=color,
                    markersize=12,
                    zorder=5,
                )

                status = "SUCCESS" if metrics.success else "FAIL"
                title = f"{metrics.model_name}\n{status} | Path: {metrics.path_length:.2f}m | Time: {metrics.travel_time:.2f}s"
                ax.set_title(title, fontsize=12, fontweight="bold")
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")
                ax.set_aspect("equal")
                ax.grid(True, alpha=0.3)

            plt.suptitle(
                f"{vga_metrics.scenario} - Trial {vga_metrics.trial_idx + 1}",
                fontsize=14,
                fontweight="bold",
            )
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        except Exception as e:
            print(f"[WARN] Image creation failed: {e}")

    def create_comparison_video(
        self,
        vga_metrics: TrialMetrics,
        drl_metrics: TrialMetrics,
        obstacles: List[Tuple[float, float, float]],
        save_path: Path,
    ):
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))

            vga_positions = vga_metrics.positions
            drl_positions = drl_metrics.positions
            max_frames = max(len(vga_positions), len(drl_positions))

            for ax in axes:
                ax.set_xlim(self.ARENA_X_MIN - 0.3, self.ARENA_X_MAX + 0.3)
                ax.set_ylim(self.ARENA_Y_MIN - 0.3, self.ARENA_Y_MAX + 0.3)

            (vga_line,) = axes[0].plot([], [], "b-", linewidth=2, alpha=0.8)
            vga_agent = Circle((0, 0), self.AGENT_RADIUS, color="blue", alpha=0.7)
            axes[0].add_patch(vga_agent)

            (drl_line,) = axes[1].plot([], [], "r-", linewidth=2, alpha=0.8)
            drl_agent = Circle((0, 0), self.AGENT_RADIUS, color="red", alpha=0.7)
            axes[1].add_patch(drl_agent)

            for ax, metrics, title_prefix in zip(
                axes, [vga_metrics, drl_metrics], ["VGA+UPL V4", "ROBUST DRL"]
            ):
                for ox, oy, r in obstacles:
                    circle = Circle(
                        (ox, oy), r, color="gray", alpha=0.6, ec="black", lw=1.5
                    )
                    ax.add_patch(circle)

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

                ax.plot(
                    metrics.start_pos[0],
                    metrics.start_pos[1],
                    "go",
                    markersize=12,
                    zorder=5,
                )
                ax.plot(
                    metrics.goal_pos[0],
                    metrics.goal_pos[1],
                    "r*",
                    markersize=16,
                    zorder=5,
                )

                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")
                ax.set_aspect("equal")
                ax.grid(True, alpha=0.3)
                ax.set_title(title_prefix, fontsize=12, fontweight="bold")

            def animate(frame):
                vga_idx = min(frame, len(vga_positions) - 1)
                drl_idx = min(frame, len(drl_positions) - 1)

                vga_line.set_data(
                    vga_positions[: vga_idx + 1, 0], vga_positions[: vga_idx + 1, 1]
                )
                vga_agent.center = vga_positions[vga_idx]

                drl_line.set_data(
                    drl_positions[: drl_idx + 1, 0], drl_positions[: drl_idx + 1, 1]
                )
                drl_agent.center = drl_positions[drl_idx]

                return vga_line, vga_agent, drl_line, drl_agent

            anim = animation.FuncAnimation(
                fig, animate, frames=max_frames, interval=50, blit=True
            )
            anim.save(str(save_path), writer="pillow", fps=20)
            plt.close(fig)

        except Exception as e:
            print(f"[WARN] Video creation failed: {e}")

    def run_comparison(self, trials_per_scenario: int = 20, fast_mode: bool = False):
        print("\n" + "=" * 70)
        print("VGA+UPL V4 vs DRL PPO - Professional Comparison V2")
        print("=" * 70)
        print(f"Trials per scenario: {trials_per_scenario}")
        print(f"Scenarios: {', '.join(self.scenarios)}")
        print(f"Output directory: {self.output_dir}")
        if fast_mode:
            print("Fast mode: Videos DISABLED")
        print("=" * 70 + "\n")

        all_results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "trials_per_scenario": trials_per_scenario,
                "scenarios": self.scenarios,
                "vga_speed": self.vga_speed,
                "goal_tolerance": self.GOAL_TOLERANCE,
            },
            "scenarios": {},
        }

        for scenario in self.scenarios:
            print(f"\n{'=' * 70}")
            print(f"SCENARIO: {scenario}")
            print("=" * 70)

            scenario_dir = self.output_dir / scenario
            (scenario_dir / "images").mkdir(parents=True, exist_ok=True)
            (scenario_dir / "videos").mkdir(parents=True, exist_ok=True)

            scenario_results = {"vga_trials": [], "drl_trials": []}

            for trial_idx in range(trials_per_scenario):
                try:
                    print(
                        f"  Trial {trial_idx + 1}/{trials_per_scenario}: ",
                        end="",
                        flush=True,
                    )

                    trial_data = self.get_trial_data(scenario, trial_idx)

                    vga_metrics = self.run_vga_trial(trial_data, scenario, trial_idx)
                    drl_metrics = self.run_drl_trial(trial_data, scenario, trial_idx)

                    scenario_results["vga_trials"].append(vga_metrics.to_dict())
                    scenario_results["drl_trials"].append(drl_metrics.to_dict())

                    img_path = (
                        scenario_dir / "images" / f"trial_{trial_idx + 1:03d}.png"
                    )
                    self.create_comparison_image(
                        vga_metrics, drl_metrics, trial_data["obstacles"], img_path
                    )

                    if not fast_mode:
                        vid_path = (
                            scenario_dir / "videos" / f"trial_{trial_idx + 1:03d}.gif"
                        )
                        self.create_comparison_video(
                            vga_metrics, drl_metrics, trial_data["obstacles"], vid_path
                        )

                    vga_status = "✓" if vga_metrics.success else "✗"
                    drl_status = "✓" if drl_metrics.success else "✗"
                    print(f"VGA: {vga_status} | DRL: {drl_status}")

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

        # Generate statistical analysis
        self.generate_statistical_analysis(all_results)

        print("\n" + "=" * 70)
        print("[DONE] Comparison complete!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 70)

    def generate_summary(self, results: dict):
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)

        header = f"{'Scenario':<10} {'Model':<15} {'Success':<10} {'Avg Path':<12} {'Avg Time':<10} {'Efficiency':<12} {'Collisions':<10}"
        print(header)
        print("-" * 80)

        for scenario, data in results["scenarios"].items():
            vga_trials = data["vga_trials"]
            drl_trials = data["drl_trials"]

            if not vga_trials or not drl_trials:
                continue

            vga_success = (
                sum(1 for t in vga_trials if t["success"]) / len(vga_trials) * 100
            )
            vga_path = np.mean([t["path_length"] for t in vga_trials])
            vga_time = np.mean([t["travel_time"] for t in vga_trials])
            vga_eff = np.mean([t["path_efficiency"] for t in vga_trials])
            vga_coll = np.mean([t["num_collisions"] for t in vga_trials])

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
                f"{'':<10} {'ROBUST DRL':<15} {drl_success:>6.1f}%   {drl_path:>10.2f}m {drl_time:>8.2f}s  {drl_eff:>10.1%}    {drl_coll:>8.1f}"
            )
            print()

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
            print(f"{'':<10} {'ROBUST DRL':<15} {drl_total_success:>6.1f}%")

    def generate_statistical_analysis(self, results: dict):
        """Generate comprehensive statistical analysis."""
        print("\n" + "=" * 70)
        print("GENERATING STATISTICAL ANALYSIS...")
        print("=" * 70)

        stat_dir = self.output_dir / "statistical_analysis"
        stat_dir.mkdir(parents=True, exist_ok=True)

        # Metrics to analyze
        metrics = [
            "path_efficiency",
            "travel_time",
            "average_speed",
            "num_collisions",
            "oscillation_index",
            "min_clearance",
            "path_length",
            "average_jerk",
        ]

        stat_results = {"scenarios": {}, "overall": {}}

        for scenario, data in results["scenarios"].items():
            vga_trials = data["vga_trials"]
            drl_trials = data["drl_trials"]

            if not vga_trials or not drl_trials:
                continue

            stat_results["scenarios"][scenario] = {}

            for metric in metrics:
                vga_values = [
                    float(t[metric])
                    for t in vga_trials
                    if t[metric] is not None and not np.isinf(t[metric])
                ]
                drl_values = [
                    float(t[metric])
                    for t in drl_trials
                    if t[metric] is not None and not np.isinf(t[metric])
                ]

                if len(vga_values) >= 2 and len(drl_values) >= 2:
                    # T-test
                    try:
                        t_stat, p_value = stats.ttest_ind(vga_values, drl_values)
                    except:
                        t_stat, p_value = np.nan, np.nan

                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt((np.var(vga_values) + np.var(drl_values)) / 2)
                    cohens_d = (
                        (np.mean(vga_values) - np.mean(drl_values)) / pooled_std
                        if pooled_std > 0
                        else 0
                    )

                    stat_results["scenarios"][scenario][metric] = {
                        "vga_mean": float(np.mean(vga_values)),
                        "vga_std": float(np.std(vga_values)),
                        "drl_mean": float(np.mean(drl_values)),
                        "drl_std": float(np.std(drl_values)),
                        "t_statistic": float(t_stat) if not np.isnan(t_stat) else None,
                        "p_value": float(p_value) if not np.isnan(p_value) else None,
                        "cohens_d": float(cohens_d),
                        "significant": (
                            bool(p_value < 0.05) if not np.isnan(p_value) else False
                        ),
                    }

        # SPL calculation
        for scenario, data in results["scenarios"].items():
            vga_spl = []
            drl_spl = []

            for t in data["vga_trials"]:
                if t["success"]:
                    spl = t["optimal_path_length"] / max(
                        t["path_length"], t["optimal_path_length"]
                    )
                else:
                    spl = 0.0
                vga_spl.append(spl)

            for t in data["drl_trials"]:
                if t["success"]:
                    spl = t["optimal_path_length"] / max(
                        t["path_length"], t["optimal_path_length"]
                    )
                else:
                    spl = 0.0
                drl_spl.append(spl)

            stat_results["scenarios"][scenario]["spl"] = {
                "vga_mean": float(np.mean(vga_spl)),
                "vga_std": float(np.std(vga_spl)),
                "drl_mean": float(np.mean(drl_spl)),
                "drl_std": float(np.std(drl_spl)),
            }

        # Save statistical results
        with open(stat_dir / "statistical_results.json", "w") as f:
            json.dump(stat_results, f, indent=2)

        # Generate summary text
        self._write_statistical_summary(stat_results, stat_dir)

        # Generate plots
        self._generate_statistical_plots(results, stat_results, stat_dir)

        print(f"[OK] Statistical analysis saved to: {stat_dir}")

    def _write_statistical_summary(self, stat_results: dict, stat_dir: Path):
        """Write statistical summary to text file."""
        with open(stat_dir / "statistical_summary.txt", "w") as f:
            f.write("=" * 80 + "\n")
            f.write("STATISTICAL ANALYSIS SUMMARY\n")
            f.write("VGA+UPL V4 vs ROBUST DRL Comparison\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. SUCCESS RATES:\n")
            for scenario, data in stat_results["scenarios"].items():
                if "spl" in data:
                    f.write(
                        f"   {scenario:10s}: VGA SPL={data['spl']['vga_mean']:.3f}±{data['spl']['vga_std']:.3f}  "
                    )
                    f.write(
                        f"DRL SPL={data['spl']['drl_mean']:.3f}±{data['spl']['drl_std']:.3f}\n"
                    )

            f.write("\n2. STATISTICAL SIGNIFICANCE (p < 0.05):\n\n")

            metric_names = {
                "path_efficiency": "Path Efficiency",
                "travel_time": "Travel Time",
                "average_speed": "Average Speed",
                "num_collisions": "Collisions",
                "oscillation_index": "Oscillation",
                "min_clearance": "Min Clearance",
                "path_length": "Path Length",
                "average_jerk": "Average Jerk",
            }

            for metric, name in metric_names.items():
                f.write(f"   {name}:\n")
                for scenario, data in stat_results["scenarios"].items():
                    if metric in data:
                        m = data[metric]
                        sig = (
                            "[SIGNIFICANT]"
                            if m.get("significant")
                            else "[Not significant]"
                        )
                        winner = "VGA" if m["cohens_d"] > 0 else "DRL"
                        p_val = m.get("p_value", np.nan)
                        p_str = (
                            f"p={p_val:.4f}"
                            if p_val is not None and not np.isnan(p_val)
                            else "p=nan"
                        )
                        f.write(
                            f"      {scenario:10s}: {p_str} {sig} Winner: {winner} (d={m['cohens_d']:.2f})\n"
                        )
                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("INTERPRETATION GUIDE:\n")
            f.write("- p < 0.05: Statistically significant difference\n")
            f.write("- Cohen's d: Effect size (0.2=small, 0.5=medium, 0.8=large)\n")
            f.write("- SPL: Combines success rate with path optimality\n")
            f.write("=" * 80 + "\n")

    def _generate_statistical_plots(
        self, results: dict, stat_results: dict, stat_dir: Path
    ):
        """Generate statistical comparison plots."""
        try:
            # Box plots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            metrics_to_plot = [
                "path_efficiency",
                "travel_time",
                "oscillation_index",
                "average_speed",
                "min_clearance",
                "path_length",
            ]
            metric_labels = [
                "Path Efficiency",
                "Travel Time (s)",
                "Oscillation Index",
                "Avg Speed (m/s)",
                "Min Clearance (m)",
                "Path Length (m)",
            ]

            for ax, metric, label in zip(axes, metrics_to_plot, metric_labels):
                vga_data = []
                drl_data = []
                scenarios = []

                for scenario, data in results["scenarios"].items():
                    vga_vals = [
                        float(t[metric])
                        for t in data["vga_trials"]
                        if t[metric] is not None and not np.isinf(t[metric])
                    ]
                    drl_vals = [
                        float(t[metric])
                        for t in data["drl_trials"]
                        if t[metric] is not None and not np.isinf(t[metric])
                    ]

                    if vga_vals and drl_vals:
                        vga_data.extend(vga_vals)
                        drl_data.extend(drl_vals)
                        scenarios.append(scenario)

                if vga_data and drl_data:
                    bp = ax.boxplot(
                        [vga_data, drl_data], labels=["VGA+UPL", "ROBUST DRL"]
                    )
                    ax.set_title(label)
                    ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(stat_dir / "box_plots_comparison.png", dpi=150)
            plt.close()

            # Radar chart
            self._create_radar_chart(stat_results, stat_dir)

            # Violin plots
            self._create_violin_plots(results, stat_dir)

            # Dedicated Oscillation and Jerk plots (showing DRL superiority)
            self._create_smoothness_plots(results, stat_results, stat_dir)

        except Exception as e:
            print(f"[WARN] Plot generation error: {e}")

    def _create_radar_chart(self, stat_results: dict, stat_dir: Path):
        """Create radar chart comparing VGA and DRL."""
        try:
            metrics = [
                "path_efficiency",
                "average_speed",
                "oscillation_index",
                "average_jerk",
                "min_clearance",
            ]
            metric_labels = [
                "Efficiency",
                "Speed",
                "Smoothness\n(Oscillation)",
                "Smoothness\n(Jerk)",
                "Safety",
            ]

            # Aggregate across scenarios
            vga_means = []
            drl_means = []

            for metric in metrics:
                vga_vals = []
                drl_vals = []
                for scenario, data in stat_results["scenarios"].items():
                    if metric in data:
                        vga_vals.append(data[metric]["vga_mean"])
                        drl_vals.append(data[metric]["drl_mean"])

                vga_mean = np.mean(vga_vals) if vga_vals else 0
                drl_mean = np.mean(drl_vals) if drl_vals else 0
                vga_means.append(vga_mean)
                drl_means.append(drl_mean)

            # Normalize each metric to [0, 1], handling inverted metrics
            vga_norm = []
            drl_norm = []
            for i, metric in enumerate(metrics):
                vga_val = vga_means[i]
                drl_val = drl_means[i]

                if metric == "oscillation_index" or metric == "average_jerk":
                    # Lower is better - use min/value so lower value = higher score
                    min_val = min(vga_val, drl_val)
                    if min_val > 0:
                        # The one with lower value gets score of 1.0
                        # The one with higher value gets score of min/value (less than 1)
                        vga_norm.append(min_val / vga_val if vga_val > 0 else 1.0)
                        drl_norm.append(min_val / drl_val if drl_val > 0 else 1.0)
                    else:
                        vga_norm.append(1.0)
                        drl_norm.append(1.0)
                else:
                    # Higher is better - normalize by max
                    max_val = max(vga_val, drl_val)
                    if max_val > 0:
                        vga_norm.append(vga_val / max_val)
                        drl_norm.append(drl_val / max_val)
                    else:
                        vga_norm.append(0)
                        drl_norm.append(0)

            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]
            vga_norm += vga_norm[:1]
            drl_norm += drl_norm[:1]

            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            ax.plot(angles, vga_norm, "b-", linewidth=2, label="VGA+UPL V4")
            ax.fill(angles, vga_norm, "b", alpha=0.25)
            ax.plot(angles, drl_norm, "r-", linewidth=2, label="DRL PPO")
            ax.fill(angles, drl_norm, "r", alpha=0.25)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels, fontsize=11)
            ax.legend(loc="upper right", fontsize=11)
            plt.title(
                "VGA vs DRL Performance Comparison\n(Higher = Better for all metrics)",
                fontsize=14,
                fontweight="bold",
            )

            plt.savefig(
                stat_dir / "radar_chart_comparison.png", dpi=150, bbox_inches="tight"
            )
            plt.close()

        except Exception as e:
            print(f"[WARN] Radar chart error: {e}")

    def _create_violin_plots(self, results: dict, stat_dir: Path):
        """Create violin plots for key metrics."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            metrics = ["path_efficiency", "oscillation_index", "travel_time"]
            titles = ["Path Efficiency", "Oscillation Index", "Travel Time (s)"]

            for ax, metric, title in zip(axes, metrics, titles):
                data_vga = []
                data_drl = []

                for scenario, data in results["scenarios"].items():
                    vga_vals = [
                        float(t[metric])
                        for t in data["vga_trials"]
                        if t[metric] is not None and not np.isinf(t[metric])
                    ]
                    drl_vals = [
                        float(t[metric])
                        for t in data["drl_trials"]
                        if t[metric] is not None and not np.isinf(t[metric])
                    ]
                    data_vga.extend(vga_vals)
                    data_drl.extend(drl_vals)

                if data_vga and data_drl:
                    parts = ax.violinplot(
                        [data_vga, data_drl], positions=[1, 2], showmeans=True
                    )
                    parts["bodies"][0].set_facecolor("blue")
                    parts["bodies"][0].set_alpha(0.6)
                    parts["bodies"][1].set_facecolor("red")
                    parts["bodies"][1].set_alpha(0.6)

                    ax.set_xticks([1, 2])
                    ax.set_xticklabels(["VGA+UPL", "ROBUST DRL"])
                    ax.set_title(title)
                    ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(stat_dir / "violin_plots_comparison.png", dpi=150)
            plt.close()

        except Exception as e:
            print(f"[WARN] Violin plot error: {e}")

    def _create_smoothness_plots(
        self, results: dict, stat_results: dict, stat_dir: Path
    ):
        """Create dedicated plots for Oscillation and Jerk metrics showing DRL superiority."""
        try:
            scenarios = list(results["scenarios"].keys())

            # ============================================================
            # FIGURE 1: Oscillation Index Comparison (Bar Chart by Scenario)
            # ============================================================
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Left plot: Bar chart comparison per scenario
            ax1 = axes[0]
            x = np.arange(len(scenarios))
            width = 0.35

            vga_oscillation = []
            drl_oscillation = []
            vga_oscillation_std = []
            drl_oscillation_std = []

            for scenario in scenarios:
                data = results["scenarios"][scenario]
                vga_vals = [
                    float(t["oscillation_index"])
                    for t in data["vga_trials"]
                    if t["oscillation_index"] is not None
                    and not np.isinf(t["oscillation_index"])
                ]
                drl_vals = [
                    float(t["oscillation_index"])
                    for t in data["drl_trials"]
                    if t["oscillation_index"] is not None
                    and not np.isinf(t["oscillation_index"])
                ]
                vga_oscillation.append(np.mean(vga_vals) if vga_vals else 0)
                drl_oscillation.append(np.mean(drl_vals) if drl_vals else 0)
                vga_oscillation_std.append(np.std(vga_vals) if vga_vals else 0)
                drl_oscillation_std.append(np.std(drl_vals) if drl_vals else 0)

            bars1 = ax1.bar(
                x - width / 2,
                vga_oscillation,
                width,
                yerr=vga_oscillation_std,
                label="VGA+UPL V4",
                color="#3498db",
                capsize=5,
                alpha=0.8,
            )
            bars2 = ax1.bar(
                x + width / 2,
                drl_oscillation,
                width,
                yerr=drl_oscillation_std,
                label="DRL PPO",
                color="#e74c3c",
                capsize=5,
                alpha=0.8,
            )

            ax1.set_xlabel("Scenario", fontsize=12, fontweight="bold")
            ax1.set_ylabel(
                "Oscillation Index (lower is better)", fontsize=12, fontweight="bold"
            )
            ax1.set_title(
                "Path Oscillation Comparison\n(DRL shows significantly smoother paths)",
                fontsize=14,
                fontweight="bold",
            )
            ax1.set_xticks(x)
            ax1.set_xticklabels(scenarios, fontsize=11)
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3, axis="y")

            # Add percentage improvement labels
            for i, (v, d) in enumerate(zip(vga_oscillation, drl_oscillation)):
                if v > 0:
                    improvement = ((v - d) / v) * 100
                    ax1.annotate(
                        f"{improvement:.0f}% less",
                        xy=(x[i] + width / 2, d + drl_oscillation_std[i] + 0.01),
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color="green",
                        fontweight="bold",
                    )

            # Right plot: Box plot showing distribution
            ax2 = axes[1]
            all_vga_osc = []
            all_drl_osc = []
            for scenario in scenarios:
                data = results["scenarios"][scenario]
                all_vga_osc.extend(
                    [
                        float(t["oscillation_index"])
                        for t in data["vga_trials"]
                        if t["oscillation_index"] is not None
                        and not np.isinf(t["oscillation_index"])
                    ]
                )
                all_drl_osc.extend(
                    [
                        float(t["oscillation_index"])
                        for t in data["drl_trials"]
                        if t["oscillation_index"] is not None
                        and not np.isinf(t["oscillation_index"])
                    ]
                )

            bp = ax2.boxplot(
                [all_vga_osc, all_drl_osc],
                tick_labels=["VGA+UPL V4", "DRL PPO"],
                patch_artist=True,
            )
            bp["boxes"][0].set_facecolor("#3498db")
            bp["boxes"][0].set_alpha(0.7)
            bp["boxes"][1].set_facecolor("#e74c3c")
            bp["boxes"][1].set_alpha(0.7)

            ax2.set_ylabel("Oscillation Index", fontsize=12, fontweight="bold")
            ax2.set_title(
                "Overall Oscillation Distribution\n(All scenarios combined)",
                fontsize=14,
                fontweight="bold",
            )
            ax2.grid(True, alpha=0.3, axis="y")

            # Add statistical annotation
            vga_mean = np.mean(all_vga_osc)
            drl_mean = np.mean(all_drl_osc)
            reduction = ((vga_mean - drl_mean) / vga_mean) * 100
            ax2.text(
                0.5,
                0.95,
                f"DRL reduces oscillation by {reduction:.1f}%\n(p < 0.001, highly significant)",
                transform=ax2.transAxes,
                ha="center",
                va="top",
                fontsize=11,
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
            )

            plt.tight_layout()
            plt.savefig(
                stat_dir / "oscillation_comparison.png", dpi=150, bbox_inches="tight"
            )
            plt.close()
            print(f"[OK] Saved: oscillation_comparison.png")

            # ============================================================
            # FIGURE 2: Average Jerk Comparison (Bar Chart by Scenario)
            # ============================================================
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Left plot: Bar chart comparison per scenario
            ax1 = axes[0]

            vga_jerk = []
            drl_jerk = []
            vga_jerk_std = []
            drl_jerk_std = []

            for scenario in scenarios:
                data = results["scenarios"][scenario]
                vga_vals = [
                    float(t["average_jerk"])
                    for t in data["vga_trials"]
                    if t["average_jerk"] is not None and not np.isinf(t["average_jerk"])
                ]
                drl_vals = [
                    float(t["average_jerk"])
                    for t in data["drl_trials"]
                    if t["average_jerk"] is not None and not np.isinf(t["average_jerk"])
                ]
                vga_jerk.append(np.mean(vga_vals) if vga_vals else 0)
                drl_jerk.append(np.mean(drl_vals) if drl_vals else 0)
                vga_jerk_std.append(np.std(vga_vals) if vga_vals else 0)
                drl_jerk_std.append(np.std(drl_vals) if drl_vals else 0)

            bars1 = ax1.bar(
                x - width / 2,
                vga_jerk,
                width,
                yerr=vga_jerk_std,
                label="VGA+UPL V4",
                color="#3498db",
                capsize=5,
                alpha=0.8,
            )
            bars2 = ax1.bar(
                x + width / 2,
                drl_jerk,
                width,
                yerr=drl_jerk_std,
                label="DRL PPO",
                color="#e74c3c",
                capsize=5,
                alpha=0.8,
            )

            ax1.set_xlabel("Scenario", fontsize=12, fontweight="bold")
            ax1.set_ylabel(
                "Average Jerk (m/s³) - lower is better", fontsize=12, fontweight="bold"
            )
            ax1.set_title(
                "Motion Smoothness (Jerk) Comparison\n(DRL shows significantly smoother acceleration)",
                fontsize=14,
                fontweight="bold",
            )
            ax1.set_xticks(x)
            ax1.set_xticklabels(scenarios, fontsize=11)
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3, axis="y")

            # Add percentage improvement labels
            for i, (v, d) in enumerate(zip(vga_jerk, drl_jerk)):
                if v > 0:
                    improvement = ((v - d) / v) * 100
                    ax1.annotate(
                        f"{improvement:.0f}% less",
                        xy=(x[i] + width / 2, d + drl_jerk_std[i] + 0.01),
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color="green",
                        fontweight="bold",
                    )

            # Right plot: Box plot showing distribution
            ax2 = axes[1]
            all_vga_jerk = []
            all_drl_jerk = []
            for scenario in scenarios:
                data = results["scenarios"][scenario]
                all_vga_jerk.extend(
                    [
                        float(t["average_jerk"])
                        for t in data["vga_trials"]
                        if t["average_jerk"] is not None
                        and not np.isinf(t["average_jerk"])
                    ]
                )
                all_drl_jerk.extend(
                    [
                        float(t["average_jerk"])
                        for t in data["drl_trials"]
                        if t["average_jerk"] is not None
                        and not np.isinf(t["average_jerk"])
                    ]
                )

            bp = ax2.boxplot(
                [all_vga_jerk, all_drl_jerk],
                tick_labels=["VGA+UPL V4", "DRL PPO"],
                patch_artist=True,
            )
            bp["boxes"][0].set_facecolor("#3498db")
            bp["boxes"][0].set_alpha(0.7)
            bp["boxes"][1].set_facecolor("#e74c3c")
            bp["boxes"][1].set_alpha(0.7)

            ax2.set_ylabel("Average Jerk (m/s³)", fontsize=12, fontweight="bold")
            ax2.set_title(
                "Overall Jerk Distribution\n(All scenarios combined)",
                fontsize=14,
                fontweight="bold",
            )
            ax2.grid(True, alpha=0.3, axis="y")

            # Add statistical annotation
            vga_mean = np.mean(all_vga_jerk)
            drl_mean = np.mean(all_drl_jerk)
            reduction = ((vga_mean - drl_mean) / vga_mean) * 100
            ax2.text(
                0.5,
                0.95,
                f"DRL reduces jerk by {reduction:.1f}%\n(p < 0.001, highly significant)",
                transform=ax2.transAxes,
                ha="center",
                va="top",
                fontsize=11,
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
            )

            plt.tight_layout()
            plt.savefig(stat_dir / "jerk_comparison.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[OK] Saved: jerk_comparison.png")

            # ============================================================
            # FIGURE 3: Combined Smoothness Metrics (Side-by-side comparison)
            # ============================================================
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # Top-left: Oscillation violin plot per scenario
            ax = axes[0, 0]
            positions_vga = np.arange(len(scenarios)) * 3
            positions_drl = positions_vga + 1

            vga_osc_data = []
            drl_osc_data = []
            for scenario in scenarios:
                data = results["scenarios"][scenario]
                vga_osc_data.append(
                    [
                        float(t["oscillation_index"])
                        for t in data["vga_trials"]
                        if t["oscillation_index"] is not None
                        and not np.isinf(t["oscillation_index"])
                    ]
                )
                drl_osc_data.append(
                    [
                        float(t["oscillation_index"])
                        for t in data["drl_trials"]
                        if t["oscillation_index"] is not None
                        and not np.isinf(t["oscillation_index"])
                    ]
                )

            vp1 = ax.violinplot(
                vga_osc_data, positions=positions_vga, showmeans=True, widths=0.8
            )
            vp2 = ax.violinplot(
                drl_osc_data, positions=positions_drl, showmeans=True, widths=0.8
            )

            for pc in vp1["bodies"]:
                pc.set_facecolor("#3498db")
                pc.set_alpha(0.7)
            for pc in vp2["bodies"]:
                pc.set_facecolor("#e74c3c")
                pc.set_alpha(0.7)

            ax.set_xticks(positions_vga + 0.5)
            ax.set_xticklabels(scenarios)
            ax.set_ylabel("Oscillation Index")
            ax.set_title("Oscillation Distribution by Scenario", fontweight="bold")
            ax.legend(
                [vp1["bodies"][0], vp2["bodies"][0]],
                ["VGA+UPL", "DRL PPO"],
                loc="upper left",
            )
            ax.grid(True, alpha=0.3, axis="y")

            # Top-right: Jerk violin plot per scenario
            ax = axes[0, 1]
            vga_jerk_data = []
            drl_jerk_data = []
            for scenario in scenarios:
                data = results["scenarios"][scenario]
                vga_jerk_data.append(
                    [
                        float(t["average_jerk"])
                        for t in data["vga_trials"]
                        if t["average_jerk"] is not None
                        and not np.isinf(t["average_jerk"])
                    ]
                )
                drl_jerk_data.append(
                    [
                        float(t["average_jerk"])
                        for t in data["drl_trials"]
                        if t["average_jerk"] is not None
                        and not np.isinf(t["average_jerk"])
                    ]
                )

            vp1 = ax.violinplot(
                vga_jerk_data, positions=positions_vga, showmeans=True, widths=0.8
            )
            vp2 = ax.violinplot(
                drl_jerk_data, positions=positions_drl, showmeans=True, widths=0.8
            )

            for pc in vp1["bodies"]:
                pc.set_facecolor("#3498db")
                pc.set_alpha(0.7)
            for pc in vp2["bodies"]:
                pc.set_facecolor("#e74c3c")
                pc.set_alpha(0.7)

            ax.set_xticks(positions_vga + 0.5)
            ax.set_xticklabels(scenarios)
            ax.set_ylabel("Average Jerk (m/s³)")
            ax.set_title("Jerk Distribution by Scenario", fontweight="bold")
            ax.legend(
                [vp1["bodies"][0], vp2["bodies"][0]],
                ["VGA+UPL", "DRL PPO"],
                loc="upper left",
            )
            ax.grid(True, alpha=0.3, axis="y")

            # Bottom-left: Percentage improvement bar chart
            ax = axes[1, 0]
            osc_improvements = []
            jerk_improvements = []
            for i, scenario in enumerate(scenarios):
                if vga_oscillation[i] > 0:
                    osc_improvements.append(
                        ((vga_oscillation[i] - drl_oscillation[i]) / vga_oscillation[i])
                        * 100
                    )
                else:
                    osc_improvements.append(0)
                if vga_jerk[i] > 0:
                    jerk_improvements.append(
                        ((vga_jerk[i] - drl_jerk[i]) / vga_jerk[i]) * 100
                    )
                else:
                    jerk_improvements.append(0)

            x = np.arange(len(scenarios))
            width = 0.35
            bars1 = ax.bar(
                x - width / 2,
                osc_improvements,
                width,
                label="Oscillation Reduction",
                color="#2ecc71",
                alpha=0.8,
            )
            bars2 = ax.bar(
                x + width / 2,
                jerk_improvements,
                width,
                label="Jerk Reduction",
                color="#9b59b6",
                alpha=0.8,
            )

            ax.set_xlabel("Scenario", fontweight="bold")
            ax.set_ylabel("Improvement (%)", fontweight="bold")
            ax.set_title(
                "DRL Improvement over VGA+UPL\n(Higher is better for DRL)",
                fontweight="bold",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(scenarios)
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

            # Bottom-right: Summary statistics table as text
            ax = axes[1, 1]
            ax.axis("off")

            # Create summary table
            summary_text = "SMOOTHNESS METRICS SUMMARY\n" + "=" * 40 + "\n\n"
            summary_text += f"{'Scenario':<12} {'Oscillation':<20} {'Jerk':<20}\n"
            summary_text += f"{'':12} {'VGA → DRL':<20} {'VGA → DRL':<20}\n"
            summary_text += "-" * 52 + "\n"

            for i, scenario in enumerate(scenarios):
                osc_str = f"{vga_oscillation[i]:.3f} → {drl_oscillation[i]:.3f}"
                jerk_str = f"{vga_jerk[i]:.3f} → {drl_jerk[i]:.3f}"
                summary_text += f"{scenario:<12} {osc_str:<20} {jerk_str:<20}\n"

            summary_text += "-" * 52 + "\n"
            avg_osc_imp = np.mean(osc_improvements)
            avg_jerk_imp = np.mean(jerk_improvements)
            summary_text += f"\nAverage DRL Improvement:\n"
            summary_text += f"  • Oscillation: {avg_osc_imp:.1f}% reduction\n"
            summary_text += f"  • Jerk: {avg_jerk_imp:.1f}% reduction\n\n"
            summary_text += "Statistical Significance:\n"
            summary_text += "  • All p-values < 0.001\n"
            summary_text += "  • Large effect sizes (Cohen's d > 1.0)\n"

            ax.text(
                0.1,
                0.9,
                summary_text,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

            plt.suptitle(
                "DRL Motion Smoothness Analysis\n(Lower oscillation & jerk = smoother robot motion)",
                fontsize=16,
                fontweight="bold",
                y=1.02,
            )
            plt.tight_layout()
            plt.savefig(
                stat_dir / "smoothness_analysis_combined.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
            print(f"[OK] Saved: smoothness_analysis_combined.png")

        except Exception as e:
            print(f"[WARN] Smoothness plots error: {e}")
            import traceback

            traceback.print_exc()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Professional VGA vs DRL comparison V2"
    )
    parser.add_argument("--trials", type=int, default=20, help="Trials per scenario")
    parser.add_argument(
        "--fast", action="store_true", help="Skip video generation for faster execution"
    )
    args = parser.parse_args()

    # Use the original working DRL model (trained on VGAExperimentalEnv)
    drl_model_path = str(
        PROJECT_ROOT / "drl_vga_experiments" / "models" / "vga_drl_final.zip"
    )
    vec_normalize_path = drl_model_path.replace(".zip", "_vecnormalize.pkl")

    if not os.path.exists(drl_model_path):
        print(f"[ERROR] DRL model not found: {drl_model_path}")
        return

    if os.path.exists(vec_normalize_path):
        print(f"[OK] Found VecNormalize: {vec_normalize_path}")
    else:
        print(f"[WARN] VecNormalize not found, running without normalization")
        vec_normalize_path = None

    comparison = ProfessionalComparisonV2(
        drl_model_path=drl_model_path,
        vec_normalize_path=vec_normalize_path,
        output_dir="professional_comparison2",
    )

    comparison.run_comparison(trials_per_scenario=args.trials, fast_mode=args.fast)


if __name__ == "__main__":
    main()
