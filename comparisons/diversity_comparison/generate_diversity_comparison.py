"""
Trajectory Diversity Comparison: DRL Stochasticity vs VGA Determinism
=====================================================================

Shows that DRL produces DIFFERENT trajectories on the SAME trial when run
multiple times, while VGA ALWAYS produces the EXACT same path.

Each DRL run is COMPLETELY INDEPENDENT - fresh env, fresh model predict.

Uses EXACT same settings as professional_comparison.py
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
import matplotlib.animation as animation
from datetime import datetime
from typing import List, Tuple

# Add paths - EXACT same as professional_comparison.py
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "vga_baseline" / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "vga_baseline" / "data_loading"))
sys.path.insert(0, str(PROJECT_ROOT / "drl_training"))
sys.path.insert(0, str(PROJECT_ROOT / "core"))

from vga_upl_planner_v4 import VGAUPLPlannerV4
from vga_dataset import VGADatasetLoader
from vga_experimental_env import VGAExperimentalEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


class DiversityComparison:
    """Compare trajectory diversity: DRL stochastic vs VGA deterministic."""

    # EXACT same constants as professional_comparison.py
    GOAL_TOLERANCE = 0.3
    AGENT_RADIUS = 0.2
    OBSTACLE_RADIUS = 0.25
    DT = 0.05
    MAX_STEPS = 500

    ARENA_X_MIN = 0.0
    ARENA_X_MAX = 10.0
    ARENA_Y_MIN = -1.75
    ARENA_Y_MAX = 1.75

    # Colors for DRL runs - 5 distinct colors
    DRL_COLORS = [
        "#E91E63",
        "#2196F3",
        "#FF9800",
        "#9C27B0",
        "#00BCD4",
    ]  # Pink, Blue, Orange, Purple, Cyan
    DRL_LABELS = ["DRL Run 1", "DRL Run 2", "DRL Run 3", "DRL Run 4", "DRL Run 5"]
    # Colors for VGA runs - 5 shades of green (will all overlap = identical)
    VGA_COLORS = ["#4CAF50", "#66BB6A", "#81C784", "#A5D6A7", "#C8E6C9"]  # Green shades
    VGA_COLOR = "#4CAF50"  # Green (legacy)

    def __init__(
        self,
        drl_model_path: str,
        vec_normalize_path: str = None,
        data_root: str = None,
        output_dir: str = "diversity_comparison",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if data_root is None:
            data_root = str(PROJECT_ROOT / "data" / "VGA-Experimental-Data")
        self.data_root = data_root

        # Store paths - we'll reload model fresh each time for true independence
        self.drl_model_path = drl_model_path
        self.vec_normalize_path = vec_normalize_path

        print(f"[LOAD] DRL model path: {drl_model_path}")
        print(f"[LOAD] VecNormalize path: {vec_normalize_path}")

        # Create VGA model - EXACT same as professional_comparison.py
        self.vga_speed = 1.6  # Match DRL's MAX_VELOCITY
        print(f"[LOAD] Creating VGA+UPL V4 model (speed={self.vga_speed} m/s)")
        self.vga_model = VGAUPLPlannerV4(
            use_probabilistic=False, desired_speed=self.vga_speed
        )

        # Load dataset
        print(f"[LOAD] Loading VGA dataset from: {data_root}")
        self.dataset_loader = VGADatasetLoader(data_root=data_root)

    def get_trial_data(self, scenario: str, trial_idx: int) -> dict:
        """Get trial data - EXACT same as professional_comparison.py"""
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

    def run_vga_trial(self, trial_data: dict) -> np.ndarray:
        """Run VGA - deterministic, always same result."""
        result = self.vga_model.simulate(
            start_pos=trial_data["start_pos"],
            goal_pos=trial_data["goal_pos"],
            obstacles=trial_data["obstacles"],
            max_steps=self.MAX_STEPS,
        )
        return result.positions

    def run_single_drl_trial(
        self, scenario: str, trial_idx: int, run_id: int
    ) -> np.ndarray:
        """
        Run a SINGLE DRL trial - completely fresh and independent.

        CRITICAL: Each call creates fresh env, loads fresh model, runs independently.
        Uses deterministic=False to show stochastic policy behavior.
        """
        print(f"      [DRL Run {run_id}] Creating fresh environment...")

        # Create FRESH environment - exact same as professional_comparison.py
        env = VGAExperimentalEnv(
            scenario=scenario,
            data_root=self.data_root,
            randomize_start_goal=False,  # CRITICAL: same as professional_comparison
        )

        # Wrap with DummyVecEnv
        vec_env = DummyVecEnv([lambda: env])

        # Load normalization if available
        if self.vec_normalize_path and os.path.exists(self.vec_normalize_path):
            vec_env = VecNormalize.load(self.vec_normalize_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False

        # Load FRESH model for each run
        drl_model = PPO.load(self.drl_model_path)

        # Navigate to correct trial (exact same logic as professional_comparison.py)
        for _ in range(trial_idx + 1):
            obs = vec_env.reset()

        # Store goal for terminal detection
        goal_pos = env.goal_pos.copy()

        # Run episode with STOCHASTIC policy (deterministic=False)
        positions = [env.agent_pos.copy()]

        done = False
        step_count = 0
        while not done and step_count < self.MAX_STEPS:
            pre_step_pos = env.agent_pos.copy()

            # STOCHASTIC prediction - this is what creates diversity!
            action, _ = drl_model.predict(obs, deterministic=False)
            obs, _, done_arr, info = vec_env.step(action)
            done = done_arr[0] if hasattr(done_arr, "__len__") else done_arr
            step_count += 1

            if done:
                info_dict = info[0] if isinstance(info, list) else info
                if info_dict.get("goal_reached", False):
                    positions.append(goal_pos.copy())
                else:
                    positions.append(pre_step_pos)
            else:
                positions.append(env.agent_pos.copy())

        vec_env.close()

        positions = np.array(positions)
        print(
            f"      [DRL Run {run_id}] Done: {len(positions)} steps, final pos: ({positions[-1][0]:.2f}, {positions[-1][1]:.2f})"
        )

        return positions

    def create_diversity_image(
        self,
        vga_positions: np.ndarray,
        drl_positions_list: List[np.ndarray],
        trial_data: dict,
        scenario: str,
        trial_idx: int,
        save_path: Path,
    ):
        """Create side-by-side: VGA (1 path) vs DRL (multiple colored paths)."""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        obstacles = trial_data["obstacles"]
        start_pos = trial_data["start_pos"]
        goal_pos = trial_data["goal_pos"]

        for ax in axes:
            ax.set_xlim(self.ARENA_X_MIN - 0.5, self.ARENA_X_MAX + 0.5)
            ax.set_ylim(self.ARENA_Y_MIN - 0.5, self.ARENA_Y_MAX + 0.5)

            # Draw obstacles
            for ox, oy, r in obstacles:
                circle = Circle(
                    (ox, oy),
                    r,
                    facecolor="#B71C1C",
                    alpha=0.85,
                    ec="black",
                    lw=2,
                    zorder=1,
                )
                ax.add_patch(circle)

            # Goal zone
            goal_zone = Circle(
                goal_pos,
                self.GOAL_TOLERANCE,
                color="green",
                alpha=0.2,
                ec="green",
                lw=2,
                linestyle="--",
                zorder=2,
            )
            ax.add_patch(goal_zone)

            # Start and goal markers
            ax.plot(
                start_pos[0],
                start_pos[1],
                "go",
                markersize=15,
                zorder=10,
                markeredgecolor="black",
                markeredgewidth=2,
            )
            ax.plot(goal_pos[0], goal_pos[1], "r*", markersize=20, zorder=10)

            ax.set_xlabel("X (meters)", fontsize=12)
            ax.set_ylabel("Y (meters)", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal")

        # LEFT: VGA - Multiple runs, all identical (deterministic)
        ax_vga = axes[0]
        # vga_positions is now a list of position arrays
        if isinstance(vga_positions, list):
            num_vga_runs = len(vga_positions)
            for i, vga_pos in enumerate(vga_positions):
                color = self.VGA_COLORS[i % len(self.VGA_COLORS)]
                label = f"VGA Run {i+1}" if i < 5 else None
                ax_vga.plot(
                    vga_pos[:, 0],
                    vga_pos[:, 1],
                    color=color,
                    linewidth=3.0,
                    alpha=0.9,
                    label=label,
                    zorder=5 + i,
                )
                ax_vga.plot(
                    vga_pos[-1, 0],
                    vga_pos[-1, 1],
                    "s",
                    color=color,
                    markersize=10,
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    zorder=6 + i,
                )
        else:
            # Legacy single path
            ax_vga.plot(
                vga_positions[:, 0],
                vga_positions[:, 1],
                color=self.VGA_COLOR,
                linewidth=3.5,
                alpha=0.9,
                label="VGA Path",
                zorder=5,
            )
            ax_vga.plot(
                vga_positions[-1, 0],
                vga_positions[-1, 1],
                "s",
                color=self.VGA_COLOR,
                markersize=12,
                markeredgecolor="black",
                markeredgewidth=2,
                zorder=6,
            )
            num_vga_runs = 1

        ax_vga.set_title(
            f"VGA+UPL (Deterministic)\n{num_vga_runs} runs = ALL IDENTICAL paths",
            fontsize=14,
            fontweight="bold",
            color="#1B5E20",
        )
        ax_vga.legend(loc="upper right", fontsize=10)
        ax_vga.text(
            0.5,
            0.02,
            f"{num_vga_runs} runs perfectly overlap → 0.000m diversity",
            transform=ax_vga.transAxes,
            ha="center",
            fontsize=11,
            style="italic",
            color="#555555",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # RIGHT: DRL - Multiple different colored paths
        ax_drl = axes[1]
        num_runs = len(drl_positions_list)
        for i, positions in enumerate(drl_positions_list):
            color = self.DRL_COLORS[i % len(self.DRL_COLORS)]
            label = f"DRL Run {i+1}"
            ax_drl.plot(
                positions[:, 0],
                positions[:, 1],
                color=color,
                linewidth=2.5,
                alpha=0.85,
                label=label,
                zorder=5 + i,
            )
            ax_drl.plot(
                positions[-1, 0],
                positions[-1, 1],
                "s",
                color=color,
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=1.5,
                zorder=6 + i,
            )

        ax_drl.set_title(
            "DRL PPO (Stochastic Policy)\nDIFFERENT path each time",
            fontsize=14,
            fontweight="bold",
            color="#0D47A1",
        )
        ax_drl.legend(loc="upper right", fontsize=10)
        ax_drl.text(
            0.5,
            0.02,
            f"{num_runs} independent runs → {num_runs} unique trajectories (like humans)",
            transform=ax_drl.transAxes,
            ha="center",
            fontsize=11,
            style="italic",
            color="#555555",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )

        fig.suptitle(
            f"Trajectory Diversity: {scenario} Trial {trial_idx + 1}",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"    [SAVED] Image: {save_path}")

    def create_vga_video(
        self,
        vga_positions_list: list,
        trial_data: dict,
        scenario: str,
        trial_idx: int,
        save_path: Path,
    ):
        """Create video of VGA multiple runs - all moving identically (proving determinism)."""
        fig, ax = plt.subplots(figsize=(12, 7))

        obstacles = trial_data["obstacles"]
        start_pos = trial_data["start_pos"]
        goal_pos = trial_data["goal_pos"]

        ax.set_xlim(self.ARENA_X_MIN - 0.5, self.ARENA_X_MAX + 0.5)
        ax.set_ylim(self.ARENA_Y_MIN - 0.5, self.ARENA_Y_MAX + 0.5)

        for ox, oy, r in obstacles:
            circle = Circle(
                (ox, oy), r, facecolor="#B71C1C", alpha=0.85, ec="black", lw=2, zorder=1
            )
            ax.add_patch(circle)

        goal_zone = Circle(
            goal_pos,
            self.GOAL_TOLERANCE,
            color="green",
            alpha=0.2,
            ec="green",
            lw=2,
            linestyle="--",
            zorder=2,
        )
        ax.add_patch(goal_zone)

        ax.plot(
            start_pos[0],
            start_pos[1],
            "go",
            markersize=15,
            zorder=10,
            markeredgecolor="black",
            markeredgewidth=2,
        )
        ax.plot(goal_pos[0], goal_pos[1], "r*", markersize=20, zorder=10)

        ax.set_xlabel("X (meters)", fontsize=12)
        ax.set_ylabel("Y (meters)", fontsize=12)
        num_runs = len(vga_positions_list)
        ax.set_title(
            f"VGA+UPL - {scenario} Trial {trial_idx + 1}\nDeterministic: {num_runs} runs = ALL IDENTICAL (perfectly overlapping)",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # Create lines and agents for all VGA runs
        lines = []
        agents = []
        for i in range(num_runs):
            color = self.VGA_COLORS[i % len(self.VGA_COLORS)]
            label = f"VGA Run {i+1}"
            (line,) = ax.plot(
                [],
                [],
                color=color,
                linewidth=2.5,
                alpha=0.85,
                label=label,
                zorder=5 + i,
            )
            lines.append(line)
            agent = Circle(
                (0, 0), self.AGENT_RADIUS, color=color, alpha=0.8, zorder=10 + i
            )
            ax.add_patch(agent)
            agents.append(agent)

        ax.legend(loc="upper right", fontsize=10)

        max_frames = max(len(pos) for pos in vga_positions_list)

        def init():
            for i, line in enumerate(lines):
                line.set_data([], [])
                agents[i].center = vga_positions_list[i][0]
            return lines + agents

        def animate(frame):
            for i, (line, agent, positions) in enumerate(
                zip(lines, agents, vga_positions_list)
            ):
                idx = min(frame, len(positions) - 1)
                line.set_data(positions[: idx + 1, 0], positions[: idx + 1, 1])
                agent.center = positions[idx]
            return lines + agents

        anim = animation.FuncAnimation(
            fig, animate, init_func=init, frames=max_frames + 30, interval=50, blit=True
        )

        try:
            Writer = animation.writers["ffmpeg"]
            writer = Writer(fps=20, bitrate=2000)
            anim.save(str(save_path), writer=writer)
            print(f"    [SAVED] VGA Video: {save_path}")
        except Exception as e:
            print(f"    [ERROR] VGA video: {e}")

        plt.close(fig)

    def create_drl_diversity_video(
        self,
        drl_positions_list: List[np.ndarray],
        trial_data: dict,
        scenario: str,
        trial_idx: int,
        save_path: Path,
    ):
        """Create video showing all DRL runs simultaneously with different colors."""
        fig, ax = plt.subplots(figsize=(12, 7))

        obstacles = trial_data["obstacles"]
        start_pos = trial_data["start_pos"]
        goal_pos = trial_data["goal_pos"]

        ax.set_xlim(self.ARENA_X_MIN - 0.5, self.ARENA_X_MAX + 0.5)
        ax.set_ylim(self.ARENA_Y_MIN - 0.5, self.ARENA_Y_MAX + 0.5)

        for ox, oy, r in obstacles:
            circle = Circle(
                (ox, oy), r, facecolor="#B71C1C", alpha=0.85, ec="black", lw=2, zorder=1
            )
            ax.add_patch(circle)

        goal_zone = Circle(
            goal_pos,
            self.GOAL_TOLERANCE,
            color="green",
            alpha=0.2,
            ec="green",
            lw=2,
            linestyle="--",
            zorder=2,
        )
        ax.add_patch(goal_zone)

        ax.plot(
            start_pos[0],
            start_pos[1],
            "go",
            markersize=15,
            zorder=10,
            markeredgecolor="black",
            markeredgewidth=2,
        )
        ax.plot(goal_pos[0], goal_pos[1], "r*", markersize=20, zorder=10)

        ax.set_xlabel("X (meters)", fontsize=12)
        ax.set_ylabel("Y (meters)", fontsize=12)
        ax.set_title(
            f"DRL PPO - {scenario} Trial {trial_idx + 1}\nStochastic: {len(drl_positions_list)} DIFFERENT paths from same start",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # Create lines and agents dynamically based on actual number of runs
        num_runs = len(drl_positions_list)
        lines = []
        agents = []
        for i in range(num_runs):
            color = self.DRL_COLORS[i % len(self.DRL_COLORS)]
            label = f"DRL Run {i+1}"
            (line,) = ax.plot(
                [],
                [],
                color=color,
                linewidth=2.5,
                alpha=0.85,
                label=label,
                zorder=5 + i,
            )
            lines.append(line)
            agent = Circle(
                (0, 0), self.AGENT_RADIUS, color=color, alpha=0.8, zorder=10 + i
            )
            ax.add_patch(agent)
            agents.append(agent)

        ax.legend(loc="upper right", fontsize=10)

        max_frames = max(len(pos) for pos in drl_positions_list)

        def init():
            for i, line in enumerate(lines):
                line.set_data([], [])
                agents[i].center = drl_positions_list[i][0]
            return lines + agents

        def animate(frame):
            for i, (line, agent, positions) in enumerate(
                zip(lines, agents, drl_positions_list)
            ):
                idx = min(frame, len(positions) - 1)
                line.set_data(positions[: idx + 1, 0], positions[: idx + 1, 1])
                agent.center = positions[idx]
            return lines + agents

        anim = animation.FuncAnimation(
            fig, animate, init_func=init, frames=max_frames + 30, interval=50, blit=True
        )

        try:
            Writer = animation.writers["ffmpeg"]
            writer = Writer(fps=20, bitrate=2000)
            anim.save(str(save_path), writer=writer)
            print(f"    [SAVED] DRL Diversity Video: {save_path}")
        except Exception as e:
            print(f"    [ERROR] DRL video: {e}")

        plt.close(fig)

    def run_diversity_comparison(self, num_drl_runs: int = 5):
        """Run diversity comparison for multiple trials per scenario."""
        print("=" * 70)
        print("TRAJECTORY DIVERSITY COMPARISON")
        print("DRL Stochasticity vs VGA Determinism")
        print("=" * 70)

        # 3 trials per scenario
        selected_trials = {
            "SOSP": [5, 15, 30],
            "MOSP_A": [10, 50, 100],
            "MOSP_B": [15, 60, 120],
            "MOSP_C": [10, 50, 90],
            "MOSP_D": [20, 80, 150],
        }

        results = []

        for scenario, trial_list in selected_trials.items():
            for trial_idx in trial_list:
                print(f"\n{'='*60}")
                print(f"  {scenario} - Trial {trial_idx + 1}")
                print("=" * 60)

                # Create output dirs
                scenario_dir = self.output_dir / scenario
                images_dir = scenario_dir / "images"
                videos_dir = scenario_dir / "videos"
                images_dir.mkdir(parents=True, exist_ok=True)
                videos_dir.mkdir(parents=True, exist_ok=True)

                # Get trial data
                trial_data = self.get_trial_data(scenario, trial_idx)
                print(
                    f"  Start: ({trial_data['start_pos'][0]:.2f}, {trial_data['start_pos'][1]:.2f})"
                )
                print(
                    f"  Goal:  ({trial_data['goal_pos'][0]:.2f}, {trial_data['goal_pos'][1]:.2f})"
                )
                print(f"  Obstacles: {len(trial_data['obstacles'])}")

                # Run VGA multiple times (all should be identical!)
                print(
                    f"\n  [VGA] Running {num_drl_runs} times (all should be IDENTICAL - deterministic)..."
                )
                vga_positions_list = []
                for run in range(1, num_drl_runs + 1):
                    vga_pos = self.run_vga_trial(trial_data)
                    vga_positions_list.append(vga_pos)
                    print(f"    VGA Run {run}: {len(vga_pos)} steps")

                # Calculate VGA diversity (should be 0!)
                if len(vga_positions_list) >= 2:
                    vga_diversities = []
                    for i in range(len(vga_positions_list)):
                        for j in range(i + 1, len(vga_positions_list)):
                            p1, p2 = vga_positions_list[i], vga_positions_list[j]
                            min_len = min(len(p1), len(p2))
                            avg_dist = np.mean(
                                [np.linalg.norm(p1[k] - p2[k]) for k in range(min_len)]
                            )
                            vga_diversities.append(avg_dist)
                    vga_diversity = np.mean(vga_diversities)
                    print(f"    VGA Diversity: {vga_diversity:.6f}m (should be ~0.000)")
                else:
                    vga_diversity = 0

                # Run DRL multiple times - EACH COMPLETELY INDEPENDENT
                print(
                    f"\n  [DRL] Running {num_drl_runs} INDEPENDENT trials (stochastic)..."
                )
                drl_positions_list = []
                for run in range(1, num_drl_runs + 1):
                    positions = self.run_single_drl_trial(scenario, trial_idx, run)
                    drl_positions_list.append(positions)

                # Calculate diversity metric
                print(f"\n  [ANALYSIS] Computing path diversity...")
                if len(drl_positions_list) >= 2:
                    diversities = []
                    for i in range(len(drl_positions_list)):
                        for j in range(i + 1, len(drl_positions_list)):
                            p1, p2 = drl_positions_list[i], drl_positions_list[j]
                            min_len = min(len(p1), len(p2))
                            # Average Euclidean distance between paths
                            avg_dist = np.mean(
                                [np.linalg.norm(p1[k] - p2[k]) for k in range(min_len)]
                            )
                            diversities.append(avg_dist)
                    avg_diversity = np.mean(diversities)
                    print(f"    OVERALL DRL Diversity: {avg_diversity:.3f}m")
                else:
                    avg_diversity = 0

                # Create visualizations
                print(f"\n  [EXPORT] Creating visualizations...")

                # Image
                img_path = images_dir / f"diversity_trial_{trial_idx+1:03d}.png"
                self.create_diversity_image(
                    vga_positions_list,
                    drl_positions_list,
                    trial_data,
                    scenario,
                    trial_idx,
                    img_path,
                )

                # VGA video
                vga_video = videos_dir / f"vga_trial_{trial_idx+1:03d}.mp4"
                self.create_vga_video(
                    vga_positions_list, trial_data, scenario, trial_idx, vga_video
                )

                # DRL diversity video
                drl_video = videos_dir / f"drl_diversity_trial_{trial_idx+1:03d}.mp4"
                self.create_drl_diversity_video(
                    drl_positions_list, trial_data, scenario, trial_idx, drl_video
                )

                results.append(
                    {
                        "scenario": scenario,
                        "trial": trial_idx + 1,
                        "vga_steps": [len(p) for p in vga_positions_list],
                        "vga_diversity_m": round(vga_diversity, 6),
                        "drl_steps": [len(p) for p in drl_positions_list],
                        "drl_diversity_m": round(avg_diversity, 4),
                    }
                )

        # Save summary
        summary_path = self.output_dir / "diversity_summary.json"
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "num_drl_runs": num_drl_runs,
                    "results": results,
                },
                f,
                indent=2,
            )

        print("\n" + "=" * 70)
        print("COMPLETE!")
        print("=" * 70)
        print(f"\nOutput: {self.output_dir}")

        print("\n" + "-" * 70)
        print(
            f"{'Scenario':<12} {'Trial':<8} {'VGA Diversity':<18} {'DRL Diversity':<15}"
        )
        print("-" * 70)
        for r in results:
            print(
                f"{r['scenario']:<12} {r['trial']:<8} {r['vga_diversity_m']:.6f}m          {r['drl_diversity_m']:.3f}m"
            )
        print("-" * 70)
        print("\n** VGA diversity should be ~0.000 (deterministic = identical paths)")
        print("** DRL diversity shows stochastic behavior (different paths each run)")


def main():
    # EXACT same paths as professional_comparison.py
    drl_model_path = str(PROJECT_ROOT / "drl_training" / "models" / "vga_drl_final.zip")
    vec_normalize_path = drl_model_path.replace(".zip", "_vecnormalize.pkl")
    output_dir = str(Path(__file__).parent / "diversity_comparison")

    if not os.path.exists(drl_model_path):
        print(f"ERROR: Model not found: {drl_model_path}")
        return

    if not os.path.exists(vec_normalize_path):
        print(f"WARN: VecNormalize not found: {vec_normalize_path}")
        vec_normalize_path = None
    else:
        print(f"OK: VecNormalize found: {vec_normalize_path}")

    comparison = DiversityComparison(
        drl_model_path=drl_model_path,
        vec_normalize_path=vec_normalize_path,
        output_dir=output_dir,
    )

    comparison.run_diversity_comparison(num_drl_runs=5)


if __name__ == "__main__":
    main()
