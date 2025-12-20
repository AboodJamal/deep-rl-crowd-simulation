"""
VGA DRL Evaluation - Evaluate trained model and generate videos/images
======================================================================

This script evaluates the trained DRL model on VGA experimental data and:
1. Computes success rates per scenario
2. Generates trajectory videos (MP4)
3. Generates trajectory images (PNG)
4. Compares with VGA+UPL baseline (if available)
"""

import os
import sys
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from vga_experimental_env import VGAExperimentalEnv


class VGAEvaluator:
    """Evaluate DRL model on VGA data and generate visualizations."""

    def __init__(
        self,
        model_path: str,
        data_root: str = r"D:\Abdullah Jamal\downloads\VGA-exp\Pedestrian-Experimental-Data",
        output_dir: str = "vga_training/evaluation",
        vec_normalize_path: str = None,
    ):
        self.model_path = model_path
        self.data_root = data_root
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "videos").mkdir(exist_ok=True)
        (self.output_dir / "trajectories").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)

        # Load model
        print(f"[OK] Loading model from: {model_path}")
        self.model = PPO.load(model_path)

        # Load VecNormalize stats if available
        self.vec_normalize_path = vec_normalize_path
        if vec_normalize_path and os.path.exists(vec_normalize_path):
            print(f"[OK] Found VecNormalize stats: {vec_normalize_path}")

        self.scenarios = ["SOSP", "MOSP_A", "MOSP_B", "MOSP_C", "MOSP_D"]
        self.results = {}

    def evaluate_scenario(
        self,
        scenario: str,
        num_trials: int = None,
        deterministic: bool = True,
        record_trajectories: bool = True,
    ) -> Dict:
        """Evaluate model on a specific scenario."""
        print(f"\n[EVAL] Evaluating {scenario}...")

        # Create environment
        env = VGAExperimentalEnv(
            scenario=scenario,
            data_root=self.data_root,
            randomize_start_goal=False,
        )

        if num_trials is None:
            num_trials = env.num_trials

        # Wrap for VecNormalize compatibility
        vec_env = DummyVecEnv([lambda: env])
        if self.vec_normalize_path and os.path.exists(self.vec_normalize_path):
            vec_env = VecNormalize.load(self.vec_normalize_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False

        results = {
            "scenario": scenario,
            "num_trials": num_trials,
            "successes": 0,
            "failures": 0,
            "collisions_total": 0,
            "steps_total": 0,
            "trajectories": [],
        }

        for trial_idx in range(min(num_trials, env.num_trials)):
            obs = vec_env.reset()
            trajectory = {
                "trial_idx": trial_idx,
                "positions": [env.agent_pos.copy()],
                "start": env.agent_pos.copy(),
                "goal": env.goal_pos.copy(),
                "obstacles": [(x, y, r) for x, y, r in env.obstacles],
                "actions": [],
                "success": False,
                "collisions": 0,
                "steps": 0,
            }

            done = False
            while not done:
                # Record position BEFORE step (critical for terminal step!)
                pre_step_pos = env.agent_pos.copy()

                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = vec_env.step(action)

                trajectory["actions"].append(action[0].copy())
                trajectory["steps"] += 1

                if done:
                    # IMPORTANT: After done=True, VecEnv auto-resets!
                    # env.agent_pos is now the NEXT trial's start position (WRONG!)
                    # terminal_observation contains NORMALIZED values (WRONG!)
                    # We must use pre_step_pos or estimate the final position

                    info = info[0]
                    trajectory["success"] = info.get("goal_reached", False)
                    trajectory["collisions"] = info.get("collisions", 0)

                    # For the final position:
                    # - If success: the agent reached the goal, so final pos is near goal
                    # - If failure: use pre_step_pos (last known valid position)
                    if trajectory["success"]:
                        # Agent successfully reached goal - place final marker at goal
                        trajectory["positions"].append(trajectory["goal"].copy())
                    else:
                        # Failure - use the position before the terminal step
                        trajectory["positions"].append(pre_step_pos.copy())

                    if trajectory["success"]:
                        results["successes"] += 1
                    else:
                        results["failures"] += 1

                    results["collisions_total"] += trajectory["collisions"]
                    results["steps_total"] += trajectory["steps"]
                else:
                    # Not done yet - record current position (env hasn't reset)
                    trajectory["positions"].append(env.agent_pos.copy())

            if record_trajectories:
                results["trajectories"].append(trajectory)

            # Progress
            if (trial_idx + 1) % 10 == 0:
                rate = results["successes"] / (trial_idx + 1) * 100
                print(
                    f"  Trial {trial_idx + 1}/{num_trials}: Success Rate = {rate:.1f}%"
                )

        vec_env.close()

        results["success_rate"] = results["successes"] / num_trials * 100
        results["avg_collisions"] = results["collisions_total"] / num_trials
        results["avg_steps"] = results["steps_total"] / num_trials

        print(
            f"[OK] {scenario}: {results['success_rate']:.1f}% success rate "
            f"({results['successes']}/{num_trials})"
        )

        return results

    def evaluate_all(self, num_trials_per_scenario: int = None) -> Dict:
        """Evaluate on all scenarios."""
        print("=" * 70)
        print("VGA DRL EVALUATION")
        print("=" * 70)

        all_results = {}
        total_successes = 0
        total_trials = 0

        for scenario in self.scenarios:
            results = self.evaluate_scenario(
                scenario,
                num_trials=num_trials_per_scenario,
                record_trajectories=True,
            )
            all_results[scenario] = results
            total_successes += results["successes"]
            total_trials += results["num_trials"]

        all_results["overall"] = {
            "total_trials": total_trials,
            "total_successes": total_successes,
            "success_rate": (
                total_successes / total_trials * 100 if total_trials > 0 else 0
            ),
        }

        self.results = all_results

        # Save results
        results_path = self.output_dir / "results" / "evaluation_results.json"
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._prepare_for_json(all_results)
        with open(results_path, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"\n[OK] Results saved: {results_path}")

        return all_results

    def _prepare_for_json(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    def generate_trajectory_image(
        self,
        trajectory: Dict,
        scenario: str,
        trial_idx: int,
        save_path: str = None,
    ) -> plt.Figure:
        """Generate a trajectory visualization image."""
        fig, ax = plt.subplots(figsize=(14, 6))

        # Arena
        arena_x = [0, 10, 10, 0, 0]
        arena_y = [-1.75, -1.75, 1.75, 1.75, -1.75]
        ax.plot(arena_x, arena_y, "k-", linewidth=2)
        ax.fill(arena_x, arena_y, color="#f0f0f0", alpha=0.3)

        # Obstacles
        for ox, oy, r in trajectory["obstacles"]:
            circle = Circle((ox, oy), r, color="gray", alpha=0.7)
            ax.add_patch(circle)
            # Add obstacle outline
            circle_outline = Circle(
                (ox, oy), r, fill=False, edgecolor="black", linewidth=1
            )
            ax.add_patch(circle_outline)

        # Trajectory
        positions = np.array(trajectory["positions"])

        # Color gradient from start (blue) to end (green/red)
        n_points = len(positions)
        if trajectory["success"]:
            colors = plt.cm.Blues(np.linspace(0.3, 1.0, n_points))
            end_color = "green"
        else:
            colors = plt.cm.Reds(np.linspace(0.3, 1.0, n_points))
            end_color = "red"

        # Plot trajectory as line segments with gradient
        for i in range(len(positions) - 1):
            ax.plot(
                positions[i : i + 2, 0],
                positions[i : i + 2, 1],
                color=colors[i],
                linewidth=2,
                alpha=0.8,
            )

        # Start position
        ax.plot(
            trajectory["start"][0],
            trajectory["start"][1],
            "bo",
            markersize=12,
            label="Start",
            zorder=5,
        )
        ax.annotate(
            "Start",
            trajectory["start"],
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
        )

        # Goal position
        goal_circle = Circle(trajectory["goal"], 0.3, color="green", alpha=0.3)
        ax.add_patch(goal_circle)
        ax.plot(
            trajectory["goal"][0],
            trajectory["goal"][1],
            "g*",
            markersize=15,
            label="Goal",
            zorder=5,
        )
        ax.annotate(
            "Goal",
            trajectory["goal"],
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
        )

        # End position
        end_pos = positions[-1]
        ax.plot(
            end_pos[0],
            end_pos[1],
            "o",
            color=end_color,
            markersize=10,
            label="End",
            zorder=5,
        )

        # Title and labels
        status = "SUCCESS ✓" if trajectory["success"] else "FAILED ✗"
        ax.set_title(
            f"{scenario} - Trial {trial_idx + 1} - {status}\n"
            f"Steps: {trajectory['steps']} | Collisions: {trajectory['collisions']}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("X (m)", fontsize=12)
        ax.set_ylabel("Y (m)", fontsize=12)
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-2.25, 2.25)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        return fig

    def generate_all_trajectory_images(self, max_per_scenario: int = 10):
        """Generate trajectory images for all scenarios."""
        print("\n[GEN] Generating trajectory images...")

        for scenario, results in self.results.items():
            if scenario == "overall":
                continue

            scenario_dir = self.output_dir / "trajectories" / scenario
            scenario_dir.mkdir(exist_ok=True)

            trajectories = results.get("trajectories", [])
            n_images = min(len(trajectories), max_per_scenario)

            for i, traj in enumerate(trajectories[:n_images]):
                save_path = scenario_dir / f"trial_{i+1:03d}.png"
                self.generate_trajectory_image(traj, scenario, i, str(save_path))

            print(f"  [OK] {scenario}: Generated {n_images} images")

        print(
            f"[OK] All trajectory images saved to: {self.output_dir / 'trajectories'}"
        )

    def generate_video(
        self,
        trajectory: Dict,
        scenario: str,
        trial_idx: int,
        save_path: str,
        fps: int = 20,
    ):
        """Generate a video of a single trajectory."""
        fig, ax = plt.subplots(figsize=(14, 6))

        positions = np.array(trajectory["positions"])

        def init():
            ax.clear()
            ax.set_xlim(-0.5, 10.5)
            ax.set_ylim(-2.25, 2.25)
            ax.set_aspect("equal")
            return []

        def animate(frame):
            ax.clear()

            # Arena
            arena_x = [0, 10, 10, 0, 0]
            arena_y = [-1.75, -1.75, 1.75, 1.75, -1.75]
            ax.plot(arena_x, arena_y, "k-", linewidth=2)
            ax.fill(arena_x, arena_y, color="#f0f0f0", alpha=0.3)

            # Obstacles
            for ox, oy, r in trajectory["obstacles"]:
                circle = Circle((ox, oy), r, color="gray", alpha=0.7)
                ax.add_patch(circle)

            # Goal
            goal_circle = Circle(trajectory["goal"], 0.3, color="green", alpha=0.3)
            ax.add_patch(goal_circle)
            ax.plot(trajectory["goal"][0], trajectory["goal"][1], "g*", markersize=15)

            # Trajectory up to current frame
            if frame > 0:
                ax.plot(
                    positions[: frame + 1, 0],
                    positions[: frame + 1, 1],
                    "b-",
                    linewidth=2,
                    alpha=0.5,
                )

            # Current agent position
            agent_pos = positions[frame]
            agent_circle = Circle(agent_pos, 0.2, color="blue", alpha=0.8)
            ax.add_patch(agent_circle)

            # Start marker
            ax.plot(
                trajectory["start"][0],
                trajectory["start"][1],
                "bo",
                markersize=8,
                alpha=0.5,
            )

            ax.set_xlim(-0.5, 10.5)
            ax.set_ylim(-2.25, 2.25)
            ax.set_aspect("equal")
            ax.set_title(
                f"{scenario} - Trial {trial_idx + 1} - Step {frame + 1}/{len(positions)}"
            )
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.grid(True, alpha=0.3)

            return []

        anim = animation.FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=len(positions),
            interval=1000 / fps,
            blit=False,
        )

        # Save video
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        try:
            anim.save(save_path, writer=writer)
        except Exception as e:
            print(f"  [WARN] FFmpeg not available, trying pillow: {e}")
            try:
                writer = animation.PillowWriter(fps=fps)
                gif_path = save_path.replace(".mp4", ".gif")
                anim.save(gif_path, writer=writer)
                print(f"  [OK] Saved as GIF: {gif_path}")
            except Exception as e2:
                print(f"  [ERROR] Could not save video: {e2}")

        plt.close(fig)

    def generate_all_videos(self, max_per_scenario: int = 3):
        """Generate videos for sample trajectories from each scenario."""
        print("\n[GEN] Generating trajectory videos...")

        for scenario, results in self.results.items():
            if scenario == "overall":
                continue

            scenario_dir = self.output_dir / "videos" / scenario
            scenario_dir.mkdir(exist_ok=True)

            trajectories = results.get("trajectories", [])

            # Get some successes and failures
            successes = [t for t in trajectories if t["success"]][:max_per_scenario]
            failures = [t for t in trajectories if not t["success"]][:max_per_scenario]

            for i, traj in enumerate(successes):
                save_path = str(scenario_dir / f"success_{i+1:02d}.mp4")
                self.generate_video(traj, scenario, traj["trial_idx"], save_path)

            for i, traj in enumerate(failures):
                save_path = str(scenario_dir / f"failure_{i+1:02d}.mp4")
                self.generate_video(traj, scenario, traj["trial_idx"], save_path)

            print(
                f"  [OK] {scenario}: Generated {len(successes)} success + {len(failures)} failure videos"
            )

        print(f"[OK] All videos saved to: {self.output_dir / 'videos'}")

    def generate_summary_plot(self, save_path: str = None):
        """Generate a summary plot comparing all scenarios."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        scenarios = [s for s in self.results.keys() if s != "overall"]
        success_rates = [self.results[s]["success_rate"] for s in scenarios]
        avg_steps = [self.results[s]["avg_steps"] for s in scenarios]
        avg_collisions = [self.results[s]["avg_collisions"] for s in scenarios]

        # Success rates
        colors = [
            "green" if r >= 90 else "orange" if r >= 70 else "red"
            for r in success_rates
        ]
        axes[0].bar(scenarios, success_rates, color=colors, edgecolor="black")
        axes[0].set_ylabel("Success Rate (%)")
        axes[0].set_title("Success Rate by Scenario")
        axes[0].set_ylim(0, 105)
        for i, v in enumerate(success_rates):
            axes[0].text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")

        # Average steps
        axes[1].bar(scenarios, avg_steps, color="steelblue", edgecolor="black")
        axes[1].set_ylabel("Average Steps")
        axes[1].set_title("Average Episode Length")
        for i, v in enumerate(avg_steps):
            axes[1].text(i, v + 5, f"{v:.0f}", ha="center")

        # Average collisions
        axes[2].bar(scenarios, avg_collisions, color="coral", edgecolor="black")
        axes[2].set_ylabel("Average Collisions")
        axes[2].set_title("Average Collisions per Episode")
        for i, v in enumerate(avg_collisions):
            axes[2].text(i, v + 0.1, f"{v:.2f}", ha="center")

        plt.suptitle("VGA DRL Evaluation Summary", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "results" / "summary_plot.png"

        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Summary plot saved: {save_path}")

    def print_results_table(self):
        """Print a formatted results table."""
        print("\n" + "=" * 70)
        print("VGA DRL EVALUATION RESULTS")
        print("=" * 70)
        print(
            f"{'Scenario':<12} {'Trials':>8} {'Success':>10} {'Rate':>8} {'Avg Steps':>10} {'Avg Coll':>10}"
        )
        print("-" * 70)

        for scenario in self.scenarios:
            if scenario not in self.results:
                continue
            r = self.results[scenario]
            print(
                f"{scenario:<12} {r['num_trials']:>8} {r['successes']:>10} "
                f"{r['success_rate']:>7.1f}% {r['avg_steps']:>10.1f} {r['avg_collisions']:>10.2f}"
            )

        print("-" * 70)
        overall = self.results.get("overall", {})
        print(
            f"{'TOTAL':<12} {overall.get('total_trials', 0):>8} "
            f"{overall.get('total_successes', 0):>10} "
            f"{overall.get('success_rate', 0):>7.1f}%"
        )
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate VGA DRL Model")
    parser.add_argument(
        "--model",
        type=str,
        default="vga_training/models/vga_drl_final.zip",
        help="Path to trained model",
    )
    parser.add_argument(
        "--vec-normalize", type=str, default=None, help="Path to VecNormalize stats"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="vga_training/evaluation",
        help="Output directory",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=r"D:\Abdullah Jamal\downloads\VGA-exp\Pedestrian-Experimental-Data",
        help="Path to VGA dataset",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Maximum trials per scenario (None = all)",
    )
    parser.add_argument(
        "--max-videos", type=int, default=3, help="Maximum videos per scenario"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=10,
        help="Maximum trajectory images per scenario",
    )
    parser.add_argument(
        "--no-videos", action="store_true", help="Skip video generation"
    )
    parser.add_argument(
        "--no-images", action="store_true", help="Skip image generation"
    )

    args = parser.parse_args()

    # Auto-detect VecNormalize path
    if args.vec_normalize is None:
        potential_path = args.model.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(potential_path):
            args.vec_normalize = potential_path

    # Create evaluator
    evaluator = VGAEvaluator(
        model_path=args.model,
        data_root=args.data_root,
        output_dir=args.output_dir,
        vec_normalize_path=args.vec_normalize,
    )

    # Run evaluation
    evaluator.evaluate_all(num_trials_per_scenario=args.max_trials)

    # Print results
    evaluator.print_results_table()

    # Generate visualizations
    evaluator.generate_summary_plot()

    if not args.no_images:
        evaluator.generate_all_trajectory_images(max_per_scenario=args.max_images)

    if not args.no_videos:
        evaluator.generate_all_videos(max_per_scenario=args.max_videos)

    print("\n[COMPLETE] Evaluation finished!")
    print(f"  Results: {args.output_dir}/results/")
    print(f"  Images:  {args.output_dir}/trajectories/")
    print(f"  Videos:  {args.output_dir}/videos/")


if __name__ == "__main__":
    main()
