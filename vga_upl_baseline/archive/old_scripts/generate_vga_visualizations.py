"""
Generate videos and plots for VGA+UPL final experiment results.
Run this after run_vga_final.py completes.
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_single_video(positions, obstacles, start, goal, title, output_path):
    """Create video of single trajectory."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate bounds
    all_x = (
        [p[0] for p in positions]
        + [start[0], goal[0]]
        + [o["position"][0] for o in obstacles]
    )
    all_y = (
        [p[1] for p in positions]
        + [start[1], goal[1]]
        + [o["position"][1] for o in obstacles]
    )

    margin = 1.0
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)

    # Draw obstacles
    for obs in obstacles:
        circle = patches.Circle(obs["position"], obs["radius"], color="red", alpha=0.5)
        ax.add_patch(circle)

    # Start and goal
    ax.plot(*start, "go", markersize=10, label="Start")
    ax.plot(*goal, "b*", markersize=15, label="Goal")

    # Path
    (path_line,) = ax.plot([], [], "b-", linewidth=1, alpha=0.5)
    (agent_dot,) = ax.plot([], [], "bo", markersize=8)

    step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    def init():
        path_line.set_data([], [])
        agent_dot.set_data([], [])
        return path_line, agent_dot, step_text

    def animate(frame):
        path_line.set_data(
            [p[0] for p in positions[: frame + 1]],
            [p[1] for p in positions[: frame + 1]],
        )
        agent_dot.set_data([positions[frame][0]], [positions[frame][1]])
        step_text.set_text(f"Step: {frame}/{len(positions)}")
        return path_line, agent_dot, step_text

    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(positions),
        interval=50,
        blit=True,
        repeat=True,
    )

    anim.save(output_path, writer="ffmpeg", fps=20, dpi=100)
    plt.close()


def generate_visualizations():
    """Generate videos and plots from experiment results."""

    results_dir = Path("validation/results/vga_final_experiment")
    videos_dir = results_dir / "videos"
    plots_dir = results_dir / "plots"

    videos_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Load results
    with open(results_dir / "comprehensive_results.json") as f:
        all_results = json.load(f)

    # Generate for first 3 trials of each scenario
    for scenario_name in ["SOSP", "MOSP_A", "MOSP_B", "MOSP_C", "MOSP_D"]:
        print(f"\n{scenario_name}:")
        scenario_results = all_results[scenario_name]

        # Get first 3 trials
        det_results = scenario_results["VGA_UPL_Det"][:3]
        stoch_results = scenario_results["VGA_UPL_Stoch"][:3]

        for i, (det_trial, stoch_trial) in enumerate(zip(det_results, stoch_results)):
            trial_id = det_trial["trial_id"]
            print(f"  Trial {trial_id}...")

            # Deterministic video
            video_path = videos_dir / f"{scenario_name.lower()}_trial{trial_id}_det.mp4"
            create_single_video(
                positions=det_trial["deterministic"]["positions"],
                obstacles=det_trial["obstacles"],
                start=det_trial["start_pos"],
                goal=det_trial["goal_pos"],
                title=f"{scenario_name} Trial {trial_id} - VGA+UPL Deterministic",
                output_path=str(video_path),
            )

            # Stochastic video (first run)
            if stoch_trial["stochastic_runs"]:
                video_path = (
                    videos_dir / f"{scenario_name.lower()}_trial{trial_id}_stoch.mp4"
                )
                create_single_video(
                    positions=stoch_trial["stochastic_runs"][0]["positions"],
                    obstacles=stoch_trial["obstacles"],
                    start=stoch_trial["start_pos"],
                    goal=stoch_trial["goal_pos"],
                    title=f"{scenario_name} Trial {trial_id} - VGA+UPL Stochastic",
                    output_path=str(video_path),
                )

            # Comparison plot
            fig, ax = plt.subplots(figsize=(12, 6))

            # Obstacles
            for obs in det_trial["obstacles"]:
                circle = patches.Circle(
                    obs["position"], obs["radius"], color="gray", alpha=0.5
                )
                ax.add_patch(circle)

            # Trajectories
            det_pos = np.array(det_trial["deterministic"]["positions"])
            ax.plot(
                det_pos[:, 0],
                det_pos[:, 1],
                "b-",
                linewidth=2,
                label="Deterministic",
                alpha=0.7,
            )

            if stoch_trial["stochastic_runs"]:
                stoch_pos = np.array(stoch_trial["stochastic_runs"][0]["positions"])
                ax.plot(
                    stoch_pos[:, 0],
                    stoch_pos[:, 1],
                    "r-",
                    linewidth=2,
                    label="Stochastic",
                    alpha=0.7,
                )

            # Start and goal
            ax.plot(
                *det_trial["start_pos"], "go", markersize=12, label="Start", zorder=5
            )
            ax.plot(*det_trial["goal_pos"], "b*", markersize=15, label="Goal", zorder=5)

            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title(f"{scenario_name} Trial {trial_id} - Trajectory Comparison")

            plt.tight_layout()
            plt.savefig(
                plots_dir / f"{scenario_name.lower()}_trial{trial_id}_comparison.png",
                dpi=150,
            )
            plt.close()

    print(f"\n{'='*80}")
    print("VISUALIZATIONS COMPLETE!")
    print(f"Videos: {videos_dir}")
    print(f"Plots: {plots_dir}")
    print("=" * 80)


if __name__ == "__main__":
    generate_visualizations()
