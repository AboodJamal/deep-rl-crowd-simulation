"""
Generate animated videos from VGA+UPL results with arrows and visualizations.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_video(
    start_pos,
    goal_pos,
    obstacles,
    trajectory,
    success,
    output_path,
    title="VGA+UPL Navigation",
):
    """Create animated video of trajectory with arrows."""

    if len(trajectory) < 2:
        print(f"  Skipping {output_path.name} - trajectory too short")
        return

    traj_array = np.array(trajectory)

    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 9))

    # Get bounds
    all_x = [start_pos[0], goal_pos[0]] + [obs["position"][0] for obs in obstacles]
    all_y = [start_pos[1], goal_pos[1]] + [obs["position"][1] for obs in obstacles]
    all_x.extend(traj_array[:, 0])
    all_y.extend(traj_array[:, 1])

    margin = 1.0
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Draw static elements (obstacles, start, goal)
    for obs in obstacles:
        pos = obs["position"]
        radius = obs["radius"]

        # Obstacle physical boundary
        circle_physical = Circle(pos, radius, color="red", alpha=0.6, zorder=2)
        ax.add_patch(circle_physical)

        # Personal space (light red halo)
        personal_radius = radius + 0.1  # personal_distance
        circle_personal = Circle(
            pos,
            personal_radius,
            color="red",
            alpha=0.15,
            zorder=1,
            linestyle="--",
            linewidth=1,
            fill=True,
        )
        ax.add_patch(circle_personal)

    # Start position
    ax.plot(
        start_pos[0],
        start_pos[1],
        "go",
        markersize=15,
        label="Start",
        zorder=10,
        markeredgecolor="darkgreen",
        markeredgewidth=2,
    )

    # Goal position
    ax.plot(
        goal_pos[0],
        goal_pos[1],
        marker="*",
        color="gold",
        markersize=25,
        label="Goal",
        zorder=10,
        markeredgecolor="orange",
        markeredgewidth=2,
    )

    # Dynamic elements (will be updated in animation)
    (path_line,) = ax.plot(
        [], [], "b-", linewidth=2.5, alpha=0.7, label="Path", zorder=5
    )
    agent_circle = Circle(
        (0, 0), 0.2, color="blue", alpha=0.8, zorder=15
    )  # VGA paper: 0.2m
    ax.add_patch(agent_circle)

    # Arrow for direction (will be updated)
    arrow = None

    # Title
    status = "SUCCESS ✓" if success else "FAILED ✗"
    status_color = "green" if success else "red"
    title_text = ax.text(
        0.5,
        1.02,
        f"{title}\n{status}",
        transform=ax.transAxes,
        fontsize=16,
        color=status_color,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    # Step counter
    step_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=12,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.set_xlabel("X (m)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Y (m)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)

    def init():
        path_line.set_data([], [])
        agent_circle.center = (traj_array[0, 0], traj_array[0, 1])
        step_text.set_text("")
        return path_line, agent_circle, step_text

    def animate(frame):
        nonlocal arrow

        # Update path line (show trail)
        path_line.set_data(traj_array[: frame + 1, 0], traj_array[: frame + 1, 1])

        # Update agent position
        pos = traj_array[frame]
        agent_circle.center = (pos[0], pos[1])

        # Update direction arrow
        if arrow is not None:
            try:
                arrow.remove()
            except:
                pass
            arrow = None

        if frame < len(traj_array) - 1:
            # Calculate velocity direction
            next_pos = traj_array[frame + 1]
            direction = next_pos - pos
            dir_norm = np.linalg.norm(direction)

            if dir_norm > 0.01:
                direction = direction / dir_norm * 0.4  # Arrow length
                arrow = FancyArrow(
                    pos[0],
                    pos[1],
                    direction[0],
                    direction[1],
                    width=0.15,
                    head_width=0.25,
                    head_length=0.15,
                    fc="darkblue",
                    ec="darkblue",
                    alpha=0.9,
                    zorder=20,
                )
                ax.add_patch(arrow)

        # Update step counter
        dist_to_goal = np.linalg.norm(pos - np.array(goal_pos))
        step_text.set_text(
            f"Step: {frame+1}/{len(traj_array)}\nDist to goal: {dist_to_goal:.2f}m"
        )

        return path_line, agent_circle, step_text

    # Create animation
    total_frames = len(traj_array)
    # Slow down: show every 2nd frame if trajectory is long
    skip = max(1, total_frames // 200)
    frames = range(0, total_frames, skip)

    anim = FuncAnimation(
        fig, animate, init_func=init, frames=frames, interval=50, blit=False
    )

    # Save as GIF (no FFmpeg needed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Change extension to .gif
        gif_path = output_path.with_suffix(".gif")
        anim.save(str(gif_path), writer="pillow", fps=20, dpi=80)
        plt.close(fig)
        print(f"  Created: {gif_path.name}")
    except Exception as e:
        print(f"  Error creating {output_path.name}: {e}")
        plt.close(fig)


def main():
    """Generate videos for successes and failures."""

    print("=" * 80)
    print("GENERATING SAMPLE VIDEOS - VGA+UPL")
    print("=" * 80)

    # Load results
    results_dir = Path("validation/results/vga_final_experiment")
    results_file = results_dir / "comprehensive_results.json"

    if not results_file.exists():
        print(f"ERROR: Results file not found: {results_file}")
        return

    with open(results_file, "r") as f:
        all_results = json.load(f)

    # Output directory
    videos_dir = results_dir / "sample_videos"
    videos_dir.mkdir(exist_ok=True)

    scenarios = ["SOSP", "MOSP_A", "MOSP_B", "MOSP_C", "MOSP_D"]

    for scenario in scenarios:
        print(f"\n{scenario}")
        print("-" * 80)

        if scenario not in all_results:
            print(f"  No results for {scenario}")
            continue

        scenario_results = all_results[scenario].get("VGA_UPL_Det", [])

        if not scenario_results:
            print(f"  No trials for {scenario}")
            continue

        # Separate successes and failures based on deterministic results
        successes = []
        failures = []

        for r in scenario_results:
            if "deterministic" in r and "positions" in r["deterministic"]:
                det = r["deterministic"]
                result_data = {
                    "start_pos": r["start_pos"],
                    "goal_pos": r["goal_pos"],
                    "obstacles": r["obstacles"],
                    "trajectory": det["positions"],
                    "success": det["success"],
                    "trial_id": r.get("trial_id", 0),
                }

                if det["success"]:
                    successes.append(result_data)
                else:
                    failures.append(result_data)

        # If no trajectory data, skip this scenario
        if len(successes) == 0 and len(failures) == 0:
            print(f"  No trajectory data available")
            continue

        print(f"  Successes: {len(successes)}, Failures: {len(failures)}")

        # Generate videos for successes (up to 2)
        num_success_videos = min(2, len(successes))
        if num_success_videos > 0:
            print(f"  Generating {num_success_videos} success videos...")
            for i in range(num_success_videos):
                result = successes[i]
                video_path = videos_dir / f"{scenario}_success_{i+1}.mp4"

                create_video(
                    start_pos=result["start_pos"],
                    goal_pos=result["goal_pos"],
                    obstacles=result["obstacles"],
                    trajectory=result["trajectory"],
                    success=True,
                    output_path=video_path,
                    title=f"{scenario} - Trial {result.get('trial_id', i+1)} (Success)",
                )

        # Generate videos for failures (up to 2)
        num_failure_videos = min(2, len(failures))
        if num_failure_videos > 0:
            print(f"  Generating {num_failure_videos} failure videos...")
            for i in range(num_failure_videos):
                result = failures[i]
                video_path = videos_dir / f"{scenario}_failure_{i+1}.mp4"

                create_video(
                    start_pos=result["start_pos"],
                    goal_pos=result["goal_pos"],
                    obstacles=result["obstacles"],
                    trajectory=result["trajectory"],
                    success=False,
                    output_path=video_path,
                    title=f"{scenario} - Trial {result.get('trial_id', i+1)} (Failed)",
                )

    print("\n" + "=" * 80)
    print(f"VIDEOS SAVED TO: {videos_dir}")
    print("=" * 80)
    print(f"\nTo view the videos, open the folder:")
    print(f"  {videos_dir.absolute()}")


if __name__ == "__main__":
    main()
