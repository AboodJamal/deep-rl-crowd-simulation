"""
Generate sample videos from VGA+UPL Deterministic results.
Creates videos for both successful and failed trials.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_trajectory_video(
    start_pos,
    goal_pos,
    obstacles,
    trajectory,
    success,
    output_path,
    title="VGA+UPL Navigation",
):
    """Create animated video of a single trajectory."""

    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get bounds
    all_x = [start_pos[0], goal_pos[0]] + [obs["position"][0] for obs in obstacles]
    all_y = [start_pos[1], goal_pos[1]] + [obs["position"][1] for obs in obstacles]

    if len(trajectory) > 0:
        traj_array = np.array(trajectory)
        all_x.extend(traj_array[:, 0])
        all_y.extend(traj_array[:, 1])

    margin = 1.0
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Plot static elements
    # Start position
    ax.plot(start_pos[0], start_pos[1], "go", markersize=15, label="Start", zorder=5)

    # Goal position
    ax.plot(goal_pos[0], goal_pos[1], "y*", markersize=20, label="Goal", zorder=5)

    # Obstacles
    for obs in obstacles:
        pos = obs["position"]
        radius = obs["radius"]
        circle = Circle(pos, radius, color="red", alpha=0.6, zorder=3)
        ax.add_patch(circle)
        # Personal distance
        personal_circle = Circle(
            pos,
            radius + 0.5,
            color="red",
            alpha=0.1,
            linestyle="--",
            fill=False,
            zorder=2,
        )
        ax.add_patch(personal_circle)

    # Trajectory line and agent
    if len(trajectory) > 0:
        traj_array = np.array(trajectory)
        (line,) = ax.plot([], [], "b-", linewidth=2, alpha=0.6, label="Path", zorder=4)
        agent = Circle((start_pos[0], start_pos[1]), 0.2, color="blue", zorder=6)
        ax.add_patch(agent)
    else:
        line = None
        agent = None

    # Title
    status = "SUCCESS ✓" if success else "FAILED ✗"
    status_color = "green" if success else "red"
    ax.set_title(
        f"{title}\n{status}", fontsize=14, color=status_color, fontweight="bold"
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right")

    # Animation function
    def animate(frame):
        if line is not None and agent is not None and len(trajectory) > 0:
            # Update line
            line.set_data(traj_array[: frame + 1, 0], traj_array[: frame + 1, 1])
            # Update agent position
            agent.center = (traj_array[frame, 0], traj_array[frame, 1])
        return (line, agent) if line is not None else ()

    # Create animation
    frames = len(trajectory) if len(trajectory) > 0 else 1
    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=50, blit=True, repeat=True
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=20, bitrate=1800)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)

    print(f"  Created: {output_path.name}")


def main():
    """Generate sample videos for successes and failures."""

    print("=" * 80)
    print("GENERATING SAMPLE VIDEOS - VGA+UPL DETERMINISTIC")
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

    # Process each scenario
    for scenario_name, scenario_data in all_results.items():
        print(f"\n{scenario_name.upper()}")
        print("-" * 80)

        if "VGA_UPL_Det" not in scenario_data:
            print("  No VGA_UPL_Det data found")
            continue

        trials = scenario_data["VGA_UPL_Det"]

        # Separate successes and failures
        successes = [t for t in trials if t["deterministic"]["success"]]
        failures = [t for t in trials if not t["deterministic"]["success"]]

        print(f"  Successes: {len(successes)}, Failures: {len(failures)}")

        # Generate videos for successes (up to 5)
        num_success_videos = min(5, len(successes))
        if num_success_videos > 0:
            print(f"  Generating {num_success_videos} success videos...")
            for i in range(num_success_videos):
                trial = successes[i]
                output_path = videos_dir / f"{scenario_name}_success_{i+1}.mp4"

                create_trajectory_video(
                    start_pos=trial["start_pos"],
                    goal_pos=trial["goal_pos"],
                    obstacles=trial["obstacles"],
                    trajectory=trial["deterministic"]["positions"],
                    success=True,
                    output_path=output_path,
                    title=f"{scenario_name.upper()} - Trial {trial['trial_id']} (Success)",
                )

        # Generate videos for failures (up to 5)
        num_failure_videos = min(5, len(failures))
        if num_failure_videos > 0:
            print(f"  Generating {num_failure_videos} failure videos...")
            for i in range(num_failure_videos):
                trial = failures[i]
                output_path = videos_dir / f"{scenario_name}_failure_{i+1}.mp4"

                create_trajectory_video(
                    start_pos=trial["start_pos"],
                    goal_pos=trial["goal_pos"],
                    obstacles=trial["obstacles"],
                    trajectory=trial["deterministic"]["positions"],
                    success=False,
                    output_path=output_path,
                    title=f"{scenario_name.upper()} - Trial {trial['trial_id']} (Failed)",
                )

    print("\n" + "=" * 80)
    print(f"VIDEOS SAVED TO: {videos_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
