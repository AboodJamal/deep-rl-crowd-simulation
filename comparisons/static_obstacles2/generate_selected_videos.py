"""
Generate videos for selected trials:
- 3 collision trials
- 2 sample trials from each scenario (10 trials)
Total: ~13 videos
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "vga_baseline" / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "vga_baseline" / "data_loading"))

# Import dataset loader
from vga_dataset import VGADatasetLoader

# Load dataset to get obstacle information
data_root = str(PROJECT_ROOT / "data" / "VGA-Experimental-Data")
dataset_loader = VGADatasetLoader(data_root=data_root)

# Load comparison results
with open("comparison_results.json", "r") as f:
    data = json.load(f)

# Constants
AGENT_RADIUS = 0.2
OBSTACLE_RADIUS = 0.25
ARENA_X_MIN = 0.0
ARENA_X_MAX = 10.0
ARENA_Y_MIN = -1.75
ARENA_Y_MAX = 1.75


def create_video(
    vga_positions,
    drl_positions,
    obstacles,
    start_pos,
    goal_pos,
    scenario,
    trial_idx,
    vga_success,
    drl_success,
    output_path,
):
    """Create side-by-side comparison video."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    def setup_ax(ax, title):
        ax.set_xlim(ARENA_X_MIN - 0.5, ARENA_X_MAX + 0.5)
        ax.set_ylim(ARENA_Y_MIN - 0.5, ARENA_Y_MAX + 0.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight="bold")

        # Draw obstacles - LARGER and more visible
        for ox, oy, r in obstacles:
            circle = Circle(
                (ox, oy),
                r,
                color="darkred",
                alpha=0.8,
                zorder=1,
                linewidth=2,
                edgecolor="black",
            )
            ax.add_patch(circle)

        # Draw start and goal
        ax.plot(
            start_pos[0], start_pos[1], "go", markersize=12, label="Start", zorder=5
        )
        ax.plot(goal_pos[0], goal_pos[1], "r*", markersize=15, label="Goal", zorder=5)
        ax.legend(loc="upper right", fontsize=8)

    setup_ax(ax1, f'VGA+UPL V4 ({"SUCCESS" if vga_success else "FAIL"})')
    setup_ax(ax2, f'DRL PPO ({"SUCCESS" if drl_success else "FAIL"})')

    # Initialize agents
    vga_agent = Circle((0, 0), AGENT_RADIUS, color="blue", alpha=0.7, zorder=10)
    drl_agent = Circle((0, 0), AGENT_RADIUS, color="red", alpha=0.7, zorder=10)
    ax1.add_patch(vga_agent)
    ax2.add_patch(drl_agent)

    # Trajectory lines
    (vga_line,) = ax1.plot([], [], "b-", alpha=0.3, linewidth=1.5)
    (drl_line,) = ax2.plot([], [], "r-", alpha=0.3, linewidth=1.5)

    max_frames = max(len(vga_positions), len(drl_positions))

    def animate(frame):
        # VGA
        if frame < len(vga_positions):
            vga_agent.center = vga_positions[frame]
            vga_line.set_data(
                vga_positions[: frame + 1, 0], vga_positions[: frame + 1, 1]
            )

        # DRL
        if frame < len(drl_positions):
            drl_agent.center = drl_positions[frame]
            drl_line.set_data(
                drl_positions[: frame + 1, 0], drl_positions[: frame + 1, 1]
            )

        return vga_agent, drl_agent, vga_line, drl_line

    plt.suptitle(f"{scenario} - Trial {trial_idx}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    anim = animation.FuncAnimation(
        fig, animate, frames=max_frames, interval=50, blit=True, repeat=True
    )

    # Save as MP4 instead of GIF for better stability
    print(f"  Saving video to {output_path}...")
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=20, metadata=dict(artist="VGA Comparison"), bitrate=1800)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)
    print(f"  [OK] Video saved")


# Trials to generate videos for
trials_to_generate = {
    "SOSP": [1, 2],  # First 2 trials as samples
    "MOSP_A": [1, 2],
    "MOSP_B": [1, 2],
    "MOSP_C": [1, 2, 11, 14],  # Include collision trials 11 & 14
    "MOSP_D": [1, 2, 14],  # Include collision trial 14
}

print("=" * 70)
print("GENERATING SELECTED VIDEOS")
print("=" * 70)
print()

total_videos = sum(len(trials) for trials in trials_to_generate.values())
video_count = 0

for scenario in ["SOSP", "MOSP_A", "MOSP_B", "MOSP_C", "MOSP_D"]:
    if scenario not in data["scenarios"]:
        continue

    print(f"[{scenario}]")

    scenario_data = data["scenarios"][scenario]
    vga_trials = scenario_data["vga_trials"]
    drl_trials = scenario_data["drl_trials"]

    video_dir = Path(scenario) / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # Load obstacles from dataset for this scenario
    if scenario == "SOSP":
        dataset_trials = dataset_loader.load_sosp()
    else:
        case = scenario.split("_")[1]
        dataset_trials = dataset_loader.load_mosp(case)

    for trial_idx in trials_to_generate[scenario]:
        trial_num = trial_idx - 1  # 0-indexed

        if trial_num >= len(vga_trials):
            continue

        vga_trial = vga_trials[trial_num]
        drl_trial = drl_trials[trial_num]

        # Get obstacles from dataset
        dataset_trial = dataset_trials[trial_num]
        obstacles = [
            (obs["position"][0], obs["position"][1], obs["radius"])
            for obs in dataset_trial.obstacles
        ]

        # Check if this is a collision trial
        is_collision = vga_trial["num_collisions"] > 0
        collision_marker = " [COLLISION]" if is_collision else ""

        print(f"  Trial {trial_idx}{collision_marker}...")

        # Convert to numpy arrays
        vga_positions = np.array(vga_trial["positions"])
        drl_positions = np.array(drl_trial["positions"])
        start_pos = np.array(vga_trial["start_pos"])
        goal_pos = np.array(vga_trial["goal_pos"])

        # obstacles already loaded from dataset above

        output_path = video_dir / f"trial_{trial_idx:03d}.mp4"

        try:
            create_video(
                vga_positions,
                drl_positions,
                obstacles,
                start_pos,
                goal_pos,
                scenario,
                trial_idx,
                vga_trial["success"],
                drl_trial["success"],
                output_path,
            )
            video_count += 1
        except Exception as e:
            print(f"  [ERROR] {str(e)[:100]}")

    print()

print("=" * 70)
print(f"COMPLETE: Generated {video_count}/{total_videos} videos")
print("=" * 70)
