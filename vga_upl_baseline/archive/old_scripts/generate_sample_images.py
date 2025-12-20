"""
Generate sample visualizations (PNG) from VGA+UPL Deterministic results.
Creates images for both successful and failed trials.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_trajectory_plot(
    start_pos,
    goal_pos,
    obstacles,
    trajectory,
    success,
    output_path,
    title="VGA+UPL Navigation",
):
    """Create static plot of a trajectory."""

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
        # Personal distance (0.1m buffer zone)
        personal_circle = Circle(
            pos,
            radius + 0.1,  # Matches VGA personal_distance parameter
            color="red",
            alpha=0.1,
            linestyle="--",
            fill=False,
            zorder=2,
        )
        ax.add_patch(personal_circle)

    # Trajectory
    if len(trajectory) > 0:
        traj_array = np.array(trajectory)
        ax.plot(
            traj_array[:, 0],
            traj_array[:, 1],
            "b-",
            linewidth=2,
            alpha=0.6,
            label="Path",
            zorder=4,
        )
        # Final position
        ax.plot(
            traj_array[-1, 0],
            traj_array[-1, 1],
            "bs",
            markersize=10,
            label="Final",
            zorder=5,
        )

    # Title
    status = "SUCCESS ✓" if success else "FAILED ✗"
    status_color = "green" if success else "red"
    ax.set_title(
        f"{title}\n{status}", fontsize=14, color=status_color, fontweight="bold"
    )
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Created: {output_path.name}")


def main():
    """Generate sample images for successes and failures."""

    print("=" * 80)
    print("GENERATING SAMPLE IMAGES - VGA+UPL DETERMINISTIC")
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
    images_dir = results_dir / "sample_images"
    images_dir.mkdir(exist_ok=True)

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

        # Generate images for successes (up to 3)
        num_success_images = min(3, len(successes))
        if num_success_images > 0:
            print(f"  Generating {num_success_images} success images...")
            for i in range(num_success_images):
                trial = successes[i]
                output_path = images_dir / f"{scenario_name}_success_{i+1}.png"

                create_trajectory_plot(
                    start_pos=trial["start_pos"],
                    goal_pos=trial["goal_pos"],
                    obstacles=trial["obstacles"],
                    trajectory=trial["deterministic"]["positions"],
                    success=True,
                    output_path=output_path,
                    title=f"{scenario_name.upper()} - Trial {trial['trial_id']} (Success)",
                )

        # Generate images for failures (up to 3)
        num_failure_images = min(3, len(failures))
        if num_failure_images > 0:
            print(f"  Generating {num_failure_images} failure images...")
            for i in range(num_failure_images):
                trial = failures[i]
                output_path = images_dir / f"{scenario_name}_failure_{i+1}.png"

                create_trajectory_plot(
                    start_pos=trial["start_pos"],
                    goal_pos=trial["goal_pos"],
                    obstacles=trial["obstacles"],
                    trajectory=trial["deterministic"]["positions"],
                    success=False,
                    output_path=output_path,
                    title=f"{scenario_name.upper()} - Trial {trial['trial_id']} (Failed)",
                )

    print("\n" + "=" * 80)
    print(f"IMAGES SAVED TO: {images_dir}")
    print("=" * 80)
    print("\nTo view the images, open the folder:")
    print(f"  {images_dir.absolute()}")


if __name__ == "__main__":
    main()
