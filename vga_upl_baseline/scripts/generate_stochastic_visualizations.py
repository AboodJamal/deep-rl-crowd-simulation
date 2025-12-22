"""
VGA+UPL Stochastic Visualization Generator
==========================================

Generates paper-style visualizations showing:
- Multiple stochastic trajectories overlaid on same plot
- Path percentages (like the paper's Figure 7)
- Summary images with numbered paths and percentages

For each scenario, runs N stochastic simulations and groups
similar trajectories to show path distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib import animation
from pathlib import Path
from datetime import datetime
import json
import sys
import os
from collections import defaultdict

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

from models.vga_upl_planner_v4 import VGAUPLPlannerV4
from data_loading.vga_dataset import VGADatasetLoader as VGADataset


# ============================================================
# Configuration
# ============================================================

NUM_STOCHASTIC_RUNS = 100  # Number of stochastic runs per trial
NUM_TRIALS_PER_SCENARIO = 5  # How many trials to test
NUM_VIZ_PER_SCENARIO = 5  # How many visualizations to generate

# Path cluster settings
CLUSTER_TOLERANCE = 0.5  # meters - paths within this are same "route"

# Distinct colors for different path groups (matching paper style)
PATH_COLORS = [
    "#FF0000",  # Red
    "#0000FF",  # Blue
    "#00AA00",  # Green
    "#FF8C00",  # Orange
    "#800080",  # Purple
    "#00CED1",  # Cyan
    "#FFD700",  # Gold
    "#FF1493",  # Pink
    "#32CD32",  # Lime
    "#8B4513",  # Brown
]


# ============================================================
# Path Clustering
# ============================================================


def trajectory_to_key_points(
    trajectory: np.ndarray, num_points: int = 10
) -> np.ndarray:
    """
    Reduce trajectory to key points for comparison.
    Uses uniform sampling along path.
    """
    if len(trajectory) < 2:
        return trajectory

    # Uniform index sampling
    indices = np.linspace(0, len(trajectory) - 1, num_points, dtype=int)
    return trajectory[indices]


def trajectory_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """
    Calculate distance between two trajectories.
    Uses mean squared distance between key points.
    """
    kp1 = trajectory_to_key_points(traj1)
    kp2 = trajectory_to_key_points(traj2)

    # Handle different lengths
    if len(kp1) != len(kp2):
        min_len = min(len(kp1), len(kp2))
        kp1 = kp1[:min_len]
        kp2 = kp2[:min_len]

    return np.mean(np.linalg.norm(kp1 - kp2, axis=1))


def cluster_trajectories(
    trajectories: list, tolerance: float = CLUSTER_TOLERANCE
) -> dict:
    """
    Cluster trajectories into groups based on similarity.
    Returns dict mapping cluster_id -> list of trajectory indices.
    """
    if not trajectories:
        return {}

    n = len(trajectories)
    assigned = [-1] * n
    cluster_id = 0

    for i in range(n):
        if assigned[i] >= 0:
            continue

        # Start new cluster with this trajectory
        assigned[i] = cluster_id

        # Find all similar trajectories
        for j in range(i + 1, n):
            if assigned[j] >= 0:
                continue

            dist = trajectory_distance(trajectories[i], trajectories[j])
            if dist < tolerance:
                assigned[j] = cluster_id

        cluster_id += 1

    # Group by cluster
    clusters = defaultdict(list)
    for i, cid in enumerate(assigned):
        clusters[cid].append(i)

    return dict(clusters)


def get_cluster_representative(trajectories: list, indices: list) -> np.ndarray:
    """Get a representative trajectory for a cluster (one with median length)."""
    cluster_trajs = [trajectories[i] for i in indices]
    lengths = [len(t) for t in cluster_trajs]
    median_idx = np.argsort(lengths)[len(lengths) // 2]
    return cluster_trajs[median_idx]


# ============================================================
# Visualization Functions
# ============================================================


def create_paper_style_summary(
    trajectories: list,
    obstacles: list,
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    output_path: str,
    title: str = "Simulation",
    scenario_label: str = "",
):
    """
    Create paper-style summary image showing all path variations with percentages.

    Like the paper's Figure 7:
    - Multiple colored paths
    - Numbered circles on paths
    - Percentage labels on right side
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    # Calculate bounds
    all_points = [np.vstack(trajectories)] if trajectories else []
    all_points.append(np.array([start_pos, goal_pos]))

    for obs in obstacles:
        obs_pos = np.array(obs["position"])
        obs_r = obs.get("radius", 0.25)
        all_points.append(
            np.array(
                [
                    [obs_pos[0] - obs_r, obs_pos[1] - obs_r],
                    [obs_pos[0] + obs_r, obs_pos[1] + obs_r],
                ]
            )
        )

    if all_points:
        all_coords = np.vstack(all_points)
        x_min, x_max = all_coords[:, 0].min() - 0.5, all_coords[:, 0].max() + 1.5
        y_min, y_max = all_coords[:, 1].min() - 0.5, all_coords[:, 1].max() + 0.5
    else:
        x_min, x_max, y_min, y_max = -1, 12, -2, 2

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")

    # Grid and labels
    ax.set_xlabel("X-coordinate (m)", fontsize=11)
    ax.set_ylabel("Y-coordinate (m)", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Draw obstacles as circles with X pattern
    for obs in obstacles:
        obs_pos = np.array(obs["position"])
        obs_r = obs.get("radius", 0.25)
        circle = plt.Circle(
            obs_pos, obs_r, color="gray", alpha=0.7, ec="black", linewidth=1
        )
        ax.add_patch(circle)
        # Add X pattern
        ax.plot(
            [obs_pos[0] - obs_r * 0.5, obs_pos[0] + obs_r * 0.5],
            [obs_pos[1] - obs_r * 0.5, obs_pos[1] + obs_r * 0.5],
            "k-",
            linewidth=1,
        )
        ax.plot(
            [obs_pos[0] - obs_r * 0.5, obs_pos[0] + obs_r * 0.5],
            [obs_pos[1] + obs_r * 0.5, obs_pos[1] - obs_r * 0.5],
            "k-",
            linewidth=1,
        )

    # Draw start/goal
    ax.plot(
        start_pos[0],
        start_pos[1],
        "ko",
        markersize=12,
        markerfacecolor="white",
        markeredgewidth=2,
        zorder=100,
    )
    ax.plot(
        goal_pos[0],
        goal_pos[1],
        "ko",
        markersize=14,
        markerfacecolor="white",
        markeredgewidth=2,
        zorder=100,
    )
    # Cross in goal
    ax.plot(
        goal_pos[0], goal_pos[1], "kx", markersize=10, markeredgewidth=2, zorder=101
    )

    if not trajectories:
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close()
        return {}

    # Cluster trajectories
    clusters = cluster_trajectories(trajectories)

    # Sort clusters by size (most common first)
    sorted_clusters = sorted(clusters.items(), key=lambda x: -len(x[1]))

    total_runs = len(trajectories)
    percentages = {}

    # Draw each cluster
    for cluster_idx, (cid, indices) in enumerate(sorted_clusters):
        if cluster_idx >= len(PATH_COLORS):
            color = "gray"
        else:
            color = PATH_COLORS[cluster_idx]

        percentage = len(indices) / total_runs * 100
        percentages[cluster_idx + 1] = percentage

        # Get representative trajectory
        rep_traj = get_cluster_representative(trajectories, indices)

        # Draw all trajectories in this cluster (faded)
        for idx in indices:
            traj = trajectories[idx]
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.15, linewidth=1)

        # Draw representative (bold)
        ax.plot(rep_traj[:, 0], rep_traj[:, 1], color=color, linewidth=2.5, alpha=0.9)

        # Add number label on path
        mid_idx = len(rep_traj) // 3  # Place number 1/3 along path
        mid_point = rep_traj[mid_idx]

        # Draw circle with number
        circle = plt.Circle(
            mid_point,
            0.3,
            color="lightgray",
            ec="black",
            linewidth=1,
            zorder=50,
            alpha=0.9,
        )
        ax.add_patch(circle)
        ax.text(
            mid_point[0],
            mid_point[1],
            str(cluster_idx + 1),
            fontsize=10,
            ha="center",
            va="center",
            fontweight="bold",
            zorder=51,
        )

    # Add percentage legend on right side
    legend_x = x_max - 0.3
    legend_y_start = y_max - 0.3
    legend_spacing = 0.5

    for cluster_idx, (cid, indices) in enumerate(
        sorted_clusters[: min(10, len(sorted_clusters))]
    ):
        if cluster_idx >= len(PATH_COLORS):
            color = "gray"
        else:
            color = PATH_COLORS[cluster_idx]

        percentage = len(indices) / total_runs * 100
        y_pos = legend_y_start - cluster_idx * legend_spacing

        # Color line
        ax.plot([legend_x - 0.5, legend_x], [y_pos, y_pos], color=color, linewidth=3)
        # Percentage text
        ax.text(
            legend_x + 0.15,
            y_pos,
            f"{percentage:.1f}%",
            fontsize=9,
            va="center",
            ha="left",
        )

    # Add scenario label box (like "Case A" in paper)
    if scenario_label:
        ax.text(
            0.02,
            0.98,
            scenario_label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="square,pad=0.3",
                facecolor="white",
                edgecolor="black",
                linewidth=1,
            ),
        )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

    return percentages


def create_stochastic_video(
    trajectories: list,
    obstacles: list,
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    output_path: str,
    title: str = "Stochastic VGA",
    fps: int = 30,
):
    """
    Create video showing multiple stochastic runs.
    Draws all trajectories progressively.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    # Calculate bounds
    all_points = [np.vstack(trajectories)] if trajectories else [np.array([[0, 0]])]
    all_points.append(np.array([start_pos, goal_pos]))

    for obs in obstacles:
        obs_pos = np.array(obs["position"])
        obs_r = obs.get("radius", 0.25)
        all_points.append(
            np.array(
                [
                    [obs_pos[0] - obs_r, obs_pos[1] - obs_r],
                    [obs_pos[0] + obs_r, obs_pos[1] + obs_r],
                ]
            )
        )

    all_coords = np.vstack(all_points)
    x_min, x_max = all_coords[:, 0].min() - 0.5, all_coords[:, 0].max() + 0.5
    y_min, y_max = all_coords[:, 1].min() - 0.5, all_coords[:, 1].max() + 0.5

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X-coordinate (m)")
    ax.set_ylabel("Y-coordinate (m)")
    ax.set_title(title)

    # Draw obstacles
    for obs in obstacles:
        obs_pos = np.array(obs["position"])
        obs_r = obs.get("radius", 0.25)
        circle = plt.Circle(obs_pos, obs_r, color="red", alpha=0.6)
        ax.add_patch(circle)

    # Draw start/goal
    ax.plot(start_pos[0], start_pos[1], "go", markersize=12, label="Start", zorder=100)
    ax.plot(goal_pos[0], goal_pos[1], "b*", markersize=15, label="Goal", zorder=100)

    # Cluster for colors
    clusters = cluster_trajectories(trajectories)
    traj_colors = ["gray"] * len(trajectories)
    for cid, indices in clusters.items():
        color = PATH_COLORS[cid % len(PATH_COLORS)]
        for idx in indices:
            traj_colors[idx] = color

    # Create line objects
    lines = []
    for i, traj in enumerate(trajectories):
        (line,) = ax.plot([], [], color=traj_colors[i], alpha=0.4, linewidth=1)
        lines.append(line)

    # Agent marker
    (agent,) = ax.plot([], [], "ko", markersize=8, zorder=50)

    # Text
    info_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    ax.legend(loc="upper right")

    max_len = max(len(t) for t in trajectories)

    def init():
        for line in lines:
            line.set_data([], [])
        agent.set_data([], [])
        return lines + [agent, info_text]

    def animate(frame):
        completed = 0
        for i, traj in enumerate(trajectories):
            if frame < len(traj):
                lines[i].set_data(traj[: frame + 1, 0], traj[: frame + 1, 1])
            else:
                lines[i].set_data(traj[:, 0], traj[:, 1])
                completed += 1

        # Show first trajectory's agent
        if frame < len(trajectories[0]):
            agent.set_data([trajectories[0][frame, 0]], [trajectories[0][frame, 1]])

        info_text.set_text(
            f"Runs: {len(trajectories)}\nFrame: {frame}/{max_len}\nCompleted: {completed}"
        )
        return lines + [agent, info_text]

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=max_len + 30,
        interval=1000 / fps,
        blit=False,
    )

    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(output_path, writer=writer)
    plt.close()


# ============================================================
# Main Runner
# ============================================================


def run_stochastic_trial(planner, trial_data: dict, num_runs: int = 100) -> list:
    """
    Run multiple stochastic simulations on one trial.
    Returns list of trajectory arrays.
    """
    trajectories = []

    start = np.array(trial_data["start_pos"])
    goal = np.array(trial_data["goal_pos"])
    obstacles = trial_data.get("obstacles", [])

    for run_idx in range(num_runs):
        # Reset planner
        planner.reset(start, goal, obstacles=obstacles)

        positions = [planner.pos.copy()]
        max_steps = 500

        for step in range(max_steps):
            done = planner.step()
            positions.append(planner.pos.copy())

            if done:
                break

        trajectories.append(np.array(positions))

    return trajectories


def main():
    print("=" * 70)
    print("VGA+UPL Stochastic - Visualization Generator")
    print("=" * 70)

    # Setup paths
    base_dir = Path(__file__).parent.parent
    # Data is in project root's data folder
    project_root = base_dir.parent
    data_dir = project_root / "data" / "VGA-Experimental-Data"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / "results" / f"vga_v4_Stochastic_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "videos").mkdir(exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "summary_images").mkdir(exist_ok=True)

    # Load dataset
    print(f"\n📁 Loading data from: {data_dir}")
    dataset = VGADataset(str(data_dir))

    # Load all trials - using correct method names
    sosp_raw = dataset.load_sosp()
    mosp_a_raw = dataset.load_mosp("A")
    mosp_b_raw = dataset.load_mosp("B")
    mosp_c_raw = dataset.load_mosp("C")
    mosp_d_raw = dataset.load_mosp("D")

    # Convert ExperimentalTrial objects to dict format
    def trial_to_dict(trial):
        return {
            "trial_id": trial.trial_id,
            "start_pos": trial.initial_pos.tolist(),
            "goal_pos": trial.final_pos.tolist(),
            "obstacles": trial.obstacles,
        }

    scenarios = {
        "SOSP": [trial_to_dict(t) for t in sosp_raw],
        "MOSP_A": [trial_to_dict(t) for t in mosp_a_raw],
        "MOSP_B": [trial_to_dict(t) for t in mosp_b_raw],
        "MOSP_C": [trial_to_dict(t) for t in mosp_c_raw],
        "MOSP_D": [trial_to_dict(t) for t in mosp_d_raw],
    }

    for name, trials in scenarios.items():
        print(f"Loaded {name}: {len(trials)} trials")

    # Create stochastic planner
    planner = VGAUPLPlannerV4(
        use_probabilistic=True,
        agent_radius=0.2,
        min_clearance=0.02,
        desired_speed=1.34,
    )

    # Results storage
    all_results = {}

    # Scenario labels like paper
    scenario_labels = {
        "SOSP": "SOSP",
        "MOSP_A": "Case A",
        "MOSP_B": "Case B",
        "MOSP_C": "Case C",
        "MOSP_D": "Case D",
    }

    # Process each scenario
    for scenario_name, trials in scenarios.items():
        print(f"\n{'='*50}")
        print(f"📍 Processing {scenario_name}")
        print(f"{'='*50}")

        scenario_results = []

        # Select trials to process
        num_trials = min(NUM_TRIALS_PER_SCENARIO, len(trials))
        trial_indices = np.linspace(0, len(trials) - 1, num_trials, dtype=int)

        print(
            f"   Running {NUM_STOCHASTIC_RUNS} stochastic simulations on {num_trials} trials..."
        )

        for viz_idx, trial_idx in enumerate(trial_indices):
            trial = trials[trial_idx]
            trial_id = trial.get("trial_id", trial_idx + 1)

            print(f"\n   Trial {trial_id} ({viz_idx+1}/{num_trials}):")

            # Run stochastic simulations
            trajectories = run_stochastic_trial(planner, trial, NUM_STOCHASTIC_RUNS)

            # Count successes
            successes = sum(
                1
                for t in trajectories
                if np.linalg.norm(t[-1] - np.array(trial["goal_pos"])) < 0.3
            )

            print(
                f"      Success rate: {successes}/{NUM_STOCHASTIC_RUNS} ({100*successes/NUM_STOCHASTIC_RUNS:.1f}%)"
            )

            # Cluster analysis
            clusters = cluster_trajectories(trajectories)
            print(f"      Distinct paths: {len(clusters)}")

            # Generate summary image (paper style)
            summary_path = output_dir / "summary_images" / scenario_name
            summary_path.mkdir(exist_ok=True)

            percentages = create_paper_style_summary(
                trajectories=trajectories,
                obstacles=trial.get("obstacles", []),
                start_pos=np.array(trial["start_pos"]),
                goal_pos=np.array(trial["goal_pos"]),
                output_path=str(summary_path / f"trial_{trial_id:03d}_summary.png"),
                title="Simulation",
                scenario_label=scenario_labels.get(scenario_name, scenario_name),
            )
            print(f"      ✅ Summary image saved")

            # Generate video (only for first NUM_VIZ_PER_SCENARIO)
            if viz_idx < NUM_VIZ_PER_SCENARIO:
                video_path = output_dir / "videos" / scenario_name
                video_path.mkdir(exist_ok=True)

                create_stochastic_video(
                    trajectories=trajectories,
                    obstacles=trial.get("obstacles", []),
                    start_pos=np.array(trial["start_pos"]),
                    goal_pos=np.array(trial["goal_pos"]),
                    output_path=str(
                        video_path / f"trial_{trial_id:03d}_stochastic.mp4"
                    ),
                    title=f"{scenario_name} Trial {trial_id} - Stochastic VGA ({NUM_STOCHASTIC_RUNS} runs)",
                )
                print(f"      ✅ Video saved")

            # Store results
            scenario_results.append(
                {
                    "trial_id": trial_id,
                    "num_runs": NUM_STOCHASTIC_RUNS,
                    "successes": successes,
                    "success_rate": successes / NUM_STOCHASTIC_RUNS,
                    "num_distinct_paths": len(clusters),
                    "path_percentages": percentages,
                }
            )

        all_results[scenario_name] = scenario_results

        # Print scenario summary
        avg_success = np.mean([r["success_rate"] for r in scenario_results])
        avg_paths = np.mean([r["num_distinct_paths"] for r in scenario_results])
        print(f"\n   📊 {scenario_name} Summary:")
        print(f"      Avg Success Rate: {100*avg_success:.1f}%")
        print(f"      Avg Distinct Paths: {avg_paths:.1f}")

    # Save results JSON
    results_path = output_dir / "stochastic_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "config": {
                    "num_stochastic_runs": NUM_STOCHASTIC_RUNS,
                    "num_trials_per_scenario": NUM_TRIALS_PER_SCENARIO,
                    "cluster_tolerance": CLUSTER_TOLERANCE,
                },
                "results": all_results,
            },
            f,
            indent=2,
        )

    print(f"\n{'='*70}")
    print(f"✅ COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📁 Output: {output_dir}")
    print(f"📊 Results: {results_path}")

    # Final summary
    print(f"\n📊 FINAL SUMMARY")
    print("-" * 50)
    for scenario_name, results in all_results.items():
        avg_success = np.mean([r["success_rate"] for r in results]) * 100
        avg_paths = np.mean([r["num_distinct_paths"] for r in results])
        print(
            f"   {scenario_name:10s}: {avg_success:5.1f}% success, {avg_paths:.1f} distinct paths"
        )


if __name__ == "__main__":
    main()
