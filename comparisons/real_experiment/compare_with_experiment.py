"""
Special Comparison with Real Experiment Results
================================================

Compare VGA Stochastic and DRL-VGA against real human experiment results
on SPECIFIC trials from the VGA paper.

Target Experiments:
- MOSP A: Row 183, Exp #184
- MOSP B: Row 23, Exp #24
- MOSP C: Row 153, Exp #154
- MOSP D: Row 160, Exp #161

Generates visualizations similar to the paper's Figure 7:
- Multiple colored paths with percentages
- Obstacles with numbers
- Comparison: Experiment vs VGA Stochastic vs DRL-VGA
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import datetime
import json
import sys
from collections import defaultdict

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "vga_upl_baseline"))
sys.path.insert(0, str(PROJECT_ROOT / "vga_upl_baseline" / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "vga_upl_baseline" / "data_loading"))
sys.path.insert(0, str(PROJECT_ROOT / "drl_vga_experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "core"))

from models.vga_upl_planner_v4 import VGAUPLPlannerV4
from data_loading.vga_dataset import VGADatasetLoader
from vga_experimental_env import VGAExperimentalEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# ============================================================
# Configuration
# ============================================================

# Specific experiments to compare
TARGET_EXPERIMENTS = {
    "MOSP_A": {"row": 183, "exp_no": 184},
    "MOSP_B": {"row": 23, "exp_no": 24},
    "MOSP_C": {"row": 153, "exp_no": 154},
    "MOSP_D": {"row": 160, "exp_no": 161},
}

# Number of runs for stochastic comparison
NUM_RUNS = 100

# Path clustering tolerance
CLUSTER_TOLERANCE = 0.5  # meters

# Colors matching paper style
PATH_COLORS = [
    "#0000CD",  # Blue (most common)
    "#FF0000",  # Red
    "#00AA00",  # Green
    "#FFD700",  # Gold/Yellow
    "#00CED1",  # Cyan
    "#FF8C00",  # Orange
    "#800080",  # Purple
    "#FF1493",  # Pink
]

# Arena bounds (VGA paper)
ARENA_X_MIN = 0.0
ARENA_X_MAX = 10.0
ARENA_Y_MIN = -2.0
ARENA_Y_MAX = 2.0

# Simulation parameters
DT = 0.05
MAX_STEPS = 500
GOAL_TOLERANCE = 0.3
AGENT_RADIUS = 0.2
OBSTACLE_RADIUS = 0.25


# ============================================================
# Path Clustering
# ============================================================


def trajectory_to_key_points(
    trajectory: np.ndarray, num_points: int = 10
) -> np.ndarray:
    """Reduce trajectory to key points for comparison."""
    if len(trajectory) < 2:
        return trajectory
    indices = np.linspace(0, len(trajectory) - 1, num_points, dtype=int)
    return trajectory[indices]


def trajectory_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """Calculate distance between two trajectories."""
    kp1 = trajectory_to_key_points(traj1)
    kp2 = trajectory_to_key_points(traj2)

    if len(kp1) != len(kp2):
        min_len = min(len(kp1), len(kp2))
        kp1 = kp1[:min_len]
        kp2 = kp2[:min_len]

    return np.mean(np.linalg.norm(kp1 - kp2, axis=1))


def cluster_trajectories(
    trajectories: list, tolerance: float = CLUSTER_TOLERANCE
) -> dict:
    """
    Cluster similar trajectories together.
    Returns dict with cluster info: {cluster_id: {'trajectories': [...], 'count': N, 'percentage': X}}
    """
    if not trajectories:
        return {}

    clusters = []
    assigned = [False] * len(trajectories)

    for i, traj in enumerate(trajectories):
        if assigned[i]:
            continue

        # Start new cluster
        cluster = [traj]
        assigned[i] = True

        # Find similar trajectories
        for j in range(i + 1, len(trajectories)):
            if assigned[j]:
                continue

            dist = trajectory_distance(traj, trajectories[j])
            if dist < tolerance:
                cluster.append(trajectories[j])
                assigned[j] = True

        clusters.append(cluster)

    # Sort by cluster size (most common first)
    clusters.sort(key=len, reverse=True)

    # Create result dict
    result = {}
    total = len(trajectories)
    for idx, cluster in enumerate(clusters):
        result[idx] = {
            "trajectories": cluster,
            "representative": cluster[0],  # Use first as representative
            "count": len(cluster),
            "percentage": len(cluster) / total * 100,
        }

    return result


# ============================================================
# Simulation Functions
# ============================================================


def run_vga_stochastic(trial_data: dict, num_runs: int = NUM_RUNS) -> list:
    """Run VGA stochastic planner multiple times."""
    trajectories = []

    start = trial_data["start_pos"]
    goal = trial_data["goal_pos"]
    obstacles = trial_data["obstacles"]

    for run in range(num_runs):
        # Create new planner for each run (to reset stochastic decisions)
        planner = VGAUPLPlannerV4(
            use_probabilistic=True,  # Stochastic!
            desired_speed=1.4,  # Typical walking speed
        )

        # Run simulation
        result = planner.simulate(
            start_pos=start, goal_pos=goal, max_steps=MAX_STEPS, obstacles=obstacles
        )

        # Extract trajectory
        trajectory = np.array(result.positions)
        trajectories.append(trajectory)

    return trajectories


def run_drl_vga(trial_data: dict, model, vec_env, num_runs: int = NUM_RUNS) -> list:
    """Run DRL-VGA model multiple times."""
    trajectories = []

    start = np.array(trial_data["start_pos"], dtype=np.float64)
    goal = np.array(trial_data["goal_pos"], dtype=np.float64)
    obstacles = trial_data["obstacles"]

    # Convert obstacles to env format: list of (x, y, r) tuples
    env_obstacles = [
        (obs["position"][0], obs["position"][1], obs["radius"]) for obs in obstacles
    ]

    # Get the underlying environment (unwrapped through VecNormalize -> DummyVecEnv)
    base_env = vec_env.venv.envs[0]

    for run in range(num_runs):
        # First do a normal reset to initialize everything properly
        _ = vec_env.reset()

        # Then override with our specific scenario
        base_env.steps = 0
        base_env.agent_pos = start.copy()
        base_env.goal_pos = goal.copy()
        base_env.agent_heading = np.arctan2(goal[1] - start[1], goal[0] - start[0])
        base_env.agent_vel = np.zeros(2, dtype=np.float64)
        base_env.obstacles = env_obstacles
        base_env.collision_count = 0
        base_env.total_distance_traveled = 0.0
        base_env.previous_distance_to_goal = np.linalg.norm(goal - start)
        base_env.desired_speed = 1.4  # Typical walking speed

        trajectory = [base_env.agent_pos.copy()]

        for step in range(MAX_STEPS):
            # Get raw observation from base env
            raw_obs = base_env._get_observation()

            # Normalize using VecNormalize stats
            norm_obs = vec_env.normalize_obs(raw_obs.reshape(1, -1))

            # Get action from model
            action, _ = model.predict(norm_obs, deterministic=True)

            # Step the BASE environment directly (not vec_env)
            obs, reward, terminated, truncated, info = base_env.step(action[0])

            trajectory.append(base_env.agent_pos.copy())

            if terminated or truncated:
                break

        trajectories.append(np.array(trajectory))

    return trajectories


# ============================================================
# Visualization Functions
# ============================================================


def create_path_visualization(
    trajectories: list,
    obstacles: list,
    start: np.ndarray,
    goal: np.ndarray,
    title: str,
    output_path: Path,
):
    """
    Create visualization matching the paper's style.
    Shows clustered paths with percentages.
    """
    # Cluster trajectories
    clusters = cluster_trajectories(trajectories)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))

    # Set arena bounds
    ax.set_xlim(ARENA_X_MIN - 0.5, ARENA_X_MAX + 0.5)
    ax.set_ylim(ARENA_Y_MIN - 0.5, ARENA_Y_MAX + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X-coordinate (m)", fontsize=12)
    ax.set_ylabel("Y-coordinate (m)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Draw obstacles with numbers
    for idx, obs in enumerate(obstacles):
        x, y = obs["position"]
        r = obs["radius"]

        # Draw obstacle
        circle = patches.Circle(
            (x, y),
            r,
            fill=True,
            facecolor="gray",
            edgecolor="black",
            linewidth=2,
            alpha=0.8,
            zorder=10,
        )
        ax.add_patch(circle)

        # Add number
        ax.text(
            x,
            y,
            str(idx),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
            zorder=11,
        )

    # Draw start and goal
    ax.plot(start[0], start[1], "ko", markersize=15, zorder=12)
    ax.annotate(
        "O",
        (start[0], start[1]),
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="white",
        zorder=13,
    )

    ax.plot(goal[0], goal[1], "kx", markersize=15, markeredgewidth=3, zorder=12)

    # Draw paths by cluster
    legend_elements = []

    for cluster_id, cluster_info in clusters.items():
        if cluster_id >= len(PATH_COLORS):
            break  # Limit to available colors

        color = PATH_COLORS[cluster_id]
        percentage = cluster_info["percentage"]

        # Draw all trajectories in this cluster
        for traj in cluster_info["trajectories"]:
            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.5, alpha=0.6)

        # Draw representative trajectory thicker
        rep = cluster_info["representative"]
        ax.plot(rep[:, 0], rep[:, 1], color=color, linewidth=3, alpha=0.9)

        # Add to legend
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=3, label=f"{percentage:.1f}%")
        )

    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return clusters


def create_combined_comparison(
    scenario: str,
    trial_data: dict,
    vga_clusters: dict,
    drl_clusters: dict,
    output_path: Path,
):
    """
    Create 2-column comparison: VGA Stochastic | DRL-VGA
    (Experiment data would be 3rd column if available)
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    obstacles = trial_data["obstacles"]
    start = trial_data["start_pos"]
    goal = trial_data["goal_pos"]

    titles = ["VGA Stochastic", "DRL-VGA"]
    all_clusters = [vga_clusters, drl_clusters]

    for ax, title, clusters in zip(axes, titles, all_clusters):
        # Set arena bounds
        ax.set_xlim(ARENA_X_MIN - 0.5, ARENA_X_MAX + 0.5)
        ax.set_ylim(ARENA_Y_MIN - 0.5, ARENA_Y_MAX + 0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("X-coordinate (m)", fontsize=12)
        ax.set_ylabel("Y-coordinate (m)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Draw obstacles
        for idx, obs in enumerate(obstacles):
            x, y = obs["position"]
            r = obs["radius"]
            circle = patches.Circle(
                (x, y),
                r,
                fill=True,
                facecolor="gray",
                edgecolor="black",
                linewidth=2,
                alpha=0.8,
                zorder=10,
            )
            ax.add_patch(circle)
            ax.text(
                x,
                y,
                str(idx),
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white",
                zorder=11,
            )

        # Draw start and goal
        ax.plot(start[0], start[1], "ko", markersize=12, zorder=12)
        ax.plot(goal[0], goal[1], "kx", markersize=12, markeredgewidth=3, zorder=12)

        # Draw paths
        legend_elements = []
        for cluster_id, cluster_info in clusters.items():
            if cluster_id >= len(PATH_COLORS):
                break

            color = PATH_COLORS[cluster_id]
            percentage = cluster_info["percentage"]

            for traj in cluster_info["trajectories"]:
                ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.5, alpha=0.5)

            rep = cluster_info["representative"]
            ax.plot(rep[:, 0], rep[:, 1], color=color, linewidth=3, alpha=0.9)

            legend_elements.append(
                Line2D([0], [0], color=color, linewidth=3, label=f"{percentage:.1f}%")
            )

        ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    fig.suptitle(f"{scenario} - Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================


def main():
    print("=" * 70)
    print("SPECIAL COMPARISON WITH REAL EXPERIMENT")
    print("=" * 70)
    print()
    print("Comparing VGA Stochastic and DRL-VGA on specific experiments:")
    for scenario, info in TARGET_EXPERIMENTS.items():
        print(f"  - {scenario}: Row {info['row']}, Exp #{info['exp_no']}")
    print()

    # Setup paths
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    data_root = PROJECT_ROOT / "data" / "VGA-Experimental-Data"

    # Load dataset
    print("[1] Loading dataset...")
    dataset = VGADatasetLoader(data_root=str(data_root))

    # Load DRL model WITH VecNormalize (critical!)
    print("[2] Loading DRL model with VecNormalize...")
    drl_model_path = (
        PROJECT_ROOT / "drl_vga_experiments" / "models" / "vga_drl_final.zip"
    )
    vecnorm_path = (
        PROJECT_ROOT
        / "drl_vga_experiments"
        / "models"
        / "vga_drl_final_vecnormalize.pkl"
    )

    if not drl_model_path.exists():
        print(f"[ERROR] DRL model not found: {drl_model_path}")
        return

    if not vecnorm_path.exists():
        print(f"[ERROR] VecNormalize not found: {vecnorm_path}")
        return

    # Create base environment
    base_env = VGAExperimentalEnv(
        data_root=str(data_root), scenario="MOSP_A"  # Will be overridden
    )

    # Wrap in DummyVecEnv and load VecNormalize
    dummy_env = DummyVecEnv([lambda: base_env])
    env = VecNormalize.load(str(vecnorm_path), dummy_env)
    env.training = False  # Don't update stats during evaluation
    env.norm_reward = False

    # Load the model
    drl_model = PPO.load(str(drl_model_path))

    print(f"[3] Running {NUM_RUNS} simulations per method per experiment...")
    print()

    results = {}

    for scenario, exp_info in TARGET_EXPERIMENTS.items():
        print(f"\n{'='*50}")
        print(f"Processing: {scenario} (Exp #{exp_info['exp_no']})")
        print(f"{'='*50}")

        # Create output folder
        scenario_dir = output_dir / f"{scenario}_exp{exp_info['exp_no']}"
        scenario_dir.mkdir(exist_ok=True)

        # Load trial data
        case = scenario.split("_")[1]  # A, B, C, or D
        trials = dataset.load_mosp(case)

        # Find the specific trial by row index
        trial_idx = exp_info["row"]
        if trial_idx >= len(trials):
            print(
                f"  [WARN] Trial index {trial_idx} out of range (max {len(trials)-1})"
            )
            trial_idx = min(trial_idx, len(trials) - 1)

        trial = trials[trial_idx]

        trial_data = {
            "start_pos": trial.initial_pos.copy(),
            "goal_pos": trial.final_pos.copy(),
            "obstacles": trial.obstacles.copy(),
        }

        print(f"  Start: {trial_data['start_pos']}")
        print(f"  Goal: {trial_data['goal_pos']}")
        print(f"  Obstacles: {len(trial_data['obstacles'])}")

        # Run VGA Stochastic
        print(f"  Running VGA Stochastic ({NUM_RUNS} runs)...")
        vga_trajectories = run_vga_stochastic(trial_data, NUM_RUNS)
        print(f"    Generated {len(vga_trajectories)} trajectories")

        # Run DRL-VGA
        print(f"  Running DRL-VGA ({NUM_RUNS} runs)...")
        drl_trajectories = run_drl_vga(trial_data, drl_model, env, NUM_RUNS)
        print(f"    Generated {len(drl_trajectories)} trajectories")

        # Create visualizations
        print("  Creating visualizations...")

        # VGA Stochastic visualization
        vga_clusters = create_path_visualization(
            vga_trajectories,
            trial_data["obstacles"],
            trial_data["start_pos"],
            trial_data["goal_pos"],
            f"{scenario} - VGA Stochastic",
            scenario_dir / "vga_stochastic_paths.png",
        )

        # DRL-VGA visualization
        drl_clusters = create_path_visualization(
            drl_trajectories,
            trial_data["obstacles"],
            trial_data["start_pos"],
            trial_data["goal_pos"],
            f"{scenario} - DRL-VGA",
            scenario_dir / "drl_vga_paths.png",
        )

        # Combined comparison
        create_combined_comparison(
            scenario,
            trial_data,
            vga_clusters,
            drl_clusters,
            scenario_dir / "combined_comparison.png",
        )

        # Save results data
        results[scenario] = {
            "exp_no": exp_info["exp_no"],
            "row": exp_info["row"],
            "start_pos": trial_data["start_pos"].tolist(),
            "goal_pos": trial_data["goal_pos"].tolist(),
            "num_obstacles": len(trial_data["obstacles"]),
            "vga_clusters": {
                k: {"count": v["count"], "percentage": v["percentage"]}
                for k, v in vga_clusters.items()
            },
            "drl_clusters": {
                k: {"count": v["count"], "percentage": v["percentage"]}
                for k, v in drl_clusters.items()
            },
        }

        # Print summary
        print(f"\n  VGA Stochastic Path Distribution:")
        for cid, info in list(vga_clusters.items())[:5]:
            print(f"    Path {cid+1}: {info['percentage']:.1f}%")

        print(f"\n  DRL-VGA Path Distribution:")
        for cid, info in list(drl_clusters.items())[:5]:
            print(f"    Path {cid+1}: {info['percentage']:.1f}%")

    # Save overall results
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
