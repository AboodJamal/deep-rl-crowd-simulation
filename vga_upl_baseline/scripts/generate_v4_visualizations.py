"""
VGA+UPL V4 Visualization Generator
===================================

Generates high-quality videos and images with:
- Agent trajectory
- Direction arrow showing where agent is looking/heading
- Subgoal visualization
- Obstacle clearance zones
- Metrics overlay

Output: 10 videos + 10 images per scenario (SOSP, MOSP_A, MOSP_B, MOSP_C, MOSP_D)
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter
from datetime import datetime
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings("ignore")

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"),
)

from models.vga_upl_planner_v4 import VGAUPLPlannerV4
from data_loading.vga_dataset import VGADatasetLoader

# Style settings
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "agent": "#2196F3",  # Blue
    "agent_trail": "#64B5F6",  # Light blue
    "goal": "#4CAF50",  # Green
    "obstacle": "#F44336",  # Red
    "obstacle_zone": "#FFCDD2",  # Light red
    "danger_zone": "#FFCDD2",  # Light red (alias)
    "subgoal": "#FF9800",  # Orange
    "direction_arrow": "#1565C0",  # Dark blue
    "clearance": "#E3F2FD",  # Very light blue
    "start": "#9C27B0",  # Purple
}


def get_best_infobox_position(ax, positions, start, goal, obstacles):
    """
    Determine the best corner for the info box to avoid overlapping with
    trajectory, goal, start, and obstacles.

    Uses AXES coordinates (0-1 range) for the actual text placement.
    Returns: (x, y, ha, va) for ax.text positioning with transform=ax.transAxes
    """
    # Get plot bounds
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Convert data coordinates to normalized axes coordinates (0-1)
    def to_axes_coords(data_x, data_y):
        ax_x = (data_x - x_min) / (x_max - x_min)
        ax_y = (data_y - y_min) / (y_max - y_min)
        return ax_x, ax_y

    # Define corner regions in AXES coordinates (0-1 range)
    # Info box is approximately 18% width and 35% height of plot
    box_width = 0.22
    box_height = 0.38

    corner_regions = {
        "top-left": (0, box_width, 1 - box_height, 1),
        "top-right": (1 - box_width, 1, 1 - box_height, 1),
        "bottom-left": (0, box_width, 0, box_height),
        "bottom-right": (1 - box_width, 1, 0, box_height),
    }

    def point_in_region(ax_x, ax_y, region):
        """Check if axes-coord point is in region (x1, x2, y1, y2)"""
        return region[0] <= ax_x <= region[1] and region[2] <= ax_y <= region[3]

    def count_conflicts(corner_name):
        """Count how many important elements are in this corner region"""
        region = corner_regions[corner_name]
        conflicts = 0

        # Check goal (in axes coordinates)
        gx, gy = to_axes_coords(goal[0], goal[1])
        if point_in_region(gx, gy, region):
            conflicts += 100  # VERY high weight for goal

        # Check start (in axes coordinates)
        sx, sy = to_axes_coords(start[0], start[1])
        if point_in_region(sx, sy, region):
            conflicts += 80  # High weight for start

        # Check obstacles (in axes coordinates)
        for obs in obstacles:
            ox, oy = to_axes_coords(obs["position"][0], obs["position"][1])
            if point_in_region(ox, oy, region):
                conflicts += 30

        # Check trajectory points (sample points, in axes coordinates)
        if len(positions) > 0:
            sample_rate = max(1, len(positions) // 15)
            for i in range(0, len(positions), sample_rate):
                px, py = to_axes_coords(positions[i][0], positions[i][1])
                if point_in_region(px, py, region):
                    conflicts += 2

        return conflicts

    # Find corner with least conflicts
    corner_conflicts = {name: count_conflicts(name) for name in corner_regions}
    best_corner = min(corner_conflicts, key=corner_conflicts.get)

    # Return transform coordinates and alignment
    corner_to_transform = {
        "top-left": (0.02, 0.98, "left", "top"),
        "top-right": (0.98, 0.98, "right", "top"),
        "bottom-left": (0.02, 0.02, "left", "bottom"),
        "bottom-right": (0.98, 0.02, "right", "bottom"),
    }

    return corner_to_transform[best_corner]

    # Find corner with least conflicts
    corner_conflicts = {name: count_conflicts(name) for name in corners}
    best_corner = min(corner_conflicts, key=corner_conflicts.get)

    # Return transform coordinates and alignment
    corner_to_transform = {
        "top-left": (0.02, 0.98, "left", "top"),
        "top-right": (0.98, 0.98, "right", "top"),
        "bottom-left": (0.02, 0.02, "left", "bottom"),
        "bottom-right": (0.98, 0.02, "right", "bottom"),
    }

    return corner_to_transform[best_corner]


def create_video_frame(
    ax,
    pos,
    vel,
    positions_history,
    subgoal,
    obstacles,
    start,
    goal,
    agent_radius,
    frame_idx,
    total_frames,
    metrics,
    scenario_name,
    trial_id,
):
    """Create a single frame for video/image."""
    ax.clear()

    # Set up plot
    all_x = [start[0], goal[0]] + [o["position"][0] for o in obstacles]
    all_y = [start[1], goal[1]] + [o["position"][1] for o in obstacles]

    margin = 2.0
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin

    # Make it wider for typical scenario
    if x_max - x_min < 12:
        x_center = (x_min + x_max) / 2
        x_min, x_max = x_center - 7, x_center + 7

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")

    # Background grid
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Draw obstacles with clearance zones
    # Danger zone = agent_radius + min_clearance = 0.2 + 0.02 = 0.22m (visualization only)
    danger_zone_size = agent_radius + 0.02  # Matches actual collision avoidance margin

    for obs in obstacles:
        obs_pos = obs["position"]
        obs_rad = obs.get("radius", 0.25)

        # Danger zone (shows actual safety margin of 0.22m)
        danger_zone = Circle(
            obs_pos,
            obs_rad + danger_zone_size,
            color=COLORS["obstacle_zone"],
            alpha=0.3,
            zorder=1,
        )
        ax.add_patch(danger_zone)

        # Obstacle
        obstacle = Circle(
            obs_pos,
            obs_rad,
            color=COLORS["obstacle"],
            alpha=0.8,
            zorder=5,
            edgecolor="darkred",
            linewidth=2,
        )
        ax.add_patch(obstacle)

    # Draw start position
    start_marker = Circle(start, 0.15, color=COLORS["start"], alpha=0.6, zorder=3)
    ax.add_patch(start_marker)
    ax.annotate(
        "START",
        start,
        textcoords="offset points",
        xytext=(0, -20),
        ha="center",
        fontsize=8,
        color=COLORS["start"],
        fontweight="bold",
    )

    # Draw goal
    goal_marker = Circle(
        goal,
        0.25,
        color=COLORS["goal"],
        alpha=0.7,
        zorder=3,
        edgecolor="darkgreen",
        linewidth=2,
    )
    ax.add_patch(goal_marker)
    ax.annotate(
        "GOAL",
        goal,
        textcoords="offset points",
        xytext=(0, 20),
        ha="center",
        fontsize=10,
        color="darkgreen",
        fontweight="bold",
    )

    # Draw trajectory trail with gradient
    if len(positions_history) > 1:
        points = np.array(positions_history)
        # Create segments with color gradient
        segments = []
        colors = []
        for i in range(len(points) - 1):
            segments.append([points[i], points[i + 1]])
            # Fade from light to current color
            alpha = 0.3 + 0.7 * (i / len(points))
            colors.append((*plt.cm.Blues(0.6)[:3], alpha))

        lc = LineCollection(segments, colors=colors, linewidths=2, zorder=4)
        ax.add_collection(lc)

    # Draw subgoal if not at goal
    if subgoal is not None and np.linalg.norm(subgoal - goal) > 0.3:
        subgoal_marker = Circle(
            subgoal,
            0.12,
            color=COLORS["subgoal"],
            alpha=0.8,
            zorder=6,
            edgecolor="darkorange",
            linewidth=1.5,
        )
        ax.add_patch(subgoal_marker)
        ax.annotate(
            "subgoal",
            subgoal,
            textcoords="offset points",
            xytext=(10, 10),
            ha="left",
            fontsize=7,
            color="darkorange",
        )

        # Dashed line to subgoal
        ax.plot(
            [pos[0], subgoal[0]],
            [pos[1], subgoal[1]],
            "--",
            color=COLORS["subgoal"],
            alpha=0.5,
            linewidth=1.5,
            zorder=4,
        )

    # Draw agent
    agent_circle = Circle(
        pos,
        agent_radius,
        color=COLORS["agent"],
        alpha=0.9,
        zorder=10,
        edgecolor="darkblue",
        linewidth=2,
    )
    ax.add_patch(agent_circle)

    # Draw direction arrow (showing where agent is heading)
    vel_mag = np.linalg.norm(vel)
    if vel_mag > 0.01:
        direction = vel / vel_mag
        arrow_length = 0.6  # Fixed length for visibility
        arrow_end = pos + direction * arrow_length

        # Create fancy arrow
        arrow = FancyArrowPatch(
            pos,
            arrow_end,
            arrowstyle="-|>",
            mutation_scale=15,
            color=COLORS["direction_arrow"],
            linewidth=3,
            zorder=11,
        )
        ax.add_patch(arrow)

        # Add small "looking" indicator at agent center
        ax.plot(pos[0], pos[1], "o", color="white", markersize=4, zorder=12)

    # Metrics box
    progress = frame_idx / max(total_frames - 1, 1) * 100
    dist_to_goal = np.linalg.norm(pos - goal)

    metrics_text = (
        f"Scenario: {scenario_name}\n"
        f"Trial: {trial_id}\n"
        f"─────────────\n"
        f"Progress: {progress:.0f}%\n"
        f"Distance to goal: {dist_to_goal:.2f}m\n"
        f"Speed: {vel_mag:.2f} m/s\n"
        f"─────────────\n"
        f"Path length: {metrics.get('path_length', 0):.2f}m\n"
        f"Efficiency: {metrics.get('path_efficiency', 0)*100:.1f}%\n"
        f"Subgoals: {metrics.get('num_subgoals', 0)}"
    )

    # Get best position for info box (avoid overlapping with trajectory/goal)
    box_x, box_y, ha, va = get_best_infobox_position(
        ax, positions_history, start, goal, obstacles
    )

    props = dict(
        boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="gray"
    )
    ax.text(
        box_x,
        box_y,
        metrics_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment=va,
        horizontalalignment=ha,
        fontfamily="monospace",
        bbox=props,
    )

    # Title
    ax.set_title(
        f"VGA+UPL V4 Navigation - {scenario_name}", fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")


def generate_video(
    result,
    obstacles,
    start,
    goal,
    scenario_name,
    trial_id,
    output_path,
    agent_radius=0.2,
    fps=20,
):
    """Generate video for a single trial."""
    positions = result.positions
    velocities = result.velocities
    subgoals = result.metadata.get("subgoals_per_step", [])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

    # Subsample if too many frames
    total_frames = len(positions)
    max_frames = 300  # Cap at 15 seconds at 20fps

    if total_frames > max_frames:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    else:
        indices = np.arange(total_frames)

    def animate(i):
        idx = indices[i]
        pos = positions[idx]
        vel = velocities[idx] if idx < len(velocities) else np.zeros(2)
        history = positions[: idx + 1]
        subgoal = np.array(subgoals[idx]) if idx < len(subgoals) else goal

        create_video_frame(
            ax,
            pos,
            vel,
            history,
            subgoal,
            obstacles,
            start,
            goal,
            agent_radius,
            i,
            len(indices),
            result.metadata,
            scenario_name,
            trial_id,
        )

    anim = FuncAnimation(
        fig, animate, frames=len(indices), interval=1000 / fps, blit=False
    )

    # Save video
    try:
        writer = FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(output_path, writer=writer)
    except Exception as e:
        print(f"   FFmpeg error, trying alternative: {e}")
        # Try with pillow
        anim.save(output_path.replace(".mp4", ".gif"), writer="pillow", fps=fps)

    plt.close(fig)


def generate_image(
    result,
    obstacles,
    start,
    goal,
    scenario_name,
    trial_id,
    output_path,
    agent_radius=0.2,
    frame_ratio=0.7,
):
    """Generate a STATIC image showing the FULL trajectory (not mid-point snapshot)."""
    positions = result.positions
    velocities = result.velocities
    metadata = result.metadata

    # Calculate data bounds first
    all_x = np.concatenate([positions[:, 0], [start[0], goal[0]]])
    all_y = np.concatenate([positions[:, 1], [start[1], goal[1]]])
    for obs in obstacles:
        all_x = np.append(all_x, obs["position"][0])
        all_y = np.append(all_y, obs["position"][1])

    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()

    # Check if environment is narrow (Y range < 3m)
    # If so, expand Y to make room for info box OUTSIDE the main data area
    is_narrow = y_range < 3.0

    if is_narrow:
        # For narrow environments, add extra space at top for info box
        fig, ax = plt.subplots(figsize=(14, 10), dpi=150)  # Taller figure
        x_margin = max(1.5, x_range * 0.15)
        y_margin_bottom = 0.8
        y_margin_top = 2.5  # Extra space at top for info box
    else:
        fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
        x_margin = max(1.5, x_range * 0.15)
        y_margin_bottom = max(1.0, y_range * 0.2)
        y_margin_top = max(1.0, y_range * 0.2)

    ax.set_facecolor("#f8f9fa")

    # SET AXIS LIMITS
    ax.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
    ax.set_ylim(all_y.min() - y_margin_bottom, all_y.max() + y_margin_top)
    ax.set_aspect("equal")

    # Draw obstacles with danger zones
    # Danger zone = agent_radius + min_clearance = 0.2 + 0.02 = 0.22m (visualization only)
    danger_zone_size = agent_radius + 0.02  # 0.22m - matches actual collision avoidance

    for obs in obstacles:
        obs_pos = np.array(obs["position"])
        obs_rad = obs.get("radius", 0.25)

        # Danger zone (light red halo) - shows actual safety margin
        danger_zone = Circle(
            obs_pos,
            obs_rad + danger_zone_size,
            color=COLORS["danger_zone"],
            alpha=0.2,
            zorder=1,
        )
        ax.add_patch(danger_zone)

        # Obstacle body
        obstacle = Circle(
            obs_pos,
            obs_rad,
            color=COLORS["obstacle"],
            alpha=0.9,
            zorder=4,
            edgecolor="darkred",
            linewidth=1,
        )
        ax.add_patch(obstacle)

    # Draw FULL trajectory as smooth connected line
    # Use simple connected line (most reliable) with slight smoothing
    try:
        if len(positions) > 3:
            # Simple moving average smoothing for visual appeal
            window = 3
            smooth_x = np.convolve(
                positions[:, 0], np.ones(window) / window, mode="valid"
            )
            smooth_y = np.convolve(
                positions[:, 1], np.ones(window) / window, mode="valid"
            )
            ax.plot(
                smooth_x, smooth_y, color="#3498db", linewidth=2.5, zorder=3, alpha=0.9
            )
        else:
            ax.plot(
                positions[:, 0],
                positions[:, 1],
                color="#3498db",
                linewidth=2.5,
                zorder=3,
                alpha=0.9,
            )
    except Exception:
        # Fallback: draw as single connected line
        ax.plot(
            positions[:, 0],
            positions[:, 1],
            color="#3498db",
            linewidth=2.5,
            zorder=3,
            alpha=0.9,
        )

    # Draw velocity arrows along trajectory (every N points)
    arrow_interval = max(1, len(positions) // 15)  # ~15 arrows total
    for i in range(0, len(positions), arrow_interval):
        pos = positions[i]
        vel = velocities[i] if i < len(velocities) else np.zeros(2)
        speed = np.linalg.norm(vel)

        if speed > 0.1:
            arrow_len = 0.4
            direction = vel / speed
            ax.annotate(
                "",
                xy=(
                    pos[0] + direction[0] * arrow_len,
                    pos[1] + direction[1] * arrow_len,
                ),
                xytext=(pos[0], pos[1]),
                arrowprops=dict(
                    arrowstyle="->", color="#3498db", lw=1.5, mutation_scale=10
                ),
                zorder=5,
            )

    # Start marker (purple circle)
    start_marker = Circle(
        start,
        0.15,
        color="#9b59b6",
        alpha=0.9,
        zorder=6,
        edgecolor="white",
        linewidth=2,
    )
    ax.add_patch(start_marker)
    ax.annotate(
        "START",
        start,
        textcoords="offset points",
        xytext=(10, -20),
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="#9b59b6",
    )

    # Goal marker (green star)
    ax.plot(
        goal[0],
        goal[1],
        "*",
        color=COLORS["goal"],
        markersize=25,
        markeredgecolor="white",
        markeredgewidth=2,
        zorder=6,
    )
    ax.annotate(
        "GOAL",
        goal,
        textcoords="offset points",
        xytext=(10, 15),
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=COLORS["goal"],
    )

    # Final position marker (diamond)
    final_pos = positions[-1]
    ax.plot(
        final_pos[0],
        final_pos[1],
        "D",
        color="#e67e22",
        markersize=12,
        markeredgecolor="white",
        markeredgewidth=2,
        zorder=7,
    )

    # Success indicator
    success = result.success
    status_color = "#27ae60" if success else "#e74c3c"
    status_text = "✓ SUCCESS" if success else "✗ FAILED"

    # Info box with metrics
    info_text = (
        f"Scenario: {scenario_name}\n"
        f"Trial: {trial_id}\n"
        f"{'─' * 15}\n"
        f"Status: {status_text}\n"
        f"{'─' * 15}\n"
        f"Travel time: {metadata.get('travel_time', 0):.2f}s\n"
        f"Path length: {metadata.get('path_length', 0):.2f}m\n"
        f"Efficiency: {metadata.get('path_efficiency', 0)*100:.1f}%\n"
        f"Avg speed: {metadata.get('average_speed', 0):.2f} m/s\n"
        f"{'─' * 15}\n"
        f"Collisions: {metadata.get('num_collisions', 0)}\n"
        f"Min clearance: {metadata.get('min_clearance', 0):.3f}m\n"
        f"Subgoals: {metadata.get('num_subgoals', 0)}"
    )

    # For narrow environments, always place info box at top-left (we added extra space)
    if is_narrow:
        box_x, box_y, ha, va = 0.02, 0.98, "left", "top"
    else:
        # Get best position for info box (avoid overlapping with trajectory/goal)
        box_x, box_y, ha, va = get_best_infobox_position(
            ax, positions, start, goal, obstacles
        )

    props = dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        alpha=0.95,
        edgecolor=status_color,
        linewidth=2,
    )
    ax.text(
        box_x,
        box_y,
        info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment=va,
        horizontalalignment=ha,
        fontfamily="monospace",
        bbox=props,
    )

    # Legend - position opposite to info box
    legend_loc_map = {
        ("left", "top"): "lower right",
        ("right", "top"): "lower left",
        ("left", "bottom"): "upper right",
        ("right", "bottom"): "upper left",
    }
    legend_loc = legend_loc_map.get((ha, va), "upper right")

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#9b59b6",
            markersize=10,
            label="Start",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=COLORS["goal"],
            markersize=15,
            label="Goal",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="#e67e22",
            markersize=8,
            label="Final Position",
        ),
        Line2D([0], [0], color="#3498db", linewidth=2, label="Trajectory"),
        Circle((0, 0), 0.1, color=COLORS["obstacle"], label="Obstacle"),
    ]
    ax.legend(
        handles=legend_elements,
        loc=legend_loc,
        fontsize=9,
        framealpha=0.95,
        edgecolor="gray",
    )

    # Axis styling
    ax.set_xlabel("X (meters)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Y (meters)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"VGA+UPL V4 - Full Trajectory - {scenario_name}",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    print("=" * 70)
    print("VGA+UPL V4 - Visualization & Metrics Generator")
    print("=" * 70)

    # Setup paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Data is in project root's data folder
    project_root = os.path.dirname(base_path)
    data_root = os.path.join(project_root, "data", "VGA-Experimental-Data")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_path, "results", f"vga_v4_final_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print(f"\n📁 Loading data from: {data_root}")
    loader = VGADatasetLoader(data_root)
    all_data = loader.load_all()

    # Initialize planner
    planner = VGAUPLPlannerV4(
        use_probabilistic=False,
        agent_radius=0.2,
        min_clearance=0.02,
        desired_speed=1.34,
        dt=0.05,
    )

    scenario_map = {
        "SOSP": "sosp",
        "MOSP_A": "mosp_a",
        "MOSP_B": "mosp_b",
        "MOSP_C": "mosp_c",
        "MOSP_D": "mosp_d",
    }

    all_metrics = {}

    for scenario_name, key in scenario_map.items():
        print(f"\n{'='*50}")
        print(f"📍 Processing {scenario_name}")
        print(f"{'='*50}")

        trials = all_data.get(key, [])
        if not trials:
            print(f"   No trials found!")
            continue

        # Create scenario directories
        video_dir = os.path.join(output_dir, "videos", scenario_name)
        image_dir = os.path.join(output_dir, "images", scenario_name)
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)

        # Select 10 diverse trials for visualization (spread across the dataset)
        n_viz = min(10, len(trials))
        viz_indices = set(np.linspace(0, len(trials) - 1, n_viz, dtype=int))

        scenario_results = []

        # ============================================
        # EVALUATE ALL TRIALS (for comprehensive metrics)
        # ============================================
        print(f"\n   📊 Evaluating ALL {len(trials)} trials for metrics...")

        for trial_idx, trial in enumerate(trials):
            trial_id = trial.trial_id
            start = np.array(trial.initial_pos)
            goal = np.array(trial.final_pos)
            obstacles = trial.obstacles

            # Run simulation
            result = planner.simulate(start, goal, obstacles=obstacles, max_steps=2000)

            # Collect metrics
            meta = result.metadata.copy()
            meta["success"] = result.success
            meta["trial_id"] = trial_id
            # Remove large arrays for JSON
            meta.pop("subgoal_history", None)
            meta.pop("subgoals_per_step", None)
            scenario_results.append(meta)

            # ============================================
            # GENERATE VISUALIZATIONS (only for selected 10 trials)
            # ============================================
            if trial_idx in viz_indices:
                viz_num = list(viz_indices).index(trial_idx) + 1
                print(f"\n   🎨 Visualization {viz_num}/10 (Trial ID: {trial_id})")

                status = "✅" if result.success else "❌"
                print(
                    f"      {status} Success: {result.success}, Steps: {len(result.positions)}"
                )

                # Generate video
                video_path = os.path.join(video_dir, f"trial_{trial_id:03d}.mp4")
                print(f"      🎬 Generating video...")
                try:
                    generate_video(
                        result,
                        obstacles,
                        start,
                        goal,
                        scenario_name,
                        trial_id,
                        video_path,
                        agent_radius=0.2,
                        fps=20,
                    )
                    print(f"         Saved: {video_path}")
                except Exception as e:
                    print(f"         Video error: {e}")

                # Generate image
                image_path = os.path.join(image_dir, f"trial_{trial_id:03d}.png")
                print(f"      📸 Generating image...")
                try:
                    generate_image(
                        result,
                        obstacles,
                        start,
                        goal,
                        scenario_name,
                        trial_id,
                        image_path,
                        agent_radius=0.2,
                    )
                    print(f"         Saved: {image_path}")
                except Exception as e:
                    print(f"         Image error: {e}")

            # Progress indicator for non-visualization trials
            elif (trial_idx + 1) % 50 == 0:
                print(f"      ... evaluated {trial_idx + 1}/{len(trials)} trials")

        print(f"\n   ✓ Completed {len(trials)} trials")

        # Calculate aggregate metrics from ALL trials
        all_metrics[scenario_name] = {
            "total_trials": len(scenario_results),
            "trials": scenario_results,
            "aggregate": {
                "success_rate": np.mean([r["success"] for r in scenario_results]),
                "avg_travel_time": np.mean(
                    [r["travel_time"] for r in scenario_results]
                ),
                "avg_path_length": np.mean(
                    [r["path_length"] for r in scenario_results]
                ),
                "avg_path_efficiency": np.mean(
                    [r["path_efficiency"] for r in scenario_results]
                ),
                "avg_num_subgoals": np.mean(
                    [r["num_subgoals"] for r in scenario_results]
                ),
                "collision_rate": np.mean(
                    [1 if r["num_collisions"] > 0 else 0 for r in scenario_results]
                ),
                "avg_direction_changes": np.mean(
                    [r["direction_changes"] for r in scenario_results]
                ),
                "avg_deviation": np.mean(
                    [r["average_deviation"] for r in scenario_results]
                ),
                "min_clearance": np.min([r["min_clearance"] for r in scenario_results]),
            },
        }

        print(f"\n   📊 {scenario_name} Summary (ALL {len(scenario_results)} trials):")
        print(
            f"      Success Rate: {all_metrics[scenario_name]['aggregate']['success_rate']*100:.1f}%"
        )
        print(
            f"      Avg Path Length: {all_metrics[scenario_name]['aggregate']['avg_path_length']:.2f}m"
        )
        print(
            f"      Avg Efficiency: {all_metrics[scenario_name]['aggregate']['avg_path_efficiency']*100:.1f}%"
        )

    # Save all metrics
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(x) for x in obj]
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    with open(metrics_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "planner": "VGA+UPL V4",
                "parameters": {
                    "agent_radius": 0.2,
                    "min_clearance": 0.02,
                    "desired_speed": 1.34,
                    "dt": 0.05,
                },
                "scenarios": convert(all_metrics),
            },
            f,
            indent=2,
        )

    # Save aggregate_metrics.json (summary only, no per-trial data)
    aggregate_metrics_path = os.path.join(output_dir, "aggregate_metrics.json")
    aggregate_summary = {
        "timestamp": datetime.now().isoformat(),
        "planner": "VGA+UPL V4",
        "parameters": {
            "agent_radius": 0.2,
            "min_clearance": 0.02,
            "desired_speed": 1.34,
            "dt": 0.05,
            "danger_zone_visualization": 0.22,  # agent_radius + min_clearance
        },
        "scenario_metrics": {},
    }

    for scenario_name, data in all_metrics.items():
        agg = data["aggregate"]
        aggregate_summary["scenario_metrics"][scenario_name] = {
            "success_rate": convert(agg["success_rate"]),
            "total_trials": len(data["trials"]),
            "successful_trials": sum(1 for t in data["trials"] if t["success"]),
            "travel_time_mean": convert(agg["avg_travel_time"]),
            "path_length_mean": convert(agg["avg_path_length"]),
            "path_efficiency_mean": convert(agg["avg_path_efficiency"]),
            "collision_rate": convert(agg["collision_rate"]),
            "num_collisions_mean": convert(
                np.mean([t["num_collisions"] for t in data["trials"]])
            ),
            "direction_changes_mean": convert(agg["avg_direction_changes"]),
            "oscillation_index_mean": convert(
                np.mean([t.get("oscillation_index", 0) for t in data["trials"]])
            ),
            "average_deviation_mean": convert(agg["avg_deviation"]),
            "max_deviation_mean": convert(
                np.mean([t["max_deviation"] for t in data["trials"]])
            ),
            "min_clearance_min": convert(agg["min_clearance"]),
            "average_clearance_mean": convert(
                np.mean([t["average_clearance"] for t in data["trials"]])
            ),
            "danger_zone_ratio_mean": convert(
                np.mean([t.get("danger_zone_ratio", 0) for t in data["trials"]])
            ),
            "num_subgoals_mean": convert(agg["avg_num_subgoals"]),
            "subgoal_switch_rate_mean": convert(
                np.mean([t.get("subgoal_switch_rate", 0) for t in data["trials"]])
            ),
        }

    with open(aggregate_metrics_path, "w") as f:
        json.dump(aggregate_summary, f, indent=2)

    print(f"\n{'='*70}")
    print("✅ COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"   📹 Videos: {output_dir}/videos/")
    print(f"   🖼️  Images: {output_dir}/images/")
    print(f"   📊 Metrics: {metrics_path}")
    print(f"   📊 Aggregate: {aggregate_metrics_path}")

    # Print final summary
    total_trials = sum(len(all_metrics[s]["trials"]) for s in all_metrics)
    total_success = sum(
        sum(1 for t in all_metrics[s]["trials"] if t["success"]) for s in all_metrics
    )

    print(f"\n📊 FINAL SUMMARY (Evaluated {total_trials} total trials)")
    print("-" * 60)
    for scenario in scenario_map.keys():
        if scenario in all_metrics:
            agg = all_metrics[scenario]["aggregate"]
            n_trials = len(all_metrics[scenario]["trials"])
            n_success = sum(1 for t in all_metrics[scenario]["trials"] if t["success"])
            print(
                f"   {scenario:10}: {agg['success_rate']*100:5.1f}% ({n_success}/{n_trials}), "
                f"Eff: {agg['avg_path_efficiency']*100:5.1f}%, "
                f"Subgoals: {agg['avg_num_subgoals']:.1f}"
            )
    print("-" * 60)
    print(
        f"   {'OVERALL':10}: {total_success/total_trials*100:5.1f}% ({total_success}/{total_trials})"
    )

    return output_dir


if __name__ == "__main__":
    main()
