"""
Generate Scenario Visualizations for VGA Dataset
=================================================
Creates visual diagrams showing obstacle layouts for each scenario.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Obstacle configurations from VGA experimental environment
OBSTACLE_CONFIGS = {
    "SOSP": [(5.000, 0.000)],
    "MOSP_A": [(5.465, -0.446), (7.264, 0.344), (3.983, 0.208), (3.932, -0.978)],
    "MOSP_B": [
        (5.358, -0.207),
        (4.950, 0.191),
        (7.108, 0.290),
        (2.892, -1.120),
        (4.850, -1.320),
        (4.200, 1.237),
        (7.550, -0.813),
    ],
    "MOSP_C": [
        (4.114, 0.527),
        (6.647, -0.736),
        (3.052, 0.644),
        (6.246, 1.222),
        (3.336, 1.490),
        (6.045, -0.469),
        (4.331, -0.845),
        (5.142, -1.456),
        (3.052, -0.653),
        (4.607, 0.946),
        (5.426, 0.092),
        (2.174, -0.360),
    ],
    "MOSP_D": [
        (4.911, 1.455),
        (5.465, -0.446),
        (4.970, -1.259),
        (6.324, 0.336),
        (6.675, -1.053),
        (5.418, 1.103),
        (7.264, 0.344),
        (6.252, -1.370),
        (5.923, 1.430),
        (3.983, 0.208),
        (3.932, -0.978),
        (2.847, 1.107),
        (6.932, 0.733),
        (3.102, -0.519),
        (2.686, -1.270),
        (4.495, -0.878),
    ],
}

# Arena bounds
ARENA_X_MIN = 0.0
ARENA_X_MAX = 10.0
ARENA_Y_MIN = -1.75
ARENA_Y_MAX = 1.75

OBSTACLE_RADIUS = 0.25
AGENT_RADIUS = 0.2


def create_scenario_visualization(scenario_name, obstacles, output_path):
    """Create a single scenario visualization."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Set arena bounds
    ax.set_xlim(ARENA_X_MIN - 0.5, ARENA_X_MAX + 0.5)
    ax.set_ylim(ARENA_Y_MIN - 0.5, ARENA_Y_MAX + 0.5)
    ax.set_aspect("equal")

    # Draw arena boundaries
    arena = patches.Rectangle(
        (ARENA_X_MIN, ARENA_Y_MIN),
        ARENA_X_MAX - ARENA_X_MIN,
        ARENA_Y_MAX - ARENA_Y_MIN,
        linewidth=3,
        edgecolor="black",
        facecolor="#f0f0f0",
        alpha=0.3,
    )
    ax.add_patch(arena)

    # Draw obstacles
    for i, (ox, oy) in enumerate(obstacles):
        circle = patches.Circle(
            (ox, oy),
            OBSTACLE_RADIUS,
            color="#d35400",
            alpha=0.8,
            zorder=3,
            edgecolor="black",
            linewidth=1.5,
        )
        ax.add_patch(circle)
        # Label obstacle number
        ax.text(
            ox,
            oy,
            str(i + 1),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
            zorder=4,
        )

    # Draw typical start and goal regions
    start_x, start_y = ARENA_X_MIN + 0.5, 0.0
    goal_x, goal_y = ARENA_X_MAX - 0.5, 0.0

    # Start position
    start_circle = patches.Circle(
        (start_x, start_y),
        AGENT_RADIUS * 1.5,
        color="#27ae60",
        alpha=0.7,
        zorder=2,
        edgecolor="darkgreen",
        linewidth=2,
    )
    ax.add_patch(start_circle)
    ax.text(
        start_x,
        start_y - 0.6,
        "START",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#27ae60",
    )

    # Goal position
    goal_star = ax.plot(
        goal_x,
        goal_y,
        marker="*",
        markersize=25,
        color="#e74c3c",
        zorder=2,
        markeredgecolor="darkred",
        markeredgewidth=1.5,
    )
    ax.text(
        goal_x,
        goal_y - 0.6,
        "GOAL",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#e74c3c",
    )

    # Title and labels
    num_obstacles = len(obstacles)
    difficulty = {1: "Very Easy", 4: "Easy", 7: "Medium", 12: "Hard", 16: "Very Hard"}
    diff_label = difficulty.get(num_obstacles, "Unknown")

    ax.set_title(
        f'{scenario_name}: {num_obstacles} Obstacle{"s" if num_obstacles > 1 else ""} ({diff_label})',
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("X Position (meters)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Y Position (meters)", fontsize=12, fontweight="bold")

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax.axvline(x=5, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

    # Legend
    legend_elements = [
        patches.Patch(
            facecolor="#27ae60", edgecolor="darkgreen", label="Start Position"
        ),
        patches.Patch(facecolor="#e74c3c", edgecolor="darkred", label="Goal Position"),
        patches.Patch(
            facecolor="#d35400", edgecolor="black", label="Obstacles (R=0.25m)"
        ),
        patches.Patch(facecolor="#f0f0f0", edgecolor="black", label="Navigation Arena"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)

    # Add info box
    info_text = f"Arena: {ARENA_X_MAX}m × {ARENA_Y_MAX - ARENA_Y_MIN:.1f}m\n"
    info_text += f"Obstacles: {num_obstacles}\n"
    info_text += f"Agent Radius: {AGENT_RADIUS}m"

    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {output_path}")


def generate_all_visualizations():
    """Generate visualizations for all scenarios."""
    # Create output directory
    output_dir = "scenario_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("GENERATING SCENARIO VISUALIZATIONS")
    print("=" * 70)
    print()

    for scenario, obstacles in OBSTACLE_CONFIGS.items():
        output_path = os.path.join(output_dir, f"{scenario.lower()}_layout.png")
        create_scenario_visualization(scenario, obstacles, output_path)

    # Create a combined overview
    create_combined_overview(output_dir)

    print()
    print("=" * 70)
    print(f"[DONE] All visualizations saved to: {output_dir}/")
    print("=" * 70)


def create_combined_overview(output_dir):
    """Create a combined figure showing all scenarios."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    scenarios = ["SOSP", "MOSP_A", "MOSP_B", "MOSP_C", "MOSP_D"]

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        obstacles = OBSTACLE_CONFIGS[scenario]

        # Set arena bounds
        ax.set_xlim(ARENA_X_MIN - 0.3, ARENA_X_MAX + 0.3)
        ax.set_ylim(ARENA_Y_MIN - 0.3, ARENA_Y_MAX + 0.3)
        ax.set_aspect("equal")

        # Draw arena
        arena = patches.Rectangle(
            (ARENA_X_MIN, ARENA_Y_MIN),
            ARENA_X_MAX - ARENA_X_MIN,
            ARENA_Y_MAX - ARENA_Y_MIN,
            linewidth=2,
            edgecolor="black",
            facecolor="#f0f0f0",
            alpha=0.3,
        )
        ax.add_patch(arena)

        # Draw obstacles
        for ox, oy in obstacles:
            circle = patches.Circle(
                (ox, oy),
                OBSTACLE_RADIUS,
                color="#d35400",
                alpha=0.8,
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(circle)

        # Start and goal
        ax.plot(
            0.5,
            0,
            "go",
            markersize=10,
            markeredgecolor="darkgreen",
            markeredgewidth=1.5,
        )
        ax.plot(
            9.5, 0, "r*", markersize=15, markeredgecolor="darkred", markeredgewidth=1
        )

        # Title
        ax.set_title(
            f"{scenario} ({len(obstacles)} obstacles)", fontsize=12, fontweight="bold"
        )
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("X (m)", fontsize=9)
        ax.set_ylabel("Y (m)", fontsize=9)

    # Hide the last subplot
    axes[-1].axis("off")

    plt.suptitle(
        "VGA Dataset: All Scenario Layouts", fontsize=16, fontweight="bold", y=0.98
    )
    plt.tight_layout()

    output_path = os.path.join(output_dir, "all_scenarios_overview.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {output_path}")


if __name__ == "__main__":
    generate_all_visualizations()
