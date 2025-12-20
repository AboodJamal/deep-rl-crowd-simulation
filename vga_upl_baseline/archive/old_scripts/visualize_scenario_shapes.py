"""
Visualize Scenario Shapes
==========================

Creates reference images showing the characteristic layout of each scenario type.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from data_loading.vga_dataset import VGADatasetLoader


def plot_scenario_shape(scenario_type, trials, output_path):
    """
    Plot the characteristic shape/layout of a scenario.

    Shows multiple trials to illustrate the typical pattern.
    """
    fig, axes = plt.subplots(1, min(3, len(trials)), figsize=(15, 5))
    if len(trials) == 1:
        axes = [axes]
    elif len(trials) < 3:
        axes = list(axes)
    else:
        axes = list(axes[:3])

    fig.suptitle(
        f"{scenario_type.upper()} - Scenario Layout", fontsize=16, fontweight="bold"
    )

    for idx, (ax, trial) in enumerate(zip(axes, trials[:3])):
        # Plot obstacles
        for obs in trial.obstacles:
            circle = patches.Circle(
                obs["position"],
                obs["radius"],
                facecolor="red",
                edgecolor="darkred",
                alpha=0.7,
                linewidth=2,
                label="Obstacle" if idx == 0 else "",
            )
            ax.add_patch(circle)

        # Plot start and goal
        ax.plot(
            trial.initial_pos[0],
            trial.initial_pos[1],
            "o",
            color="green",
            markersize=15,
            markeredgecolor="darkgreen",
            markeredgewidth=2,
            label="Start" if idx == 0 else "",
            zorder=10,
        )
        ax.plot(
            trial.final_pos[0],
            trial.final_pos[1],
            "*",
            color="gold",
            markersize=20,
            markeredgecolor="orange",
            markeredgewidth=2,
            label="Goal" if idx == 0 else "",
            zorder=10,
        )

        # Draw a reference path line
        ax.plot(
            [trial.initial_pos[0], trial.final_pos[0]],
            [trial.initial_pos[1], trial.final_pos[1]],
            "--",
            color="gray",
            alpha=0.5,
            linewidth=1,
            label="Direct path" if idx == 0 else "",
        )

        # Set bounds including all elements
        all_x = [trial.initial_pos[0], trial.final_pos[0]] + [
            o["position"][0] for o in trial.obstacles
        ]
        all_y = [trial.initial_pos[1], trial.final_pos[1]] + [
            o["position"][1] for o in trial.obstacles
        ]

        margin = 1.5
        x_min, x_max = min(all_x) - margin, max(all_x) + margin
        y_min, y_max = min(all_y) - margin, max(all_y) + margin

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlabel("X (m)", fontsize=11)
        ax.set_ylabel("Y (m)", fontsize=11)
        ax.set_title(f"Trial {trial.trial_id}", fontsize=12)

        # Add text with info
        info_text = f"Obstacles: {len(trial.obstacles)}\n"
        info_text += f"Distance: {np.linalg.norm(np.array(trial.final_pos) - np.array(trial.initial_pos)):.2f}m"
        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    # Add legend to first subplot
    axes[0].legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {output_path.name}")


def create_overview_image(all_scenarios, output_path):
    """
    Create a single overview image showing one example from each scenario type.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    fig.suptitle("Experimental Scenario Types Overview", fontsize=18, fontweight="bold")

    scenario_order = ["sosp", "mosp_a", "mosp_b", "mosp_c", "mosp_d"]

    for idx, scenario_type in enumerate(scenario_order):
        if scenario_type not in all_scenarios or not all_scenarios[scenario_type]:
            continue

        ax = axes[idx]
        trial = all_scenarios[scenario_type][0]  # First trial

        # Plot obstacles
        for obs in trial.obstacles:
            circle = patches.Circle(
                obs["position"],
                obs["radius"],
                facecolor="red",
                edgecolor="darkred",
                alpha=0.7,
                linewidth=2,
            )
            ax.add_patch(circle)

        # Plot start and goal
        ax.plot(
            trial.initial_pos[0],
            trial.initial_pos[1],
            "o",
            color="green",
            markersize=15,
            markeredgecolor="darkgreen",
            markeredgewidth=2,
            zorder=10,
        )
        ax.plot(
            trial.final_pos[0],
            trial.final_pos[1],
            "*",
            color="gold",
            markersize=20,
            markeredgecolor="orange",
            markeredgewidth=2,
            zorder=10,
        )

        # Draw reference path
        ax.plot(
            [trial.initial_pos[0], trial.final_pos[0]],
            [trial.initial_pos[1], trial.final_pos[1]],
            "--",
            color="gray",
            alpha=0.5,
            linewidth=2,
        )

        # Set bounds
        all_x = [trial.initial_pos[0], trial.final_pos[0]] + [
            o["position"][0] for o in trial.obstacles
        ]
        all_y = [trial.initial_pos[1], trial.final_pos[1]] + [
            o["position"][1] for o in trial.obstacles
        ]

        margin = 1.5
        x_min, x_max = min(all_x) - margin, max(all_x) + margin
        y_min, y_max = min(all_y) - margin, max(all_y) + margin

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlabel("X (m)", fontsize=11)
        ax.set_ylabel("Y (m)", fontsize=11)
        ax.set_title(f"{scenario_type.upper()}", fontsize=14, fontweight="bold")

        # Add info
        n_trials = len(all_scenarios[scenario_type])
        info_text = f"Obstacles: {len(trial.obstacles)}\n"
        info_text += f"Total trials: {n_trials}"
        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

    # Hide extra subplot
    axes[-1].axis("off")

    # Add legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markersize=12,
            markeredgecolor="darkgreen",
            markeredgewidth=2,
            label="Start",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="gold",
            markersize=15,
            markeredgecolor="orange",
            markeredgewidth=2,
            label="Goal",
        ),
        patches.Circle(
            (0, 0),
            0.1,
            facecolor="red",
            edgecolor="darkred",
            alpha=0.7,
            linewidth=2,
            label="Obstacle",
        ),
        plt.Line2D(
            [0], [0], linestyle="--", color="gray", linewidth=2, label="Direct path"
        ),
    ]
    axes[-1].legend(handles=legend_elements, loc="center", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {output_path.name}")


def main():
    """Main function"""
    print("=" * 60)
    print("VISUALIZING SCENARIO SHAPES")
    print("=" * 60)

    # Load data
    data_root = r"D:\Abdullah Jamal\downloads\VGA-exp\Pedestrian-Experimental-Data"
    print(f"\nLoading data from: {data_root}")

    loader = VGADatasetLoader(data_root, obstacle_radius=0.25)
    data = loader.load_all()

    # Create output directory
    output_dir = Path(
        "validation/results/vga_comparison_experiment_with_drl/scenario_shapes"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Create individual scenario images
    print("\nCreating individual scenario images...")
    for scenario_type, trials in data.items():
        if not trials:
            continue
        output_path = output_dir / f"{scenario_type}_layout.png"
        plot_scenario_shape(scenario_type, trials, output_path)

    # Create overview image
    print("\nCreating overview image...")
    overview_path = output_dir / "scenarios_overview.png"
    create_overview_image(data, overview_path)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nGenerated images in: {output_dir}")
    print("\nFiles created:")
    for img in sorted(output_dir.glob("*.png")):
        print(f"  - {img.name}")


if __name__ == "__main__":
    main()
