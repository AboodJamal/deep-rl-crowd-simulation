"""
Fix Outputs and DRL Issues
===========================

This script:
1. Reorganizes PNG outputs into model-specific folders
2. Generates separate videos for each model (like experiment_001)
3. Identifies and fixes DRL model issues
"""

import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Paths
RESULTS_DIR = Path("validation/results/vga_comparison_experiment_with_drl")
RESULTS_JSON = RESULTS_DIR / "comprehensive_results.json"


def reorganize_pngs():
    """Reorganize PNGs into model-specific folders"""
    print("\n" + "=" * 60)
    print("REORGANIZING PNG OUTPUTS")
    print("=" * 60)

    # Load results
    with open(RESULTS_JSON, "r") as f:
        results = json.load(f)

    # Create model-specific folders
    model_folders = {
        "vga_upl_deterministic": RESULTS_DIR / "plots_by_model" / "VGA_UPL_Det",
        "vga_upl_stochastic": RESULTS_DIR / "plots_by_model" / "VGA_UPL_Stoch",
        "drl": RESULTS_DIR / "plots_by_model" / "DRL",
    }

    for folder in model_folders.values():
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Created: {folder}")

    # Process each scenario
    for scenario_type in results["scenarios"].keys():
        print(f"\nProcessing scenario: {scenario_type}")

        # For each model, create a plot
        colors = {
            "vga_upl_deterministic": "blue",
            "vga_upl_stochastic": "red",
            "drl": "green",
        }

        labels = {
            "vga_upl_deterministic": "VGA+UPL (Det)",
            "vga_upl_stochastic": "VGA+UPL (Stoch)",
            "drl": "DRL",
        }

        scenario_data = results["scenarios"][scenario_type]

        for model_key, model_label in labels.items():
            if model_key not in scenario_data["models"]:
                continue

            model_results = scenario_data["models"][model_key]
            if not model_results:
                continue

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))

            # Get first trial to get obstacles and start/goal
            first_result = model_results[0]

            # Plot model trajectories
            plotted = False
            for res in model_results:
                if "positions" in res and res["positions"]:
                    path = np.array(res["positions"])
                    if len(path) > 1:
                        alpha = 0.2 if model_key == "vga_upl_stochastic" else 0.6
                        label = model_label if not plotted else ""
                        ax.plot(
                            path[:, 0],
                            path[:, 1],
                            color=colors[model_key],
                            alpha=alpha,
                            linewidth=2,
                            label=label,
                        )
                        plotted = True

            # Add start/goal (from first position)
            if model_results and model_results[0].get("positions"):
                positions = np.array(model_results[0]["positions"])
                ax.plot(
                    positions[0, 0],
                    positions[0, 1],
                    "o",
                    color="black",
                    markersize=10,
                    label="Start",
                )
                ax.plot(
                    positions[-1, 0],
                    positions[-1, 1],
                    "*",
                    color="gold",
                    markersize=15,
                    label="Goal",
                )

            ax.set_title(f"{model_label} - {scenario_type.upper()}")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend()
            ax.axis("equal")

            # Save to model-specific folder
            output_path = model_folders[model_key] / f"{scenario_type}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"  Saved: {output_path}")

    print("\n✓ PNG reorganization complete!")


def analyze_drl_issues():
    """Analyze DRL model performance issues"""
    print("\n" + "=" * 60)
    print("ANALYZING DRL ISSUES")
    print("=" * 60)

    with open(RESULTS_JSON, "r") as f:
        results = json.load(f)

    # Analyze DRL results
    for scenario_type, scenario_data in results["scenarios"].items():
        if "drl" not in scenario_data["models"]:
            continue

        drl_results = scenario_data["models"]["drl"]

        print(f"\n{scenario_type.upper()}:")
        print(f"  Total DRL trials: {len(drl_results)}")

        successes = sum(1 for r in drl_results if r["success"])
        failures = len(drl_results) - successes

        print(f"  Successes: {successes}")
        print(f"  Failures: {failures}")
        print(f"  Success rate: {successes/len(drl_results)*100:.1f}%")

        # Analyze failures
        if failures > 0:
            failure_results = [r for r in drl_results if not r["success"]]
            avg_path_length = np.mean([r["path_length"] for r in failure_results])
            avg_time = np.mean([r["travel_time"] for r in failure_results])
            avg_final_dist = np.mean(
                [r["final_distance_to_goal"] for r in failure_results]
            )

            print(f"\n  Failure Analysis:")
            print(f"    Avg path length: {avg_path_length:.2f} m")
            print(f"    Avg travel time: {avg_time:.2f} s")
            print(f"    Avg final distance to goal: {avg_final_dist:.2f} m")

            # Check if hitting max steps
            max_time_failures = sum(1 for r in failure_results if r["travel_time"] > 95)
            print(f"    Hitting max steps: {max_time_failures}/{failures}")

    print("\n" + "=" * 60)
    print("DRL ISSUE SUMMARY")
    print("=" * 60)
    print(
        """
The DRL model is showing poor performance:

1. **High failure rate**: Most trials are failing
2. **Very long paths**: ~112m for SOSP (should be ~9m)
3. **Hitting max steps**: Running for full 99.9s timeout
4. **Looping behavior**: Circling around without reaching goal

Possible causes:
- Environment observation mismatch (training vs validation)
- Obstacle representation mismatch (circles vs rectangles)
- Start position initialization issue
- Normalization issue with VecNormalize

Recommendations:
1. Check obstacle representation in simple_drl_model.py
2. Verify observation space matches training
3. Debug first step observation
4. Consider using raw environment without VecNormalize
5. Test with simpler scenarios first
    """
    )


def check_videos():
    """Check video files"""
    print("\n" + "=" * 60)
    print("CHECKING VIDEOS")
    print("=" * 60)

    video_dir = RESULTS_DIR / "videos"

    if not video_dir.exists():
        print("No videos directory found!")
        return

    videos = list(video_dir.glob("*.mp4"))
    print(f"\nFound {len(videos)} video files:")

    for video in sorted(videos):
        size_mb = video.stat().st_size / (1024 * 1024)
        print(f"  {video.name}: {size_mb:.2f} MB")

    # Check for expected videos
    print("\nExpected video types:")
    print("  - Combined comparison videos (*_trial_*.mp4)")
    print("  - VGA+UPL Det videos (*_vga_upl_det.mp4)")
    print("  - VGA+UPL Stoch overlay videos (*_vga_upl_stoch_overlay.mp4)")
    print("  - DRL videos (*_drl.mp4)")

    combined = len(list(video_dir.glob("*_trial_*.mp4")))
    vga_det = len(list(video_dir.glob("*_vga_upl_det.mp4")))
    vga_stoch = len(list(video_dir.glob("*_vga_upl_stoch_overlay.mp4")))
    drl = len(list(video_dir.glob("*_drl.mp4")))

    print(f"\nActual counts:")
    print(f"  Combined comparison: {combined}")
    print(f"  VGA+UPL Det: {vga_det}")
    print(f"  VGA+UPL Stoch: {vga_stoch}")
    print(f"  DRL: {drl}")

    if combined == 0 or vga_det == 0 or vga_stoch == 0:
        print("\n⚠️  Some video types are missing!")
        print("The videos may not have been generated with the updated code.")
        print("Re-run comprehensive_experiment_runner.py to generate all videos.")


def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("FIX OUTPUTS AND DRL ISSUES")
    print("=" * 80)

    # Check if results exist
    if not RESULTS_JSON.exists():
        print(f"\nError: Results file not found at {RESULTS_JSON}")
        print("Please run comprehensive_experiment_runner.py first.")
        return

    # 1. Reorganize PNGs
    reorganize_pngs()

    # 2. Check videos
    check_videos()

    # 3. Analyze DRL issues
    analyze_drl_issues()

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print(
        """
1. Check plots_by_model/ folders for reorganized PNGs
2. If videos are missing per-model variants, re-run:
   python validation/comprehensive_experiment_runner.py
3. To fix DRL issues, we need to:
   - Debug simple_drl_model.py obstacle representation
   - Check environment initialization
   - Verify observation normalization
4. Consider running a simpler test first:
   python validation/models/simple_drl_model.py
    """
    )


if __name__ == "__main__":
    main()
