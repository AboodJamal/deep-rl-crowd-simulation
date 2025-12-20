"""
Re-run VGA+UPL experiment with FINAL optimized settings.
Simplified version - just run experiments and save results.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_loading.vga_dataset import VGADatasetLoader
from models.vga_upl_planner_ORIGINAL import VGAUPLPlanner


def convert_to_serializable(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        return convert_to_serializable(obj.__dict__)
    else:
        return obj


def main():
    """Run comprehensive VGA+UPL experiment."""

    # Setup
    results_dir = Path("validation/results/vga_final_experiment")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VGA+UPL FINAL EXPERIMENT - OPTIMIZED SETTINGS")
    print("=" * 80)
    print(f"Results: {results_dir}")
    print()

    # Load data
    loader = VGADatasetLoader(
        r"D:\Abdullah Jamal\downloads\VGA-exp\Pedestrian-Experimental-Data",
        obstacle_radius=0.25,
    )
    data = loader.load_all()

    # Models with FINAL settings - DETERMINISTIC ONLY
    models = {
        "VGA_UPL_Det": VGAUPLPlanner(use_probabilistic=False),
    }

    print("Models: VGA+UPL (Deterministic ONLY - faster)")
    print()

    # Scenarios
    scenarios = {
        "sosp": ("SOSP", data["sosp"]),
        "mosp_a": ("MOSP_A", data["mosp_a"]),
        "mosp_b": ("MOSP_B", data["mosp_b"]),
        "mosp_c": ("MOSP_C", data["mosp_c"]),
        "mosp_d": ("MOSP_D", data["mosp_d"]),
    }

    all_results = {}
    summary = {}

    # Run experiments
    for scenario_key, (scenario_name, trials) in scenarios.items():
        print(f"\n{scenario_name} ({len(trials)} trials)")
        print("-" * 80)

        all_results[scenario_name] = {}

        for model_name, model in models.items():
            print(f"  {model_name}...", end=" ", flush=True)

            results = []
            successes = 0

            for trial in trials:
                # Run deterministic
                result = model.simulate(
                    start_pos=np.array(trial.initial_pos),
                    goal_pos=np.array(trial.final_pos),
                    obstacles=trial.obstacles,
                    max_steps=1000,
                )

                results.append(
                    {
                        "trial_id": trial.trial_id,
                        "start_pos": convert_to_serializable(trial.initial_pos),
                        "goal_pos": convert_to_serializable(trial.final_pos),
                        "obstacles": convert_to_serializable(trial.obstacles),
                        "deterministic": {
                            "positions": result.positions.tolist(),
                            "success": bool(result.success),
                            "path_length": float(result.metadata.get("path_length", 0)),
                            "final_distance": float(
                                result.metadata.get("final_distance_to_goal", 0)
                            ),
                        },
                    }
                )

                if result.success:
                    successes += 1

            all_results[scenario_name][model_name] = results

            success_rate = (successes / len(trials)) * 100
            print(f"{successes}/{len(trials)} ({success_rate:.1f}%)")

            if model_name not in summary:
                summary[model_name] = {}
            summary[model_name][scenario_name] = {
                "total": len(trials),
                "successes": successes,
                "rate": success_rate,
            }

        # Save progress after each scenario
        with open(results_dir / "comprehensive_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        with open(results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  [Saved progress for {scenario_name}]")

    # Save
    print(f"\n{'='*80}")
    print("SAVING RESULTS...")

    with open(results_dir / "comprehensive_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved to: {results_dir}")

    # Summary table
    print(f"\n{'='*80}")
    print("SUCCESS RATES")
    print("=" * 80)
    print(f"{'Scenario':<12} | {'Deterministic':<20}")
    print("-" * 80)

    for sc in ["SOSP", "MOSP_A", "MOSP_B", "MOSP_C", "MOSP_D"]:
        det = summary["VGA_UPL_Det"][sc]
        print(f"{sc:<12} | {det['successes']}/{det['total']} ({det['rate']:.1f}%)")

    print("=" * 80)
    print("\nEXPERIMENT COMPLETE!")


if __name__ == "__main__":
    main()
