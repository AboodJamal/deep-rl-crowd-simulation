"""
Regenerate Per-Model Videos
============================

This script regenerates videos for each model separately,
like the format in experiment_001.
"""

import json
import numpy as np
from pathlib import Path
import sys
import os

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from video_generator import TrajectoryVideoGenerator
from data_loading.vga_dataset import VGADatasetLoader


def regenerate_videos():
    """Regenerate per-model videos from existing results"""
    print("=" * 60)
    print("REGENERATING PER-MODEL VIDEOS")
    print("=" * 60)

    # Load results
    results_dir = Path("validation/results/vga_comparison_experiment_with_drl")
    results_json = results_dir / "comprehensive_results.json"

    with open(results_json, "r") as f:
        results = json.load(f)

    # Load experimental data for obstacle info
    data_root = r"D:\Abdullah Jamal\downloads\VGA-exp\Pedestrian-Experimental-Data"
    loader = VGADatasetLoader(data_root, obstacle_radius=0.25)
    experimental_data = loader.load_all()

    # Video generator
    video_gen = TrajectoryVideoGenerator(fps=20)

    # Output directory
    video_dir = results_dir / "videos"

    # For each scenario
    for scenario_type, scenario_data in results["scenarios"].items():
        print(f"\n{scenario_type.upper()}:")

        # Get experimental trials for this scenario
        trials = experimental_data.get(scenario_type, [])
        if not trials:
            continue

        # Get first 3 trial IDs
        trial_ids = sorted(list(set(t.trial_id for t in trials)))[:3]

        for trial_id in trial_ids:
            # Find the trial object
            trial = next((t for t in trials if t.trial_id == trial_id), None)
            if not trial:
                continue

            print(f"  Trial {trial_id}:")

            start_pos = np.array(trial.initial_pos)
            goal_pos = np.array(trial.final_pos)
            obstacles = trial.obstacles

            # VGA+UPL Deterministic
            if "vga_upl_deterministic" in scenario_data["models"]:
                det_results = [
                    r
                    for r in scenario_data["models"]["vga_upl_deterministic"]
                    if r["trial_id"] == trial_id and r.get("positions")
                ]
                if det_results:
                    positions = np.array(det_results[0]["positions"])
                    if len(positions) > 1:
                        output_path = (
                            video_dir / f"{scenario_type}_trial_{trial_id}_vga_det.mp4"
                        )
                        try:
                            video_gen.create_single_trajectory_video(
                                positions=positions,
                                obstacles=obstacles,
                                start_pos=start_pos,
                                goal_pos=goal_pos,
                                output_path=str(output_path),
                                title=f"{scenario_type.upper()} Trial {trial_id} - VGA+UPL (Det)",
                                show_velocity=False,
                            )
                            print(f"    [OK] VGA+UPL Det video")
                        except Exception as e:
                            print(f"    [ERROR] VGA Det: {e}")

            # VGA+UPL Stochastic (overlay)
            if "vga_upl_stochastic" in scenario_data["models"]:
                stoch_results = [
                    r
                    for r in scenario_data["models"]["vga_upl_stochastic"]
                    if r["trial_id"] == trial_id and r.get("positions")
                ]
                if stoch_results:
                    traj_list = [
                        np.array(r["positions"])
                        for r in stoch_results[:20]  # Limit to 20 runs
                        if r.get("positions") and len(r["positions"]) > 1
                    ]
                    if traj_list:
                        output_path = (
                            video_dir
                            / f"{scenario_type}_trial_{trial_id}_vga_stoch_overlay.mp4"
                        )
                        try:
                            video_gen.create_stochastic_overlay_video(
                                all_trajectories=traj_list,
                                obstacles=obstacles,
                                start_pos=start_pos,
                                goal_pos=goal_pos,
                                output_path=str(output_path),
                                title=f"{scenario_type.upper()} Trial {trial_id} - VGA+UPL Stoch ({len(traj_list)} runs)",
                            )
                            print(f"    [OK] VGA+UPL Stoch overlay video")
                        except Exception as e:
                            print(f"    [ERROR] VGA Stoch: {e}")

            # DRL (if available)
            if "drl" in scenario_data["models"]:
                drl_results = [
                    r
                    for r in scenario_data["models"]["drl"]
                    if r["trial_id"] == trial_id and r.get("positions")
                ]
                if drl_results:
                    positions = np.array(drl_results[0]["positions"])
                    if len(positions) > 1:
                        output_path = (
                            video_dir / f"{scenario_type}_trial_{trial_id}_drl.mp4"
                        )
                        try:
                            video_gen.create_single_trajectory_video(
                                positions=positions,
                                obstacles=obstacles,
                                start_pos=start_pos,
                                goal_pos=goal_pos,
                                output_path=str(output_path),
                                title=f"{scenario_type.upper()} Trial {trial_id} - DRL",
                                show_velocity=False,
                            )
                            print(f"    [OK] DRL video")
                        except Exception as e:
                            print(f"    [ERROR] DRL: {e}")

    print("\n" + "=" * 60)
    print("VIDEO REGENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    regenerate_videos()
