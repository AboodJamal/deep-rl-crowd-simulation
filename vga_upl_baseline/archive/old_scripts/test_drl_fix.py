"""
Test DRL Model Fix
===================

Quick test to verify DRL model can now properly simulate experimental scenarios.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
sys.path.insert(0, str(Path(__file__).parent))

from models.simple_drl_model import SimpleDRLModel


def test_drl_simple():
    """Test DRL with a simple scenario"""
    print("=" * 60)
    print("Testing DRL Model Fix")
    print("=" * 60)

    # Load model
    base_dir = (
        Path(__file__).parent.parent
        / "docs"
        / "archive"
        / "latestRun5-12(reviewed)"
        / "models"
    )
    checkpoint_path = base_dir / "ultimate_generalized_agent_stage12.zip"
    vecnorm_path = base_dir / "ultimate_generalized_agent_stage12_vecnormalize.pkl"

    if not checkpoint_path.exists():
        # Try checkpoints dir
        base_dir = Path(__file__).parent.parent / "checkpoints"
        checkpoint_path = base_dir / "ultimate_stage12_3837248_steps.zip"
        vecnorm_path = base_dir / "ultimate_stage12_vecnormalize.pkl"

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found")
        return

    print(f"\nLoading DRL model...")
    model = SimpleDRLModel(str(checkpoint_path), str(vecnorm_path))

    # Simple test scenario (straight line with one obstacle)
    print("\nTest 1: Simple straight line with one obstacle")
    start_pos = np.array([2.0, 5.0])
    goal_pos = np.array([12.0, 5.0])
    obstacles = [{"position": [7.0, 5.0], "radius": 0.5}]

    result = model.simulate(
        start_pos=start_pos, goal_pos=goal_pos, obstacles=obstacles, max_steps=500
    )

    print(f"  Success: {result.success}")
    print(f"  Steps: {len(result.positions)}")
    print(
        f"  Path length: {np.sum(np.linalg.norm(np.diff(result.positions, axis=0), axis=1)):.2f}m"
    )
    if len(result.positions) > 0:
        final_dist = np.linalg.norm(result.positions[-1] - goal_pos)
        print(f"  Final distance to goal: {final_dist:.2f}m")

    # Test 2: More complex with multiple obstacles
    print("\nTest 2: Multiple obstacles")
    start_pos = np.array([0.0, 0.0])
    goal_pos = np.array([10.0, 0.0])
    obstacles = [
        {"position": [3.0, 0.5], "radius": 0.3},
        {"position": [5.0, -0.5], "radius": 0.3},
        {"position": [7.0, 0.3], "radius": 0.3},
    ]

    result = model.simulate(
        start_pos=start_pos, goal_pos=goal_pos, obstacles=obstacles, max_steps=500
    )

    print(f"  Success: {result.success}")
    print(f"  Steps: {len(result.positions)}")
    if len(result.positions) > 1:
        path_length = np.sum(np.linalg.norm(np.diff(result.positions, axis=0), axis=1))
        print(f"  Path length: {path_length:.2f}m")
    if len(result.positions) > 0:
        final_dist = np.linalg.norm(result.positions[-1] - goal_pos)
        print(f"  Final distance to goal: {final_dist:.2f}m")

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
    print("\nIf both tests show reasonable behavior (success or near-goal),")
    print("then the DRL fix is working. Otherwise, further debugging needed.")


if __name__ == "__main__":
    test_drl_simple()
