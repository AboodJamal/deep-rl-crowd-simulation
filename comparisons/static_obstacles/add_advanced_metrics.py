"""
Add Advanced Metrics for Robot Navigation Comparison
Based on research in:
- Human-Robot Interaction comfort metrics
- Autonomous vehicle smoothness standards
- ISO 2631 vibration comfort standards
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats


def calculate_advanced_metrics(positions, dt=0.05):
    """Calculate advanced comfort and performance metrics from trajectory."""
    positions = np.array(positions)

    if len(positions) < 3:
        return {}

    # Velocities
    velocities = np.diff(positions, axis=0) / dt
    speeds = np.linalg.norm(velocities, axis=1)

    # Accelerations
    accelerations = np.diff(velocities, axis=0) / dt
    accel_magnitudes = np.linalg.norm(accelerations, axis=1)

    # Jerks
    if len(accelerations) > 1:
        jerks = np.diff(accelerations, axis=0) / dt
        jerk_magnitudes = np.linalg.norm(jerks, axis=1)
    else:
        jerk_magnitudes = np.array([0])

    # === COMFORT METRICS ===

    # 1. RMS Acceleration (ISO 2631 comfort standard)
    # Lower is better - relates to vibration discomfort
    rms_acceleration = (
        np.sqrt(np.mean(accel_magnitudes**2)) if len(accel_magnitudes) > 0 else 0
    )

    # 2. RMS Jerk - smoothness indicator
    rms_jerk = np.sqrt(np.mean(jerk_magnitudes**2)) if len(jerk_magnitudes) > 0 else 0

    # 3. Speed Consistency Index (coefficient of variation)
    # Lower CV = more consistent speed = more comfortable
    speed_cv = np.std(speeds) / np.mean(speeds) if np.mean(speeds) > 0 else 0

    # 4. Acceleration Smoothness (variance of acceleration)
    accel_variance = np.var(accel_magnitudes) if len(accel_magnitudes) > 0 else 0

    # === PREDICTABILITY METRICS ===

    # 5. Heading Change Rate (angular velocity)
    if len(velocities) > 1:
        headings = np.arctan2(velocities[:, 1], velocities[:, 0])
        heading_changes = np.abs(np.diff(headings))
        # Handle wraparound
        heading_changes = np.minimum(heading_changes, 2 * np.pi - heading_changes)
        heading_rate = np.mean(heading_changes) / dt  # rad/s
        max_heading_rate = np.max(heading_changes) / dt
    else:
        heading_rate = 0
        max_heading_rate = 0

    # 6. Path Curvature (average curvature)
    # Lower curvature = more predictable path
    if len(positions) > 2:
        # Approximate curvature using three consecutive points
        curvatures = []
        for i in range(1, len(positions) - 1):
            p1, p2, p3 = positions[i - 1], positions[i], positions[i + 1]
            # Area of triangle formed by 3 points
            area = 0.5 * abs(
                (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
            )
            # Distances
            d1 = np.linalg.norm(p2 - p1)
            d2 = np.linalg.norm(p3 - p2)
            d3 = np.linalg.norm(p3 - p1)
            # Curvature = 4*area / (d1*d2*d3)
            if d1 * d2 * d3 > 0:
                curvatures.append(4 * area / (d1 * d2 * d3))
        avg_curvature = np.mean(curvatures) if curvatures else 0
        max_curvature = np.max(curvatures) if curvatures else 0
    else:
        avg_curvature = 0
        max_curvature = 0

    # === EFFICIENCY METRICS ===

    # 7. Control Effort (integral of squared acceleration - energy-like metric)
    # Lower = more energy efficient
    control_effort = (
        np.sum(accel_magnitudes**2) * dt if len(accel_magnitudes) > 0 else 0
    )

    # 8. Normalized Control Effort (per unit distance)
    path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    normalized_control_effort = control_effort / path_length if path_length > 0 else 0

    # === DECISIVENESS METRICS ===

    # 9. Hesitation Index (number of significant speed drops)
    # Count times speed drops below 50% of average
    avg_speed = np.mean(speeds) if len(speeds) > 0 else 0
    hesitations = np.sum(speeds < 0.5 * avg_speed) if avg_speed > 0 else 0
    hesitation_ratio = hesitations / len(speeds) if len(speeds) > 0 else 0

    # 10. Acceleration Reversals (sign changes in acceleration components)
    if len(accelerations) > 1:
        ax_reversals = np.sum(np.diff(np.sign(accelerations[:, 0])) != 0)
        ay_reversals = np.sum(np.diff(np.sign(accelerations[:, 1])) != 0)
        total_reversals = ax_reversals + ay_reversals
    else:
        total_reversals = 0

    return {
        # Comfort
        "rms_acceleration": float(rms_acceleration),
        "rms_jerk": float(rms_jerk),
        "speed_consistency": float(1 - speed_cv),  # Higher is better
        "accel_variance": float(accel_variance),
        # Predictability
        "avg_heading_rate": float(heading_rate),
        "max_heading_rate": float(max_heading_rate),
        "avg_curvature": float(avg_curvature),
        "max_curvature": float(max_curvature),
        # Efficiency
        "control_effort": float(control_effort),
        "normalized_control_effort": float(normalized_control_effort),
        # Decisiveness
        "hesitation_ratio": float(hesitation_ratio),
        "accel_reversals": int(total_reversals),
    }


def analyze_with_advanced_metrics():
    """Analyze existing results with advanced metrics."""

    # Load results
    with open("professional_comparison2/comparison_results.json", "r") as f:
        data = json.load(f)

    print("=" * 80)
    print("ADVANCED METRICS ANALYSIS")
    print("Based on Human-Robot Interaction & Autonomous Vehicle Standards")
    print("=" * 80)

    all_vga_metrics = {
        k: []
        for k in [
            "rms_acceleration",
            "rms_jerk",
            "speed_consistency",
            "accel_variance",
            "avg_heading_rate",
            "max_heading_rate",
            "avg_curvature",
            "max_curvature",
            "control_effort",
            "normalized_control_effort",
            "hesitation_ratio",
            "accel_reversals",
        ]
    }
    all_drl_metrics = {k: [] for k in all_vga_metrics.keys()}

    for scenario_name, scenario_data in data["scenarios"].items():
        print(f"\n{scenario_name}:")
        print("-" * 40)

        vga_metrics = {k: [] for k in all_vga_metrics.keys()}
        drl_metrics = {k: [] for k in all_vga_metrics.keys()}

        # Calculate for VGA trials
        for trial in scenario_data["vga_trials"]:
            if "positions" in trial and trial["positions"]:
                metrics = calculate_advanced_metrics(trial["positions"])
                for k, v in metrics.items():
                    vga_metrics[k].append(v)
                    all_vga_metrics[k].append(v)

        # Calculate for DRL trials
        for trial in scenario_data["drl_trials"]:
            if "positions" in trial and trial["positions"]:
                metrics = calculate_advanced_metrics(trial["positions"])
                for k, v in metrics.items():
                    drl_metrics[k].append(v)
                    all_drl_metrics[k].append(v)

    # Print overall comparison
    print("\n" + "=" * 80)
    print("OVERALL ADVANCED METRICS COMPARISON")
    print("=" * 80)

    metric_info = {
        "rms_acceleration": (
            "RMS Acceleration (m/s²)",
            "lower",
            "Comfort - ISO 2631 standard",
        ),
        "rms_jerk": ("RMS Jerk (m/s³)", "lower", "Comfort - Smoothness indicator"),
        "speed_consistency": (
            "Speed Consistency",
            "higher",
            "Comfort - Less speed variation",
        ),
        "accel_variance": (
            "Acceleration Variance",
            "lower",
            "Comfort - Steady acceleration",
        ),
        "avg_heading_rate": (
            "Avg Heading Rate (rad/s)",
            "lower",
            "Predictability - Less turning",
        ),
        "max_heading_rate": (
            "Max Heading Rate (rad/s)",
            "lower",
            "Predictability - No sudden turns",
        ),
        "avg_curvature": (
            "Avg Path Curvature",
            "lower",
            "Predictability - Straighter path",
        ),
        "max_curvature": (
            "Max Path Curvature",
            "lower",
            "Predictability - No sharp turns",
        ),
        "control_effort": ("Control Effort", "lower", "Efficiency - Less energy"),
        "normalized_control_effort": (
            "Norm. Control Effort",
            "lower",
            "Efficiency - Energy per meter",
        ),
        "hesitation_ratio": (
            "Hesitation Ratio",
            "lower",
            "Decisiveness - Less stopping",
        ),
        "accel_reversals": ("Accel Reversals", "lower", "Decisiveness - Fewer changes"),
    }

    vga_wins = 0
    drl_wins = 0

    print(
        f"\n{'Metric':<30} {'VGA':<12} {'DRL':<12} {'Better':<8} {'Winner':<8} {'Description'}"
    )
    print("-" * 100)

    results = {}

    for metric, (name, direction, desc) in metric_info.items():
        vga_mean = np.mean(all_vga_metrics[metric]) if all_vga_metrics[metric] else 0
        drl_mean = np.mean(all_drl_metrics[metric]) if all_drl_metrics[metric] else 0

        if direction == "lower":
            winner = "VGA" if vga_mean < drl_mean else "DRL"
        else:
            winner = "VGA" if vga_mean > drl_mean else "DRL"

        if winner == "VGA":
            vga_wins += 1
        else:
            drl_wins += 1

        results[metric] = {
            "vga_mean": vga_mean,
            "drl_mean": drl_mean,
            "winner": winner,
            "direction": direction,
            "description": desc,
        }

        print(
            f"{name:<30} {vga_mean:<12.4f} {drl_mean:<12.4f} {direction:<8} {winner:<8} {desc}"
        )

    print("-" * 100)
    print(f"\nADVANCED METRICS SUMMARY: VGA wins {vga_wins}, DRL wins {drl_wins}")

    # Statistical significance tests
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE (t-test, p < 0.05)")
    print("=" * 80)

    for metric, (name, direction, desc) in metric_info.items():
        vga_vals = all_vga_metrics[metric]
        drl_vals = all_drl_metrics[metric]

        if len(vga_vals) > 1 and len(drl_vals) > 1:
            t_stat, p_val = stats.ttest_ind(vga_vals, drl_vals)
            sig = (
                "***"
                if p_val < 0.001
                else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            )
            winner = results[metric]["winner"]
            print(f"{name:<30} p={p_val:.4f} {sig:<4} Winner: {winner}")

    return results


if __name__ == "__main__":
    results = analyze_with_advanced_metrics()
