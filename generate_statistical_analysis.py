"""
Statistical Analysis & Advanced Visualization for VGA vs DRL Comparison
========================================================================

This script processes the professional_comparison results and generates:
1. Statistical significance tests (t-tests, confidence intervals)
2. Advanced visualizations (box plots, radar charts, violin plots)
3. SPL and other aggregate metrics
4. LaTeX tables and summary reports

Author: Generated for academic comparison
Date: December 2025
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")

# Set publication-quality style
plt.style.use("seaborn-v0_8-paper")
sns.set_palette("husl")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10


class StatisticalAnalyzer:
    """Performs statistical analysis on comparison results."""

    def __init__(self, results_path: str):
        """Load comparison results."""
        with open(results_path, "r") as f:
            self.data = json.load(f)
        self.scenarios = list(self.data["scenarios"].keys())

    def extract_metric_arrays(
        self, metric: str
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Extract metric values for VGA and DRL across all scenarios."""
        vga_data = {}
        drl_data = {}

        for scenario in self.scenarios:
            scenario_data = self.data["scenarios"][scenario]
            vga_values = [trial[metric] for trial in scenario_data["vga_trials"]]
            drl_values = [trial[metric] for trial in scenario_data["drl_trials"]]
            vga_data[scenario] = np.array(vga_values)
            drl_data[scenario] = np.array(drl_values)

        return vga_data, drl_data

    def compute_spl(self) -> Dict[str, Dict[str, float]]:
        """Compute Success weighted by Path Length (SPL) metric."""
        spl_results = {}

        for scenario in self.scenarios:
            scenario_data = self.data["scenarios"][scenario]

            # VGA SPL
            vga_spl = []
            for trial in scenario_data["vga_trials"]:
                if trial["success"]:
                    spl = trial["optimal_path_length"] / max(
                        trial["path_length"], trial["optimal_path_length"]
                    )
                else:
                    spl = 0.0
                vga_spl.append(spl)

            # DRL SPL
            drl_spl = []
            for trial in scenario_data["drl_trials"]:
                if trial["success"]:
                    spl = trial["optimal_path_length"] / max(
                        trial["path_length"], trial["optimal_path_length"]
                    )
                else:
                    spl = 0.0
                drl_spl.append(spl)

            spl_results[scenario] = {
                "vga_mean": np.mean(vga_spl),
                "vga_std": np.std(vga_spl),
                "drl_mean": np.mean(drl_spl),
                "drl_std": np.std(drl_spl),
            }

        return spl_results

    def perform_t_tests(
        self, metrics: List[str]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Perform paired t-tests for each metric across scenarios."""
        results = {}

        for metric in metrics:
            vga_data, drl_data = self.extract_metric_arrays(metric)
            results[metric] = {}

            for scenario in self.scenarios:
                vga = vga_data[scenario]
                drl = drl_data[scenario]

                # Paired t-test (same trials for both models)
                t_stat, p_value = stats.ttest_rel(vga, drl)

                # Effect size (Cohen's d)
                pooled_std = np.sqrt((np.var(vga) + np.var(drl)) / 2)
                cohens_d = (
                    (np.mean(vga) - np.mean(drl)) / pooled_std if pooled_std > 0 else 0
                )

                # 95% confidence interval for difference
                diff = vga - drl
                ci_95 = stats.t.interval(
                    0.95, len(diff) - 1, loc=np.mean(diff), scale=stats.sem(diff)
                )

                results[metric][scenario] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "cohens_d": float(cohens_d),
                    "ci_95_lower": float(ci_95[0]),
                    "ci_95_upper": float(ci_95[1]),
                    "vga_mean": float(np.mean(vga)),
                    "vga_std": float(np.std(vga)),
                    "drl_mean": float(np.mean(drl)),
                    "drl_std": float(np.std(drl)),
                    "significant": bool(p_value < 0.05),
                }

        return results

    def create_box_plots(self, output_dir: Path):
        """Create box plots for key metrics."""
        metrics = [
            ("path_efficiency", "Path Efficiency", False),
            ("travel_time", "Travel Time (s)", False),
            ("average_speed", "Average Speed (m/s)", False),
            ("num_collisions", "Number of Collisions", False),
            ("oscillation_index", "Oscillation Index", False),
            ("min_clearance", "Min Clearance (m)", False),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, (metric, label, _) in enumerate(metrics):
            ax = axes[idx]
            vga_data, drl_data = self.extract_metric_arrays(metric)

            # Prepare data for box plot
            all_data = []
            labels = []
            colors = []

            for scenario in self.scenarios:
                all_data.extend(vga_data[scenario])
                labels.extend([f"{scenario}\nVGA"] * len(vga_data[scenario]))
                colors.extend(["#3498db"] * len(vga_data[scenario]))

                all_data.extend(drl_data[scenario])
                labels.extend([f"{scenario}\nDRL"] * len(drl_data[scenario]))
                colors.extend(["#e74c3c"] * len(drl_data[scenario]))

            # Create box plot
            positions = []
            data_grouped = []
            pos = 0
            for scenario in self.scenarios:
                data_grouped.append(vga_data[scenario])
                positions.append(pos)
                pos += 1
                data_grouped.append(drl_data[scenario])
                positions.append(pos)
                pos += 1.5

            bp = ax.boxplot(
                data_grouped,
                positions=positions,
                widths=0.6,
                patch_artist=True,
                showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="gold", markersize=5),
            )

            # Color boxes
            for i, patch in enumerate(bp["boxes"]):
                if i % 2 == 0:  # VGA
                    patch.set_facecolor("#3498db")
                    patch.set_alpha(0.6)
                else:  # DRL
                    patch.set_facecolor("#e74c3c")
                    patch.set_alpha(0.6)

            ax.set_ylabel(label)
            ax.set_xlabel("Scenario")
            ax.set_xticks([0.5, 2.5, 4.5, 6.5, 8.5])
            ax.set_xticklabels(self.scenarios, rotation=0)
            ax.grid(axis="y", alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#3498db", alpha=0.6, label="VGA+UPL V4"),
            Patch(facecolor="#e74c3c", alpha=0.6, label="DRL (PPO)"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.98),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(
            output_dir / "box_plots_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def create_radar_chart(self, output_dir: Path):
        """Create radar chart comparing models on key metrics."""
        # Select representative metrics (normalized to 0-1 scale)
        metrics = [
            "path_efficiency",
            "average_speed",
            "average_clearance",
            "oscillation_index",
            "travel_time",
            "num_collisions",
        ]
        labels = [
            "Path Efficiency",
            "Speed",
            "Safety",
            "Smoothness",
            "Time",
            "Collisions",
        ]

        # Aggregate across all scenarios
        vga_values = []
        drl_values = []

        for metric in metrics:
            vga_data, drl_data = self.extract_metric_arrays(metric)
            vga_all = np.concatenate([vga_data[s] for s in self.scenarios])
            drl_all = np.concatenate([drl_data[s] for s in self.scenarios])

            # Normalize to 0-1 (handle inverted metrics like time and collisions)
            if metric in ["travel_time", "num_collisions", "oscillation_index"]:
                # Lower is better - invert normalization
                max_val = max(np.max(vga_all), np.max(drl_all))
                min_val = min(np.min(vga_all), np.min(drl_all))
                if max_val > min_val:
                    vga_norm = 1 - (np.mean(vga_all) - min_val) / (max_val - min_val)
                    drl_norm = 1 - (np.mean(drl_all) - min_val) / (max_val - min_val)
                else:
                    vga_norm = drl_norm = 0.5
            else:
                # Higher is better
                max_val = max(np.max(vga_all), np.max(drl_all))
                min_val = min(np.min(vga_all), np.min(drl_all))
                if max_val > min_val:
                    vga_norm = (np.mean(vga_all) - min_val) / (max_val - min_val)
                    drl_norm = (np.mean(drl_all) - min_val) / (max_val - min_val)
                else:
                    vga_norm = drl_norm = 0.5

            vga_values.append(vga_norm)
            drl_values.append(drl_norm)

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        vga_values += vga_values[:1]
        drl_values += drl_values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

        ax.plot(
            angles, vga_values, "o-", linewidth=2, label="VGA+UPL V4", color="#3498db"
        )
        ax.fill(angles, vga_values, alpha=0.25, color="#3498db")

        ax.plot(
            angles, drl_values, "o-", linewidth=2, label="DRL (PPO)", color="#e74c3c"
        )
        ax.fill(angles, drl_values, alpha=0.25, color="#e74c3c")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        ax.set_title(
            "Multi-Metric Performance Comparison\n(Normalized to 0-1 scale)",
            size=14,
            pad=20,
        )

        plt.tight_layout()
        plt.savefig(
            output_dir / "radar_chart_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def create_violin_plots(self, output_dir: Path):
        """Create violin plots for distribution comparison."""
        metrics = [
            ("path_efficiency", "Path Efficiency"),
            ("travel_time", "Travel Time (s)"),
            ("oscillation_index", "Oscillation Index"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, (metric, label) in enumerate(metrics):
            ax = axes[idx]
            vga_data, drl_data = self.extract_metric_arrays(metric)

            # Combine all scenarios
            vga_all = np.concatenate([vga_data[s] for s in self.scenarios])
            drl_all = np.concatenate([drl_data[s] for s in self.scenarios])

            # Create violin plot
            parts = ax.violinplot(
                [vga_all, drl_all],
                positions=[1, 2],
                showmeans=True,
                showextrema=True,
                widths=0.7,
            )

            # Color the violins
            for i, pc in enumerate(parts["bodies"]):
                if i == 0:
                    pc.set_facecolor("#3498db")
                else:
                    pc.set_facecolor("#e74c3c")
                pc.set_alpha(0.6)

            ax.set_ylabel(label)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["VGA+UPL V4", "DRL (PPO)"])
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / "violin_plots_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def generate_latex_table(
        self, t_test_results: Dict, spl_results: Dict, output_dir: Path
    ):
        """Generate LaTeX table for paper."""
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append(
            "\\caption{Comparison of VGA+UPL V4 and DRL (PPO) across scenarios}"
        )
        latex.append("\\label{tab:comparison}")
        latex.append("\\begin{tabular}{llrrrr}")
        latex.append("\\toprule")
        latex.append(
            "Scenario & Metric & VGA Mean $\\pm$ SD & DRL Mean $\\pm$ SD & $p$-value & Sig. \\\\"
        )
        latex.append("\\midrule")

        # Key metrics to include
        key_metrics = [
            ("path_efficiency", "Path Efficiency"),
            ("travel_time", "Travel Time (s)"),
            ("num_collisions", "Collisions"),
        ]

        for scenario in self.scenarios:
            latex.append(f"\\multirow{{3}}{{*}}{{{scenario}}} ")

            for i, (metric, label) in enumerate(key_metrics):
                result = t_test_results[metric][scenario]
                sig_marker = (
                    "***"
                    if result["p_value"] < 0.001
                    else (
                        "**"
                        if result["p_value"] < 0.01
                        else ("*" if result["p_value"] < 0.05 else "")
                    )
                )

                if i > 0:
                    latex.append(" & ")
                else:
                    latex.append(" & ")

                latex.append(
                    f"{label} & "
                    f"{result['vga_mean']:.3f} $\\pm$ {result['vga_std']:.3f} & "
                    f"{result['drl_mean']:.3f} $\\pm$ {result['drl_std']:.3f} & "
                    f"{result['p_value']:.4f} & {sig_marker} \\\\"
                )

            # Add SPL
            spl = spl_results[scenario]
            latex.append(
                f" & SPL & {spl['vga_mean']:.3f} $\\pm$ {spl['vga_std']:.3f} & "
                f"{spl['drl_mean']:.3f} $\\pm$ {spl['drl_std']:.3f} & - & - \\\\"
            )
            latex.append("\\midrule")

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        with open(output_dir / "comparison_table.tex", "w") as f:
            f.write("\n".join(latex))

    def generate_summary_report(
        self, t_test_results: Dict, spl_results: Dict, output_dir: Path
    ):
        """Generate comprehensive text summary."""
        report = []
        report.append("=" * 80)
        report.append("STATISTICAL ANALYSIS SUMMARY")
        report.append("VGA+UPL V4 vs DRL (PPO) Comparison")
        report.append("=" * 80)
        report.append("")

        # Overall summary
        report.append("1. SUCCESS RATES:")
        for scenario in self.scenarios:
            scenario_data = self.data["scenarios"][scenario]
            vga_success = sum(
                1 if t["success"] else 0 for t in scenario_data["vga_trials"]
            ) / len(scenario_data["vga_trials"])
            drl_success = sum(
                1 if t["success"] else 0 for t in scenario_data["drl_trials"]
            ) / len(scenario_data["drl_trials"])
            report.append(
                f"   {scenario:<10}: VGA={vga_success*100:>5.1f}%  DRL={drl_success*100:>5.1f}%"
            )

        report.append("")
        report.append("2. SPL (SUCCESS WEIGHTED PATH LENGTH):")
        for scenario in self.scenarios:
            spl = spl_results[scenario]
            report.append(
                f"   {scenario:<10}: VGA={spl['vga_mean']:.3f}±{spl['vga_std']:.3f}  "
                f"DRL={spl['drl_mean']:.3f}±{spl['drl_std']:.3f}"
            )

        report.append("")
        report.append("3. STATISTICAL SIGNIFICANCE (p < 0.05):")
        report.append("")

        metrics_to_report = [
            ("path_efficiency", "Path Efficiency"),
            ("travel_time", "Travel Time"),
            ("average_speed", "Average Speed"),
            ("num_collisions", "Collisions"),
            ("oscillation_index", "Oscillation"),
            ("min_clearance", "Min Clearance"),
        ]

        for metric, label in metrics_to_report:
            report.append(f"   {label}:")
            for scenario in self.scenarios:
                result = t_test_results[metric][scenario]
                sig = "[SIGNIFICANT]" if result["significant"] else "[Not significant]"
                winner = "VGA" if result["vga_mean"] > result["drl_mean"] else "DRL"
                if metric in ["travel_time", "num_collisions", "oscillation_index"]:
                    winner = "VGA" if result["vga_mean"] < result["drl_mean"] else "DRL"

                report.append(
                    f"      {scenario:<10}: p={result['p_value']:.4f} {sig} "
                    f"Winner: {winner} (d={result['cohens_d']:.2f})"
                )
            report.append("")

        report.append("=" * 80)
        report.append("INTERPRETATION GUIDE:")
        report.append("- p < 0.05: Statistically significant difference")
        report.append("- Cohen's d: Effect size (0.2=small, 0.5=medium, 0.8=large)")
        report.append("- SPL: Combines success rate with path optimality")
        report.append("=" * 80)

        with open(output_dir / "statistical_summary.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(report))

        print("\n".join(report))


def main():
    """Main entry point."""
    # Paths
    results_path = Path("professional_comparison/comparison_results.json")
    output_dir = Path("professional_comparison/statistical_analysis")
    output_dir.mkdir(exist_ok=True)

    if not results_path.exists():
        print(f"[ERROR] Results not found: {results_path}")
        print("Please run professional_comparison.py first!")
        return

    print("[LOAD] Loading comparison results...")
    analyzer = StatisticalAnalyzer(str(results_path))

    print("[COMPUTE] Computing SPL metrics...")
    spl_results = analyzer.compute_spl()

    print("[COMPUTE] Performing statistical tests...")
    key_metrics = [
        "path_efficiency",
        "travel_time",
        "path_length",
        "average_speed",
        "speed_variance",
        "num_collisions",
        "oscillation_index",
        "min_clearance",
        "average_clearance",
        "danger_zone_ratio",
        "average_jerk",
    ]
    t_test_results = analyzer.perform_t_tests(key_metrics)

    print("[PLOT] Creating box plots...")
    analyzer.create_box_plots(output_dir)

    print("[PLOT] Creating radar chart...")
    analyzer.create_radar_chart(output_dir)

    print("[PLOT] Creating violin plots...")
    analyzer.create_violin_plots(output_dir)

    print("[EXPORT] Generating LaTeX table...")
    analyzer.generate_latex_table(t_test_results, spl_results, output_dir)

    print("[EXPORT] Generating summary report...")
    analyzer.generate_summary_report(t_test_results, spl_results, output_dir)

    # Save complete statistical results as JSON
    print("[SAVE] Saving complete statistical results...")
    complete_results = {
        "spl_metrics": spl_results,
        "t_test_results": t_test_results,
    }
    with open(output_dir / "statistical_results.json", "w") as f:
        json.dump(complete_results, f, indent=2)

    print("\n" + "=" * 80)
    print("[DONE] Statistical analysis complete!")
    print(f"Output directory: {output_dir}")
    print("Generated files:")
    print("  - box_plots_comparison.png")
    print("  - radar_chart_comparison.png")
    print("  - violin_plots_comparison.png")
    print("  - comparison_table.tex (for LaTeX papers)")
    print("  - statistical_summary.txt")
    print("  - statistical_results.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
