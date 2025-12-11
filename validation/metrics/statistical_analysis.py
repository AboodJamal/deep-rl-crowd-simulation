"""
Statistical Analysis Module

Provides statistical tests and analysis for comparing model performance:
- Significance testing (t-tests, Wilcoxon, Mann-Whitney)
- Confidence intervals
- Effect size calculations
- Distribution analysis
- Multiple comparison corrections

Used to determine if differences between models or between model and human
data are statistically significant.

Author: Abdallah Jamal
Date: December 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from scipy.stats import (
    ttest_ind, ttest_rel, wilcoxon, mannwhitneyu,
    shapiro, levene, kstest, anderson
)


@dataclass
class StatisticalTestResult:
    """Result of a statistical test"""
    test_name: str
    statistic: float
    p_value: float
    significant: bool  # p < 0.05
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""


@dataclass
class DistributionAnalysis:
    """Analysis of a data distribution"""
    mean: float
    median: float
    std: float
    min: float
    max: float
    q25: float  # 25th percentile
    q75: float  # 75th percentile
    skewness: float
    kurtosis: float
    is_normal: bool  # Shapiro-Wilk test p > 0.05
    normality_p_value: float


def analyze_distribution(data: np.ndarray) -> DistributionAnalysis:
    """
    Analyze the distribution of data
    
    Args:
        data: 1D array of values
        
    Returns:
        DistributionAnalysis object
    """
    data = np.array(data).flatten()
    data = data[~np.isnan(data)]  # Remove NaN
    
    if len(data) == 0:
        return DistributionAnalysis(
            mean=0.0, median=0.0, std=0.0, min=0.0, max=0.0,
            q25=0.0, q75=0.0, skewness=0.0, kurtosis=0.0,
            is_normal=False, normality_p_value=0.0
        )
    
    # Normality test (Shapiro-Wilk)
    if len(data) >= 3:
        _, p_normal = shapiro(data)
    else:
        p_normal = 0.0
    
    return DistributionAnalysis(
        mean=float(np.mean(data)),
        median=float(np.median(data)),
        std=float(np.std(data, ddof=1) if len(data) > 1 else 0.0),
        min=float(np.min(data)),
        max=float(np.max(data)),
        q25=float(np.percentile(data, 25)),
        q75=float(np.percentile(data, 75)),
        skewness=float(stats.skew(data)),
        kurtosis=float(stats.kurtosis(data)),
        is_normal=bool(p_normal > 0.05),
        normality_p_value=float(p_normal)
    )


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size
    
    d = (mean1 - mean2) / pooled_std
    
    Interpretation:
    - Small: |d| ~ 0.2
    - Medium: |d| ~ 0.5
    - Large: |d| ~ 0.8
    
    Args:
        group1, group2: Arrays of values
        
    Returns:
        Cohen's d
    """
    group1 = np.array(group1).flatten()
    group2 = np.array(group2).flatten()
    
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std < 1e-10:
        return 0.0
    
    d = (mean1 - mean2) / pooled_std
    return float(d)


def compute_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Compute confidence interval for mean
    
    Args:
        data: Array of values
        confidence: Confidence level (default 95%)
        
    Returns:
        (lower_bound, upper_bound)
    """
    data = np.array(data).flatten()
    data = data[~np.isnan(data)]
    
    if len(data) < 2:
        mean = np.mean(data) if len(data) > 0 else 0.0
        return (mean, mean)
    
    mean = np.mean(data)
    se = stats.sem(data)  # Standard error
    margin = se * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    
    return (float(mean - margin), float(mean + margin))


def independent_samples_test(
    group1: np.ndarray,
    group2: np.ndarray,
    parametric: Optional[bool] = None,
    alternative: str = 'two-sided'
) -> StatisticalTestResult:
    """
    Test if two independent groups have different means/medians
    
    Automatically selects appropriate test:
    - If parametric=True or data is normal: Independent t-test
    - If parametric=False or data is non-normal: Mann-Whitney U test
    
    Args:
        group1, group2: Arrays of values
        parametric: Force parametric (t-test) or non-parametric (Mann-Whitney).
                   If None, automatically decide based on normality.
        alternative: 'two-sided', 'less', or 'greater'
        
    Returns:
        StatisticalTestResult
    """
    group1 = np.array(group1).flatten()
    group2 = np.array(group2).flatten()
    
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]
    
    if len(group1) < 2 or len(group2) < 2:
        return StatisticalTestResult(
            test_name="Insufficient Data",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            interpretation="Not enough data for testing"
        )
    
    # Decide test type
    if parametric is None:
        # Check normality
        _, p1 = shapiro(group1) if len(group1) >= 3 else (0, 0)
        _, p2 = shapiro(group2) if len(group2) >= 3 else (0, 0)
        parametric = (p1 > 0.05 and p2 > 0.05)
    
    if parametric:
        # Independent t-test
        statistic, p_value = ttest_ind(group1, group2, alternative=alternative)
        test_name = "Independent t-test"
    else:
        # Mann-Whitney U test (non-parametric)
        statistic, p_value = mannwhitneyu(group1, group2, alternative=alternative)
        test_name = "Mann-Whitney U test"
    
    # Effect size
    effect_size = compute_cohens_d(group1, group2)
    
    # Confidence interval for difference
    diff = np.mean(group1) - np.mean(group2)
    ci_lower, ci_upper = compute_confidence_interval(
        np.concatenate([group1, -group2])
    )
    
    # Interpretation
    if p_value < 0.001:
        sig_str = "highly significant (p < 0.001)"
    elif p_value < 0.01:
        sig_str = "very significant (p < 0.01)"
    elif p_value < 0.05:
        sig_str = "significant (p < 0.05)"
    else:
        sig_str = "not significant (p >= 0.05)"
    
    if abs(effect_size) < 0.2:
        effect_str = "negligible effect"
    elif abs(effect_size) < 0.5:
        effect_str = "small effect"
    elif abs(effect_size) < 0.8:
        effect_str = "medium effect"
    else:
        effect_str = "large effect"
    
    interpretation = f"{sig_str}, {effect_str} (d = {effect_size:.3f})"
    
    return StatisticalTestResult(
        test_name=test_name,
        statistic=float(statistic),
        p_value=float(p_value),
        significant=bool(p_value < 0.05),
        effect_size=float(effect_size),
        confidence_interval=(float(ci_lower), float(ci_upper)),
        interpretation=interpretation
    )


def paired_samples_test(
    group1: np.ndarray,
    group2: np.ndarray,
    parametric: Optional[bool] = None,
    alternative: str = 'two-sided'
) -> StatisticalTestResult:
    """
    Test if two paired/related groups have different means/medians
    
    Use when samples are matched (e.g., same scenarios tested with different models)
    
    Automatically selects:
    - If parametric or normal: Paired t-test
    - If non-parametric or non-normal: Wilcoxon signed-rank test
    
    Args:
        group1, group2: Paired arrays of values (must be same length)
        parametric: Force parametric or non-parametric
        alternative: 'two-sided', 'less', or 'greater'
        
    Returns:
        StatisticalTestResult
    """
    group1 = np.array(group1).flatten()
    group2 = np.array(group2).flatten()
    
    if len(group1) != len(group2):
        return StatisticalTestResult(
            test_name="Error",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            interpretation="Groups must have same length for paired test"
        )
    
    # Remove paired NaN
    valid_mask = ~(np.isnan(group1) | np.isnan(group2))
    group1 = group1[valid_mask]
    group2 = group2[valid_mask]
    
    if len(group1) < 2:
        return StatisticalTestResult(
            test_name="Insufficient Data",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            interpretation="Not enough valid pairs for testing"
        )
    
    # Differences
    differences = group1 - group2
    
    # Decide test type
    if parametric is None:
        _, p_norm = shapiro(differences) if len(differences) >= 3 else (0, 0)
        parametric = (p_norm > 0.05)
    
    if parametric:
        # Paired t-test
        statistic, p_value = ttest_rel(group1, group2, alternative=alternative)
        test_name = "Paired t-test"
    else:
        # Wilcoxon signed-rank test
        statistic, p_value = wilcoxon(group1, group2, alternative=alternative)
        test_name = "Wilcoxon signed-rank test"
    
    # Effect size
    effect_size = compute_cohens_d(group1, group2)
    
    # Confidence interval
    ci_lower, ci_upper = compute_confidence_interval(differences)
    
    # Interpretation
    mean_diff = np.mean(differences)
    
    if p_value < 0.001:
        sig_str = "highly significant (p < 0.001)"
    elif p_value < 0.01:
        sig_str = "very significant (p < 0.01)"
    elif p_value < 0.05:
        sig_str = "significant (p < 0.05)"
    else:
        sig_str = "not significant (p >= 0.05)"
    
    interpretation = f"{sig_str}, mean difference = {mean_diff:.3f}, d = {effect_size:.3f}"
    
    return StatisticalTestResult(
        test_name=test_name,
        statistic=float(statistic),
        p_value=float(p_value),
        significant=bool(p_value < 0.05),
        effect_size=float(effect_size),
        confidence_interval=(float(ci_lower), float(ci_upper)),
        interpretation=interpretation
    )


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Apply Bonferroni correction for multiple comparisons
    
    Adjusted significance level: alpha / num_tests
    
    Args:
        p_values: List of p-values from multiple tests
        alpha: Original significance level
        
    Returns:
        List of booleans indicating significance after correction
    """
    n_tests = len(p_values)
    adjusted_alpha = alpha / n_tests
    
    return [p < adjusted_alpha for p in p_values]


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Apply Holm-Bonferroni correction (less conservative than Bonferroni)
    
    Args:
        p_values: List of p-values from multiple tests
        alpha: Original significance level
        
    Returns:
        List of booleans indicating significance after correction
    """
    n_tests = len(p_values)
    
    # Sort p-values with original indices
    indexed_p_values = [(p, i) for i, p in enumerate(p_values)]
    indexed_p_values.sort(key=lambda x: x[0])
    
    # Test each p-value against adjusted alpha
    rejected = [False] * n_tests
    
    for rank, (p_value, original_index) in enumerate(indexed_p_values, start=1):
        adjusted_alpha = alpha / (n_tests - rank + 1)
        if p_value < adjusted_alpha:
            rejected[original_index] = True
        else:
            # Once we fail to reject, stop (all subsequent also fail)
            break
    
    return rejected


def compare_multiple_groups(
    groups: List[np.ndarray],
    group_names: Optional[List[str]] = None,
    parametric: Optional[bool] = None
) -> Dict[str, StatisticalTestResult]:
    """
    Perform pairwise comparisons between multiple groups
    
    Args:
        groups: List of arrays (one per group)
        group_names: Optional names for groups
        parametric: Force parametric or non-parametric tests
        
    Returns:
        Dictionary of comparison results {comparison_name: result}
    """
    n_groups = len(groups)
    
    if group_names is None:
        group_names = [f"Group{i+1}" for i in range(n_groups)]
    
    results = {}
    p_values = []
    comparison_names = []
    
    # Pairwise comparisons
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            name = f"{group_names[i]}_vs_{group_names[j]}"
            comparison_names.append(name)
            
            result = independent_samples_test(
                groups[i], groups[j], parametric=parametric
            )
            results[name] = result
            p_values.append(result.p_value)
    
    # Apply multiple comparison correction
    if len(p_values) > 1:
        corrected = holm_bonferroni_correction(p_values)
        
        for name, is_significant in zip(comparison_names, corrected):
            results[name].significant = is_significant
            if is_significant:
                results[name].interpretation += " (survives Holm-Bonferroni correction)"
            else:
                results[name].interpretation += " (not significant after correction)"
    
    return results


def compute_summary_statistics(
    data_dict: Dict[str, np.ndarray]
) -> Dict[str, DistributionAnalysis]:
    """
    Compute summary statistics for multiple datasets
    
    Args:
        data_dict: Dictionary {name: data_array}
        
    Returns:
        Dictionary {name: DistributionAnalysis}
    """
    summaries = {}
    
    for name, data in data_dict.items():
        summaries[name] = analyze_distribution(data)
    
    return summaries


def test_result_to_dict(result: StatisticalTestResult) -> Dict:
    """Convert StatisticalTestResult to dictionary"""
    return {
        'test_name': result.test_name,
        'statistic': result.statistic,
        'p_value': result.p_value,
        'significant': result.significant,
        'effect_size': result.effect_size,
        'confidence_interval': result.confidence_interval,
        'interpretation': result.interpretation
    }


def distribution_to_dict(dist: DistributionAnalysis) -> Dict:
    """Convert DistributionAnalysis to dictionary"""
    return {
        'mean': dist.mean,
        'median': dist.median,
        'std': dist.std,
        'min': dist.min,
        'max': dist.max,
        'q25': dist.q25,
        'q75': dist.q75,
        'skewness': dist.skewness,
        'kurtosis': dist.kurtosis,
        'is_normal': dist.is_normal,
        'normality_p_value': dist.normality_p_value
    }


# Example usage
if __name__ == "__main__":
    print("\n=== Statistical Analysis Test ===\n")
    
    # Generate synthetic data
    np.random.seed(42)
    
    drl_scores = np.random.normal(0.75, 0.1, 50)  # DRL model
    vga_scores = np.random.normal(0.80, 0.08, 50)  # VGA model
    human_scores = np.random.normal(0.85, 0.05, 50)  # Human data
    
    # Distribution analysis
    print("Distribution Analysis:")
    print("-" * 40)
    for name, data in [("DRL", drl_scores), ("VGA", vga_scores), ("Human", human_scores)]:
        dist = analyze_distribution(data)
        print(f"\n{name}:")
        print(f"  Mean: {dist.mean:.3f} ± {dist.std:.3f}")
        print(f"  Median: {dist.median:.3f}")
        print(f"  Range: [{dist.min:.3f}, {dist.max:.3f}]")
        print(f"  Normal: {dist.is_normal} (p = {dist.normality_p_value:.3f})")
    
    # Pairwise comparisons
    print("\n\nPairwise Comparisons:")
    print("-" * 40)
    
    result1 = independent_samples_test(drl_scores, vga_scores)
    print(f"\nDRL vs VGA:")
    print(f"  {result1.interpretation}")
    
    result2 = independent_samples_test(drl_scores, human_scores)
    print(f"\nDRL vs Human:")
    print(f"  {result2.interpretation}")
    
    result3 = independent_samples_test(vga_scores, human_scores)
    print(f"\nVGA vs Human:")
    print(f"  {result3.interpretation}")
    
    # Multiple comparisons
    print("\n\nMultiple Comparisons (with correction):")
    print("-" * 40)
    
    groups = [drl_scores, vga_scores, human_scores]
    names = ["DRL", "VGA", "Human"]
    
    results = compare_multiple_groups(groups, names)
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  {result.interpretation}")
