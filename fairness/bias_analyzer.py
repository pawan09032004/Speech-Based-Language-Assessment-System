"""
Fairness and Bias Analysis Module

This module analyzes potential bias in scoring across different
nationality groups and provides mitigation suggestions.

Author: SHL AI Research Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class FairnessReport:
    """
    Container for fairness analysis results.
    
    Attributes:
        nationality_stats: Statistics per nationality group
        mean_score_differences: Pairwise mean score differences
        statistical_tests: Results of statistical significance tests
        bias_indicators: Identified bias indicators
        mitigation_suggestions: Suggestions for bias mitigation
    """
    nationality_stats: Dict[str, Dict]
    mean_score_differences: Dict[Tuple[str, str], float]
    statistical_tests: Dict[Tuple[str, str], Dict]
    bias_indicators: List[str]
    mitigation_suggestions: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "nationality_stats": self.nationality_stats,
            "mean_score_differences": {
                f"{n1}_{n2}": diff for (n1, n2), diff in self.mean_score_differences.items()
            },
            "bias_indicators": self.bias_indicators,
            "mitigation_suggestions": self.mitigation_suggestions
        }


class BiasAnalyzer:
    """
    Analyzer for detecting bias in scoring across nationality groups.
    
    Analyzes:
    - Mean score differences
    - Statistical significance
    - Accent bias risks
    - Mitigation strategies
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize bias analyzer.
        
        Args:
            significance_level: Significance level for statistical tests
        """
        self.significance_level = significance_level
    
    def analyze_nationality_scores(
        self,
        scores: List[float],
        nationalities: List[str]
    ) -> Dict[str, Dict]:
        """
        Analyze score statistics by nationality.
        
        Args:
            scores: List of scores
            nationalities: List of nationality labels (same length as scores)
            
        Returns:
            Dictionary mapping nationality to statistics
        """
        nationality_scores = defaultdict(list)
        
        for score, nationality in zip(scores, nationalities):
            nationality_scores[nationality].append(score)
        
        stats_dict = {}
        for nationality, score_list in nationality_scores.items():
            stats_dict[nationality] = {
                "count": len(score_list),
                "mean": np.mean(score_list),
                "std": np.std(score_list),
                "median": np.median(score_list),
                "min": np.min(score_list),
                "max": np.max(score_list),
                "q25": np.percentile(score_list, 25),
                "q75": np.percentile(score_list, 75)
            }
        
        return stats_dict
    
    def compute_mean_differences(
        self,
        nationality_stats: Dict[str, Dict]
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute pairwise mean score differences.
        
        Args:
            nationality_stats: Statistics by nationality
            
        Returns:
            Dictionary mapping (nationality1, nationality2) to mean difference
        """
        differences = {}
        nationalities = list(nationality_stats.keys())
        
        for i, n1 in enumerate(nationalities):
            for n2 in nationalities[i+1:]:
                mean1 = nationality_stats[n1]["mean"]
                mean2 = nationality_stats[n2]["mean"]
                differences[(n1, n2)] = mean1 - mean2
        
        return differences
    
    def test_statistical_significance(
        self,
        scores_by_nationality: Dict[str, List[float]]
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Perform statistical significance tests between nationality groups.
        
        Uses t-test for comparing means.
        
        Args:
            scores_by_nationality: Dictionary mapping nationality to score list
            
        Returns:
            Dictionary mapping (nationality1, nationality2) to test results
        """
        results = {}
        nationalities = list(scores_by_nationality.keys())
        
        for i, n1 in enumerate(nationalities):
            for n2 in nationalities[i+1:]:
                scores1 = scores_by_nationality[n1]
                scores2 = scores_by_nationality[n2]
                
                if len(scores1) < 2 or len(scores2) < 2:
                    continue
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(scores1, scores2)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(
                    ((len(scores1) - 1) * np.var(scores1) + (len(scores2) - 1) * np.var(scores2)) /
                    (len(scores1) + len(scores2) - 2)
                )
                if pooled_std > 0:
                    cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
                else:
                    cohens_d = 0.0
                
                results[(n1, n2)] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < self.significance_level,
                    "cohens_d": cohens_d,
                    "effect_size": "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
                }
        
        return results
    
    def identify_bias_indicators(
        self,
        nationality_stats: Dict[str, Dict],
        mean_differences: Dict[Tuple[str, str], float],
        statistical_tests: Dict[Tuple[str, str], Dict],
        threshold: float = 5.0
    ) -> List[str]:
        """
        Identify potential bias indicators.
        
        Args:
            nationality_stats: Statistics by nationality
            mean_differences: Mean score differences
            statistical_tests: Statistical test results
            threshold: Threshold for significant mean difference (points)
            
        Returns:
            List of bias indicator strings
        """
        indicators = []
        
        # Check for large mean differences
        for (n1, n2), diff in mean_differences.items():
            if abs(diff) > threshold:
                test_result = statistical_tests.get((n1, n2), {})
                if test_result.get("significant", False):
                    indicators.append(
                        f"Significant score difference between {n1} and {n2}: "
                        f"{diff:.2f} points (p={test_result.get('p_value', 0):.4f})"
                    )
        
        # Check for consistent patterns (one group consistently lower)
        nationality_means = {n: stats["mean"] for n, stats in nationality_stats.items()}
        sorted_nationalities = sorted(nationality_means.items(), key=lambda x: x[1])
        
        if len(sorted_nationalities) >= 2:
            lowest_mean = sorted_nationalities[0][1]
            highest_mean = sorted_nationalities[-1][1]
            gap = highest_mean - lowest_mean
            
            if gap > threshold * 2:
                indicators.append(
                    f"Large score gap between lowest ({sorted_nationalities[0][0]}: {lowest_mean:.2f}) "
                    f"and highest ({sorted_nationalities[-1][0]}: {highest_mean:.2f}) groups: {gap:.2f} points"
                )
        
        # Check for accent bias risks
        # (Simplified: groups with lower scores may indicate accent bias)
        for nationality, stats_dict in nationality_stats.items():
            if stats_dict["mean"] < 50 and stats_dict["count"] >= 10:
                indicators.append(
                    f"Potential accent bias risk: {nationality} group has mean score "
                    f"{stats_dict['mean']:.2f} (below 50) with {stats_dict['count']} samples"
                )
        
        return indicators
    
    def generate_mitigation_suggestions(
        self,
        bias_indicators: List[str],
        nationality_stats: Dict[str, Dict]
    ) -> List[str]:
        """
        Generate suggestions for bias mitigation.
        
        Args:
            bias_indicators: List of identified bias indicators
            nationality_stats: Statistics by nationality
            
        Returns:
            List of mitigation suggestions
        """
        suggestions = []
        
        if not bias_indicators:
            suggestions.append("No significant bias detected. Continue monitoring.")
            return suggestions
        
        # General suggestions
        suggestions.append(
            "Consider collecting more balanced data across nationality groups "
            "to ensure fair representation."
        )
        
        suggestions.append(
            "Review pronunciation scoring criteria to ensure they are not "
            "biased against specific accents or L1 backgrounds."
        )
        
        suggestions.append(
            "Implement nationality-aware calibration: adjust scoring thresholds "
            "or use separate models per nationality group if significant differences persist."
        )
        
        suggestions.append(
            "Conduct human expert evaluation to validate that score differences "
            "reflect actual proficiency differences rather than accent bias."
        )
        
        suggestions.append(
            "Use adversarial training or fairness constraints in model training "
            "to reduce nationality-based bias."
        )
        
        # Check for imbalanced data
        counts = [stats["count"] for stats in nationality_stats.values()]
        if max(counts) / min(counts) > 3:
            suggestions.append(
                "Data is imbalanced across nationality groups. Consider "
                "oversampling underrepresented groups or using stratified sampling."
            )
        
        return suggestions
    
    def analyze(
        self,
        scores: List[float],
        nationalities: List[str]
    ) -> FairnessReport:
        """
        Complete fairness analysis.
        
        Args:
            scores: List of scores
            nationalities: List of nationality labels
            
        Returns:
            FairnessReport object
        """
        # Analyze statistics by nationality
        nationality_stats = self.analyze_nationality_scores(scores, nationalities)
        
        # Compute mean differences
        mean_differences = self.compute_mean_differences(nationality_stats)
        
        # Prepare data for statistical tests
        scores_by_nationality = defaultdict(list)
        for score, nationality in zip(scores, nationalities):
            scores_by_nationality[nationality].append(score)
        
        # Perform statistical tests
        statistical_tests = self.test_statistical_significance(scores_by_nationality)
        
        # Identify bias indicators
        bias_indicators = self.identify_bias_indicators(
            nationality_stats,
            mean_differences,
            statistical_tests
        )
        
        # Generate mitigation suggestions
        mitigation_suggestions = self.generate_mitigation_suggestions(
            bias_indicators,
            nationality_stats
        )
        
        return FairnessReport(
            nationality_stats=nationality_stats,
            mean_score_differences=mean_differences,
            statistical_tests=statistical_tests,
            bias_indicators=bias_indicators,
            mitigation_suggestions=mitigation_suggestions
        )
    
    def print_report(self, report: FairnessReport):
        """Print fairness report to console."""
        print("\n" + "="*60)
        print("FAIRNESS AND BIAS ANALYSIS REPORT")
        print("="*60)
        
        print("\nScore Statistics by Nationality:")
        for nationality, stats in report.nationality_stats.items():
            print(f"\n  {nationality}:")
            print(f"    Count: {stats['count']}")
            print(f"    Mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
            print(f"    Median: {stats['median']:.2f}")
            print(f"    Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
        
        print("\nMean Score Differences:")
        for (n1, n2), diff in report.mean_score_differences.items():
            test_result = report.statistical_tests.get((n1, n2), {})
            sig = "***" if test_result.get("significant", False) else ""
            print(f"  {n1} vs {n2}: {diff:+.2f} {sig}")
            if test_result:
                print(f"    (p={test_result['p_value']:.4f}, d={test_result['cohens_d']:.3f})")
        
        if report.bias_indicators:
            print("\n⚠️  Bias Indicators:")
            for indicator in report.bias_indicators:
                print(f"  - {indicator}")
        
        print("\n💡 Mitigation Suggestions:")
        for suggestion in report.mitigation_suggestions:
            print(f"  - {suggestion}")
        
        print("="*60 + "\n")


def main():
    """
    Example usage of bias analyzer.
    """
    analyzer = BiasAnalyzer()
    
    # Example: analyze scores
    scores = [65, 70, 68, 72, 55, 58, 60, 57, 75, 78, 73, 76]
    nationalities = ["Arabic", "Arabic", "Arabic", "Arabic",
                    "Chinese", "Chinese", "Chinese", "Chinese",
                    "Spanish", "Spanish", "Spanish", "Spanish"]
    
    report = analyzer.analyze(scores, nationalities)
    analyzer.print_report(report)


if __name__ == "__main__":
    main()




