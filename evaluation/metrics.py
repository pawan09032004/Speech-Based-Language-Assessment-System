"""
Evaluation Metrics Module

This module provides evaluation metrics and analysis tools for the
speech assessment system.

Metrics include:
- Correlation analysis
- Error distributions
- Score distributions
- Component-wise analysis

Author: SHL AI Research Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.stats import pearsonr, spearmanr
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """
    Container for evaluation metrics.
    
    Attributes:
        score_statistics: Statistics of predicted scores
        correlation_coefficients: Correlation coefficients with ground truth (if available)
        error_distribution: Distribution of errors
        component_correlations: Correlations between components
        limitations: List of identified limitations
    """
    score_statistics: Dict
    correlation_coefficients: Optional[Dict[str, float]]
    error_distribution: Dict
    component_correlations: Dict
    limitations: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "score_statistics": self.score_statistics,
            "correlation_coefficients": self.correlation_coefficients,
            "error_distribution": self.error_distribution,
            "component_correlations": self.component_correlations,
            "limitations": self.limitations
        }


class MetricsEvaluator:
    """
    Evaluator for assessment system metrics.
    """
    
    def __init__(self):
        """Initialize metrics evaluator."""
        pass
    
    def compute_score_statistics(self, scores: List[float]) -> Dict:
        """
        Compute statistics for score distribution.
        
        Args:
            scores: List of scores
            
        Returns:
            Dictionary of statistics
        """
        scores_array = np.array(scores)
        
        return {
            "count": len(scores),
            "mean": float(np.mean(scores_array)),
            "std": float(np.std(scores_array)),
            "median": float(np.median(scores_array)),
            "min": float(np.min(scores_array)),
            "max": float(np.max(scores_array)),
            "q25": float(np.percentile(scores_array, 25)),
            "q75": float(np.percentile(scores_array, 75)),
            "iqr": float(np.percentile(scores_array, 75) - np.percentile(scores_array, 25))
        }
    
    def compute_correlations(
        self,
        predicted: List[float],
        ground_truth: Optional[List[float]] = None,
        components: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, float]:
        """
        Compute correlation coefficients.
        
        Args:
            predicted: Predicted scores
            ground_truth: Ground truth scores (optional)
            components: Component scores (optional)
            
        Returns:
            Dictionary of correlation coefficients
        """
        correlations = {}
        
        # Pearson correlation with ground truth
        if ground_truth is not None:
            if len(predicted) == len(ground_truth):
                pearson_r, pearson_p = pearsonr(predicted, ground_truth)
                spearman_r, spearman_p = spearmanr(predicted, ground_truth)
                
                correlations["pearson_r"] = float(pearson_r)
                correlations["pearson_p"] = float(pearson_p)
                correlations["spearman_r"] = float(spearman_r)
                correlations["spearman_p"] = float(spearman_p)
            else:
                logger.warning("Length mismatch between predicted and ground truth")
        
        # Component correlations
        if components:
            for comp_name, comp_scores in components.items():
                if len(comp_scores) == len(predicted):
                    r, p = pearsonr(predicted, comp_scores)
                    correlations[f"{comp_name}_correlation"] = float(r)
                    correlations[f"{comp_name}_p_value"] = float(p)
        
        return correlations
    
    def analyze_error_distribution(
        self,
        predicted: List[float],
        ground_truth: Optional[List[float]] = None
    ) -> Dict:
        """
        Analyze error distribution.
        
        Args:
            predicted: Predicted scores
            ground_truth: Ground truth scores (optional)
            
        Returns:
            Dictionary of error distribution statistics
        """
        if ground_truth is None:
            return {
                "mean_error": None,
                "std_error": None,
                "mae": None,
                "rmse": None,
                "note": "Ground truth not available for error analysis"
            }
        
        if len(predicted) != len(ground_truth):
            logger.warning("Length mismatch in error analysis")
            return {"error": "Length mismatch"}
        
        errors = np.array(predicted) - np.array(ground_truth)
        
        return {
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "error_range": [float(np.min(errors)), float(np.max(errors))],
            "error_q25": float(np.percentile(errors, 25)),
            "error_q75": float(np.percentile(errors, 75))
        }
    
    def analyze_component_correlations(
        self,
        components: Dict[str, List[float]]
    ) -> Dict:
        """
        Analyze correlations between components.
        
        Args:
            components: Dictionary mapping component names to score lists
            
        Returns:
            Dictionary of component correlations
        """
        correlations = {}
        component_names = list(components.keys())
        
        for i, name1 in enumerate(component_names):
            for name2 in component_names[i+1:]:
                scores1 = components[name1]
                scores2 = components[name2]
                
                if len(scores1) == len(scores2):
                    r, p = pearsonr(scores1, scores2)
                    correlations[f"{name1}_{name2}"] = {
                        "correlation": float(r),
                        "p_value": float(p)
                    }
        
        return correlations
    
    def identify_limitations(
        self,
        scores: List[float],
        ground_truth: Optional[List[float]] = None,
        components: Optional[Dict[str, List[float]]] = None
    ) -> List[str]:
        """
        Identify limitations in the evaluation.
        
        Args:
            scores: Predicted scores
            ground_truth: Ground truth scores (optional)
            components: Component scores (optional)
            
        Returns:
            List of limitation strings
        """
        limitations = []
        
        # Check for ground truth availability
        if ground_truth is None:
            limitations.append(
                "No ground truth labels available. Evaluation is limited to "
                "descriptive statistics and qualitative analysis."
            )
        
        # Check score distribution
        score_stats = self.compute_score_statistics(scores)
        if score_stats["std"] < 5.0:
            limitations.append(
                "Low score variance detected. Model may not be discriminating "
                "well between different proficiency levels."
            )
        
        if score_stats["mean"] < 30 or score_stats["mean"] > 70:
            limitations.append(
                f"Score distribution is skewed (mean={score_stats['mean']:.2f}). "
                "Model may need calibration."
            )
        
        # Check component correlations
        if components:
            comp_corrs = self.analyze_component_correlations(components)
            high_corrs = [
                (names, corr["correlation"])
                for names, corr in comp_corrs.items()
                if abs(corr["correlation"]) > 0.9
            ]
            if high_corrs:
                limitations.append(
                    f"High correlation between components detected: {high_corrs}. "
                    "Components may not be capturing independent aspects."
                )
        
        # Sample size
        if len(scores) < 50:
            limitations.append(
                f"Small sample size ({len(scores)}). Results may not be "
                "statistically reliable."
            )
        
        return limitations
    
    def evaluate(
        self,
        predicted_scores: List[float],
        ground_truth: Optional[List[float]] = None,
        component_scores: Optional[Dict[str, List[float]]] = None
    ) -> EvaluationMetrics:
        """
        Complete evaluation.
        
        Args:
            predicted_scores: Predicted overall scores
            ground_truth: Ground truth scores (optional)
            component_scores: Component scores (optional)
            
        Returns:
            EvaluationMetrics object
        """
        # Score statistics
        score_stats = self.compute_score_statistics(predicted_scores)
        
        # Correlations
        correlations = self.compute_correlations(
            predicted_scores,
            ground_truth,
            component_scores
        )
        
        # Error distribution
        error_dist = self.analyze_error_distribution(predicted_scores, ground_truth)
        
        # Component correlations
        component_corrs = {}
        if component_scores:
            component_corrs = self.analyze_component_correlations(component_scores)
        
        # Limitations
        limitations = self.identify_limitations(
            predicted_scores,
            ground_truth,
            component_scores
        )
        
        return EvaluationMetrics(
            score_statistics=score_stats,
            correlation_coefficients=correlations if correlations else None,
            error_distribution=error_dist,
            component_correlations=component_corrs,
            limitations=limitations
        )
    
    def print_report(self, metrics: EvaluationMetrics):
        """Print evaluation report to console."""
        print("\n" + "="*60)
        print("EVALUATION METRICS REPORT")
        print("="*60)
        
        print("\nScore Statistics:")
        stats = metrics.score_statistics
        print(f"  Count: {stats['count']}")
        print(f"  Mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
        print(f"  Median: {stats['median']:.2f}")
        print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
        print(f"  IQR: [{stats['q25']:.2f}, {stats['q75']:.2f}]")
        
        if metrics.correlation_coefficients:
            print("\nCorrelations:")
            for name, value in metrics.correlation_coefficients.items():
                if "p_value" not in name:
                    print(f"  {name}: {value:.4f}")
        
        if metrics.error_distribution and "mae" in metrics.error_distribution:
            print("\nError Distribution:")
            err_dist = metrics.error_distribution
            print(f"  MAE: {err_dist['mae']:.2f}")
            print(f"  RMSE: {err_dist['rmse']:.2f}")
            print(f"  Mean Error: {err_dist['mean_error']:.2f} ± {err_dist['std_error']:.2f}")
        
        if metrics.component_correlations:
            print("\nComponent Correlations:")
            for names, corr_info in metrics.component_correlations.items():
                print(f"  {names}: r={corr_info['correlation']:.4f}, p={corr_info['p_value']:.4f}")
        
        if metrics.limitations:
            print("\n⚠️  Limitations:")
            for limitation in metrics.limitations:
                print(f"  - {limitation}")
        
        print("="*60 + "\n")


def main():
    """
    Example usage of metrics evaluator.
    """
    evaluator = MetricsEvaluator()
    
    # Example: evaluate scores
    predicted = [65, 70, 68, 72, 55, 58, 60, 57, 75, 78, 73, 76]
    ground_truth = [67, 71, 69, 70, 56, 59, 61, 58, 74, 77, 72, 75]
    
    component_scores = {
        "pronunciation": [70, 72, 71, 73, 60, 62, 64, 61, 78, 80, 76, 79],
        "grammar": [60, 65, 63, 68, 50, 53, 55, 52, 70, 73, 68, 71],
        "fluency": [65, 73, 70, 75, 55, 59, 61, 58, 77, 81, 75, 78]
    }
    
    metrics = evaluator.evaluate(predicted, ground_truth, component_scores)
    evaluator.print_report(metrics)


if __name__ == "__main__":
    main()




