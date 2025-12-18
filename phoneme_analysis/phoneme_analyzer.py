"""
Phoneme Error Analysis Module

This module analyzes pronunciation by comparing canonical (native-like) and
spoken phoneme sequences. It computes:
- Phoneme Error Rate (PER)
- Substitution, Deletion, and Insertion counts
- Pronunciation difficulty features
- Nationality-aware analysis

Author: SHL AI Research Team
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PhonemeErrorStats:
    """
    Container for phoneme error statistics.
    
    Attributes:
        total_phonemes: Total number of phonemes in canonical sequence
        substitutions: Number of substitution errors
        deletions: Number of deletion errors
        insertions: Number of insertion errors
        correct: Number of correct phonemes
        per: Phoneme Error Rate (0-1, lower is better)
        substitution_rate: Substitution rate
        deletion_rate: Deletion rate
        insertion_rate: Insertion rate
        error_details: List of error details (type, canonical, spoken, position)
    """
    total_phonemes: int
    substitutions: int
    deletions: int
    insertions: int
    correct: int
    per: float
    substitution_rate: float
    deletion_rate: float
    insertion_rate: float
    error_details: List[Dict]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "total_phonemes": self.total_phonemes,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "correct": self.correct,
            "per": self.per,
            "substitution_rate": self.substitution_rate,
            "deletion_rate": self.deletion_rate,
            "insertion_rate": self.insertion_rate,
            "error_count": len(self.error_details)
        }


class PhonemeAnalyzer:
    """
    Analyzer for phoneme-level pronunciation errors.
    
    Uses dynamic programming (Levenshtein distance) to align canonical
    and spoken phoneme sequences and identify errors.
    """
    
    def __init__(self):
        """Initialize phoneme analyzer."""
        pass
    
    def parse_phoneme_sequence(self, phoneme_str: str) -> List[str]:
        """
        Parse phoneme sequence string into list of phonemes.
        
        Args:
            phoneme_str: Space-separated phoneme string
            
        Returns:
            List of phoneme strings
        """
        if not phoneme_str or not phoneme_str.strip():
            return []
        
        phonemes = [p.strip() for p in phoneme_str.split() if p.strip()]
        
        # Remove any phonemes containing "*" (marked as errors/uncertain)
        phonemes = [p for p in phonemes if "*" not in p]
        
        return phonemes
    
    def compute_alignment(
        self,
        canonical: List[str],
        spoken: List[str]
    ) -> Tuple[List[Tuple[str, str]], List[str]]:
        """
        Compute optimal alignment between canonical and spoken phoneme sequences.
        
        Uses dynamic programming (Wagner-Fischer algorithm) to find the
        minimum edit distance alignment.
        
        Args:
            canonical: Canonical (reference) phoneme sequence
            spoken: Spoken (actual) phoneme sequence
            
        Returns:
            Tuple of (aligned_pairs, operations)
            - aligned_pairs: List of (canonical_phoneme, spoken_phoneme) tuples
            - operations: List of operation types ("match", "sub", "del", "ins")
        """
        n = len(canonical)
        m = len(spoken)
        
        # Initialize DP table
        # dp[i][j] = minimum edit distance between canonical[:i] and spoken[:j]
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        
        # Initialize base cases
        for i in range(1, n + 1):
            dp[i][0] = i  # Deletions
        for j in range(1, m + 1):
            dp[0][j] = j  # Insertions
        
        # Fill DP table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if canonical[i-1] == spoken[j-1]:
                    # Match
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # Choose minimum of substitution, deletion, insertion
                    dp[i][j] = min(
                        dp[i-1][j-1] + 1,  # Substitution
                        dp[i-1][j] + 1,     # Deletion
                        dp[i][j-1] + 1      # Insertion
                    )
        
        # Backtrack to find alignment
        aligned_pairs = []
        operations = []
        i, j = n, m
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and canonical[i-1] == spoken[j-1]:
                # Match
                aligned_pairs.append((canonical[i-1], spoken[j-1]))
                operations.append("match")
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                # Substitution
                aligned_pairs.append((canonical[i-1], spoken[j-1]))
                operations.append("sub")
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                # Deletion
                aligned_pairs.append((canonical[i-1], None))
                operations.append("del")
                i -= 1
            else:
                # Insertion
                aligned_pairs.append((None, spoken[j-1]))
                operations.append("ins")
                j -= 1
        
        # Reverse to get forward alignment
        aligned_pairs.reverse()
        operations.reverse()
        
        return aligned_pairs, operations
    
    def analyze_errors(
        self,
        canonical: str,
        spoken: str
    ) -> PhonemeErrorStats:
        """
        Analyze phoneme errors between canonical and spoken sequences.
        
        Args:
            canonical: Canonical phoneme sequence (space-separated)
            spoken: Spoken phoneme sequence (space-separated)
            
        Returns:
            PhonemeErrorStats object
        """
        # Parse sequences
        canonical_phonemes = self.parse_phoneme_sequence(canonical)
        spoken_phonemes = self.parse_phoneme_sequence(spoken)
        
        if not canonical_phonemes:
            logger.warning("Empty canonical phoneme sequence")
            return PhonemeErrorStats(
                total_phonemes=0,
                substitutions=0,
                deletions=0,
                insertions=0,
                correct=0,
                per=0.0,
                substitution_rate=0.0,
                deletion_rate=0.0,
                insertion_rate=0.0,
                error_details=[]
            )
        
        # Compute alignment
        aligned_pairs, operations = self.compute_alignment(
            canonical_phonemes,
            spoken_phonemes
        )
        
        # Count errors
        substitutions = sum(1 for op in operations if op == "sub")
        deletions = sum(1 for op in operations if op == "del")
        insertions = sum(1 for op in operations if op == "ins")
        correct = sum(1 for op in operations if op == "match")
        
        total_phonemes = len(canonical_phonemes)
        
        # Compute PER (Phoneme Error Rate)
        # PER = (S + D + I) / N, where N is the number of reference phonemes
        total_errors = substitutions + deletions + insertions
        per = total_errors / total_phonemes if total_phonemes > 0 else 0.0
        
        # Compute rates
        substitution_rate = substitutions / total_phonemes if total_phonemes > 0 else 0.0
        deletion_rate = deletions / total_phonemes if total_phonemes > 0 else 0.0
        insertion_rate = insertions / total_phonemes if total_phonemes > 0 else 0.0
        
        # Collect error details
        error_details = []
        position = 0
        for (can_ph, spk_ph), op in zip(aligned_pairs, operations):
            if op != "match":
                error_details.append({
                    "type": op,
                    "canonical": can_ph,
                    "spoken": spk_ph,
                    "position": position
                })
            position += 1
        
        return PhonemeErrorStats(
            total_phonemes=total_phonemes,
            substitutions=substitutions,
            deletions=deletions,
            insertions=insertions,
            correct=correct,
            per=per,
            substitution_rate=substitution_rate,
            deletion_rate=deletion_rate,
            insertion_rate=insertion_rate,
            error_details=error_details
        )
    
    def aggregate_nationality_stats(
        self,
        samples: List[Tuple[str, str, str]]
    ) -> Dict[str, Dict]:
        """
        Aggregate phoneme error statistics by nationality.
        
        Args:
            samples: List of (canonical, spoken, nationality) tuples
            
        Returns:
            Dictionary mapping nationality to aggregated statistics
        """
        nationality_stats = defaultdict(lambda: {
            "total_samples": 0,
            "total_phonemes": 0,
            "total_substitutions": 0,
            "total_deletions": 0,
            "total_insertions": 0,
            "total_errors": 0,
            "avg_per": 0.0,
            "per_values": []
        })
        
        for canonical, spoken, nationality in samples:
            stats = self.analyze_errors(canonical, spoken)
            
            nationality_stats[nationality]["total_samples"] += 1
            nationality_stats[nationality]["total_phonemes"] += stats.total_phonemes
            nationality_stats[nationality]["total_substitutions"] += stats.substitutions
            nationality_stats[nationality]["total_deletions"] += stats.deletions
            nationality_stats[nationality]["total_insertions"] += stats.insertions
            nationality_stats[nationality]["total_errors"] += (
                stats.substitutions + stats.deletions + stats.insertions
            )
            nationality_stats[nationality]["per_values"].append(stats.per)
        
        # Compute averages
        for nationality in nationality_stats:
            stats = nationality_stats[nationality]
            if stats["total_samples"] > 0:
                stats["avg_per"] = np.mean(stats["per_values"])
                stats["avg_substitution_rate"] = (
                    stats["total_substitutions"] / stats["total_phonemes"]
                    if stats["total_phonemes"] > 0 else 0.0
                )
                stats["avg_deletion_rate"] = (
                    stats["total_deletions"] / stats["total_phonemes"]
                    if stats["total_phonemes"] > 0 else 0.0
                )
                stats["avg_insertion_rate"] = (
                    stats["total_insertions"] / stats["total_phonemes"]
                    if stats["total_phonemes"] > 0 else 0.0
                )
        
        return dict(nationality_stats)
    
    def extract_pronunciation_features(
        self,
        canonical: str,
        spoken: str
    ) -> Dict[str, float]:
        """
        Extract pronunciation difficulty features.
        
        These features can be used as inputs to scoring models.
        
        Args:
            canonical: Canonical phoneme sequence
            spoken: Spoken phoneme sequence
            
        Returns:
            Dictionary of feature names to values
        """
        stats = self.analyze_errors(canonical, spoken)
        
        features = {
            "phoneme_error_rate": stats.per,
            "substitution_rate": stats.substitution_rate,
            "deletion_rate": stats.deletion_rate,
            "insertion_rate": stats.insertion_rate,
            "correct_rate": stats.correct / stats.total_phonemes if stats.total_phonemes > 0 else 0.0,
            "error_density": len(stats.error_details) / stats.total_phonemes if stats.total_phonemes > 0 else 0.0
        }
        
        # Additional features: phoneme sequence length ratio
        canonical_phonemes = self.parse_phoneme_sequence(canonical)
        spoken_phonemes = self.parse_phoneme_sequence(spoken)
        
        if len(canonical_phonemes) > 0:
            features["length_ratio"] = len(spoken_phonemes) / len(canonical_phonemes)
        else:
            features["length_ratio"] = 1.0
        
        return features


def main():
    """
    Example usage of phoneme analyzer.
    """
    analyzer = PhonemeAnalyzer()
    
    # Example: analyze phoneme errors
    canonical = "f ao r dh ah t w eh n t iy"
    spoken = "f ao r dh ah t w eh n t iy ih"
    
    stats = analyzer.analyze_errors(canonical, spoken)
    print(f"PER: {stats.per:.3f}")
    print(f"Substitutions: {stats.substitutions}")
    print(f"Deletions: {stats.deletions}")
    print(f"Insertions: {stats.insertions}")
    
    # Extract features
    features = analyzer.extract_pronunciation_features(canonical, spoken)
    print(f"\nPronunciation features: {features}")


if __name__ == "__main__":
    main()




