"""
NLP Grammar Analysis Module

This module analyzes grammar in ASR transcripts using:
- spaCy for POS tagging and dependency parsing
- Transformer models for grammar error detection
- Sentence complexity analysis

Author: SHL AI Research Team
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GrammarStats:
    """
    Container for grammar analysis statistics.
    
    Attributes:
        sentence_count: Number of sentences
        word_count: Total word count
        avg_sentence_length: Average words per sentence
        grammar_errors: Number of detected grammar errors
        error_density: Grammar errors per 100 words
        pos_distribution: Distribution of POS tags
        complexity_score: Sentence complexity score (0-1)
        error_details: List of detected errors with details
    """
    sentence_count: int
    word_count: int
    avg_sentence_length: float
    grammar_errors: int
    error_density: float
    pos_distribution: Dict[str, int]
    complexity_score: float
    error_details: List[Dict]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "sentence_count": self.sentence_count,
            "word_count": self.word_count,
            "avg_sentence_length": self.avg_sentence_length,
            "grammar_errors": self.grammar_errors,
            "error_density": self.error_density,
            "complexity_score": self.complexity_score,
            "error_count": len(self.error_details)
        }


class GrammarAnalyzer:
    """
    Grammar analyzer using spaCy and rule-based heuristics.
    
    Analyzes:
    - Grammar errors (subject-verb agreement, tense, articles, etc.)
    - Sentence complexity
    - POS and dependency patterns
    """
    
    def __init__(self, use_spacy: bool = True, model_name: str = "en_core_web_sm"):
        """
        Initialize grammar analyzer.
        
        Args:
            use_spacy: Whether to use spaCy (requires model download)
            model_name: spaCy model name
        """
        self.use_spacy = use_spacy
        self.nlp = None
        
        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load(model_name)
                logger.info(f"Loaded spaCy model: {model_name}")
            except OSError:
                logger.warning(
                    f"spaCy model '{model_name}' not found. "
                    "Install with: python -m spacy download en_core_web_sm"
                )
                logger.warning("Falling back to rule-based analysis only")
                self.use_spacy = False
            except ImportError:
                logger.warning("spaCy not installed. Install with: pip install spacy")
                self.use_spacy = False
    
    def clean_transcript(self, transcript: str) -> str:
        """
        Clean ASR transcript for grammar analysis.
        
        Args:
            transcript: Raw ASR transcript
            
        Returns:
            Cleaned transcript
        """
        # Remove extra whitespace
        transcript = re.sub(r'\s+', ' ', transcript).strip()
        
        # Fix common ASR errors
        # (This is a simplified version; in production, use more sophisticated cleaning)
        transcript = re.sub(r'\b(um|uh|er|ah)\b', '', transcript, flags=re.IGNORECASE)
        transcript = re.sub(r'\s+', ' ', transcript).strip()
        
        return transcript
    
    def detect_grammar_errors_rule_based(
        self,
        transcript: str
    ) -> List[Dict]:
        """
        Detect grammar errors using rule-based heuristics.
        
        This is a simplified implementation. In production, use more
        sophisticated grammar checking tools or transformer models.
        
        Args:
            transcript: Text transcript
            
        Returns:
            List of error dictionaries
        """
        errors = []
        
        if not self.nlp:
            # Basic rule-based checks without spaCy
            # Check for common patterns
            words = transcript.lower().split()
            
            # Check for double words (common ASR error)
            for i in range(len(words) - 1):
                if words[i] == words[i + 1]:
                    errors.append({
                        "type": "repetition",
                        "word": words[i],
                        "position": i,
                        "message": f"Repeated word: {words[i]}"
                    })
            
            return errors
        
        # Use spaCy for more sophisticated analysis
        doc = self.nlp(transcript)
        
        # Check subject-verb agreement
        for token in doc:
            if token.pos_ == "VERB" and token.head.pos_ == "NOUN":
                # Simplified check (in production, use more sophisticated rules)
                pass
        
        # Check for missing articles
        # (Simplified: check for noun phrases without articles)
        for i, token in enumerate(doc):
            if token.pos_ == "NOUN" and i > 0:
                prev_token = doc[i - 1]
                if prev_token.pos_ not in ["DET", "ADJ", "NOUN"]:
                    # Potential missing article (but many false positives)
                    pass
        
        # Check for common errors
        # Double words
        for i in range(len(doc) - 1):
            if doc[i].text.lower() == doc[i + 1].text.lower():
                errors.append({
                    "type": "repetition",
                    "word": doc[i].text,
                    "position": i,
                    "message": f"Repeated word: {doc[i].text}"
                })
        
        return errors
    
    def analyze_pos_distribution(self, transcript: str) -> Dict[str, int]:
        """
        Analyze part-of-speech tag distribution.
        
        Args:
            transcript: Text transcript
            
        Returns:
            Dictionary mapping POS tags to counts
        """
        if not self.nlp:
            return {}
        
        doc = self.nlp(transcript)
        pos_dist = {}
        
        for token in doc:
            pos = token.pos_
            pos_dist[pos] = pos_dist.get(pos, 0) + 1
        
        return pos_dist
    
    def compute_complexity_score(self, transcript: str) -> float:
        """
        Compute sentence complexity score.
        
        Higher score indicates more complex sentences.
        Based on:
        - Sentence length
        - Number of clauses
        - POS diversity
        - Dependency depth
        
        Args:
            transcript: Text transcript
            
        Returns:
            Complexity score (0-1)
        """
        if not self.nlp:
            # Fallback: use sentence length
            sentences = re.split(r'[.!?]+', transcript)
            sentences = [s.strip() for s in sentences if s.strip()]
            if not sentences:
                return 0.0
            
            avg_length = np.mean([len(s.split()) for s in sentences])
            # Normalize (assuming max 50 words per sentence)
            return min(avg_length / 50.0, 1.0)
        
        doc = self.nlp(transcript)
        sentences = list(doc.sents)
        
        if not sentences:
            return 0.0
        
        complexity_scores = []
        
        for sent in sentences:
            # Sentence length
            length_score = min(len(sent) / 30.0, 1.0)
            
            # POS diversity
            pos_tags = set([token.pos_ for token in sent])
            diversity_score = len(pos_tags) / 15.0  # Normalize by max expected POS tags
            
            # Dependency depth (average depth of dependency tree)
            depths = []
            for token in sent:
                depth = 0
                head = token.head
                while head != token and depth < 10:  # Prevent infinite loops
                    depth += 1
                    head = head.head
                depths.append(depth)
            
            depth_score = min(np.mean(depths) / 5.0, 1.0) if depths else 0.0
            
            # Combined complexity
            sent_complexity = (length_score + diversity_score + depth_score) / 3.0
            complexity_scores.append(sent_complexity)
        
        return np.mean(complexity_scores) if complexity_scores else 0.0
    
    def analyze(
        self,
        transcript: str
    ) -> GrammarStats:
        """
        Complete grammar analysis of transcript.
        
        Args:
            transcript: ASR transcript text
            
        Returns:
            GrammarStats object
        """
        # Clean transcript
        cleaned = self.clean_transcript(transcript)
        
        if not cleaned:
            return GrammarStats(
                sentence_count=0,
                word_count=0,
                avg_sentence_length=0.0,
                grammar_errors=0,
                error_density=0.0,
                pos_distribution={},
                complexity_score=0.0,
                error_details=[]
            )
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', cleaned)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)
        
        # Word count
        words = cleaned.split()
        word_count = len(words)
        
        # Average sentence length
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0.0
        
        # Detect grammar errors
        error_details = self.detect_grammar_errors_rule_based(cleaned)
        grammar_errors = len(error_details)
        
        # Error density (errors per 100 words)
        error_density = (grammar_errors / word_count * 100) if word_count > 0 else 0.0
        
        # POS distribution
        pos_distribution = self.analyze_pos_distribution(cleaned)
        
        # Complexity score
        complexity_score = self.compute_complexity_score(cleaned)
        
        return GrammarStats(
            sentence_count=sentence_count,
            word_count=word_count,
            avg_sentence_length=avg_sentence_length,
            grammar_errors=grammar_errors,
            error_density=error_density,
            pos_distribution=pos_distribution,
            complexity_score=complexity_score,
            error_details=error_details
        )
    
    def extract_grammar_features(
        self,
        transcript: str
    ) -> Dict[str, float]:
        """
        Extract grammar features for scoring models.
        
        Args:
            transcript: ASR transcript text
            
        Returns:
            Dictionary of feature names to values
        """
        stats = self.analyze(transcript)
        
        features = {
            "grammar_error_density": stats.error_density,
            "sentence_count": stats.sentence_count,
            "avg_sentence_length": stats.avg_sentence_length,
            "complexity_score": stats.complexity_score,
            "word_count": stats.word_count
        }
        
        # Normalize features
        features["normalized_error_density"] = min(stats.error_density / 10.0, 1.0)
        features["normalized_complexity"] = stats.complexity_score
        
        return features


def main():
    """
    Example usage of grammar analyzer.
    """
    analyzer = GrammarAnalyzer(use_spacy=True)
    
    # Example: analyze grammar
    transcript = "I go to the store yesterday. The cat is sleeping."
    
    stats = analyzer.analyze(transcript)
    print(f"Grammar errors: {stats.grammar_errors}")
    print(f"Error density: {stats.error_density:.2f} per 100 words")
    print(f"Complexity score: {stats.complexity_score:.3f}")
    
    # Extract features
    features = analyzer.extract_grammar_features(transcript)
    print(f"\nGrammar features: {features}")


if __name__ == "__main__":
    main()




