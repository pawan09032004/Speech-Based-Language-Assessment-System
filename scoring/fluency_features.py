"""
Fluency Feature Engineering Module

This module combines audio and text features to create comprehensive
fluency indicators for spoken language assessment.

Features include:
- Speaking rate
- Hesitation ratio
- Pause statistics
- Sentence completeness
- Prosodic features

Author: SHL AI Research Team
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FluencyFeatures:
    """
    Container for fluency features.
    
    Attributes:
        speaking_rate: Words per second
        hesitation_ratio: Ratio of hesitation markers to words
        avg_pause_duration: Average pause duration in seconds
        pause_frequency: Pauses per 100 words
        pause_ratio: Ratio of pause time to total time
        sentence_completeness: Ratio of complete sentences
        prosodic_variability: Variability in pitch/energy
    """
    speaking_rate: float
    hesitation_ratio: float
    avg_pause_duration: float
    pause_frequency: float
    pause_ratio: float
    sentence_completeness: float
    prosodic_variability: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "speaking_rate": self.speaking_rate,
            "hesitation_ratio": self.hesitation_ratio,
            "avg_pause_duration": self.avg_pause_duration,
            "pause_frequency": self.pause_frequency,
            "pause_ratio": self.pause_ratio,
            "sentence_completeness": self.sentence_completeness,
            "prosodic_variability": self.prosodic_variability
        }
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.speaking_rate,
            self.hesitation_ratio,
            self.avg_pause_duration,
            self.pause_frequency,
            self.pause_ratio,
            self.sentence_completeness,
            self.prosodic_variability
        ])


class FluencyFeatureExtractor:
    """
    Extracts fluency features from audio and text data.
    """
    
    def __init__(self):
        """Initialize fluency feature extractor."""
        pass
    
    def extract_hesitation_markers(self, transcript: str) -> int:
        """
        Count hesitation markers in transcript.
        
        Hesitation markers include: "um", "uh", "er", "ah", "like", "you know"
        
        Args:
            transcript: Text transcript
            
        Returns:
            Count of hesitation markers
        """
        import re
        
        hesitation_patterns = [
            r'\bum\b',
            r'\buh\b',
            r'\ber\b',
            r'\bah\b',
            r'\blike\b',
            r'\byou know\b',
            r'\bwell\b'
        ]
        
        count = 0
        transcript_lower = transcript.lower()
        
        for pattern in hesitation_patterns:
            matches = re.findall(pattern, transcript_lower)
            count += len(matches)
        
        return count
    
    def compute_prosodic_variability(
        self,
        pitch: np.ndarray,
        energy: np.ndarray
    ) -> float:
        """
        Compute prosodic variability from pitch and energy contours.
        
        Higher variability indicates more expressive/prosodic speech.
        
        Args:
            pitch: Pitch contour (may contain NaN)
            energy: Energy contour
            
        Returns:
            Prosodic variability score (0-1)
        """
        # Remove NaN values from pitch
        valid_pitch = pitch[~np.isnan(pitch)]
        
        if len(valid_pitch) == 0:
            return 0.0
        
        # Compute coefficient of variation for pitch
        pitch_mean = np.mean(valid_pitch)
        pitch_std = np.std(valid_pitch)
        pitch_cv = pitch_std / pitch_mean if pitch_mean > 0 else 0.0
        
        # Compute coefficient of variation for energy
        energy_mean = np.mean(energy)
        energy_std = np.std(energy)
        energy_cv = energy_std / energy_mean if energy_mean > 0 else 0.0
        
        # Combined variability (normalized)
        variability = (pitch_cv + energy_cv) / 2.0
        
        # Normalize to 0-1 range (assuming max CV of 1.0)
        return min(variability, 1.0)
    
    def compute_sentence_completeness(self, transcript: str) -> float:
        """
        Compute ratio of complete sentences.
        
        Complete sentences end with proper punctuation (. ! ?)
        
        Args:
            transcript: Text transcript
            
        Returns:
            Ratio of complete sentences (0-1)
        """
        import re
        
        # Split by sentence-ending punctuation
        sentences = re.split(r'[.!?]+', transcript)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Count sentences that end with punctuation
        complete_count = 0
        for sent in sentences:
            # Check if sentence ends with punctuation in original
            # (simplified: assume all split sentences are complete)
            if len(sent) > 0:
                complete_count += 1
        
        # Also check if last sentence ends with punctuation
        if transcript and transcript[-1] in '.!?':
            # All sentences are complete
            return 1.0
        
        # Ratio of complete sentences
        return complete_count / len(sentences) if sentences else 0.0
    
    def extract(
        self,
        audio_features: Optional[Dict] = None,
        asr_result: Optional[Dict] = None,
        transcript: Optional[str] = None
    ) -> FluencyFeatures:
        """
        Extract comprehensive fluency features.
        
        Args:
            audio_features: Dictionary containing audio features (from AudioFeatures)
            asr_result: Dictionary containing ASR results (from ASRResult)
            transcript: Text transcript (optional, can be extracted from asr_result)
            
        Returns:
            FluencyFeatures object
        """
        # Get transcript
        if transcript is None and asr_result:
            transcript = asr_result.get("transcript", "")
        elif transcript is None:
            transcript = ""
        
        # Get speaking rate
        if asr_result:
            speaking_rate = asr_result.get("speaking_speed", 0.0)
        elif audio_features and audio_features.get("speech_rate"):
            speaking_rate = audio_features["speech_rate"]
        else:
            speaking_rate = 0.0
        
        # Get pause information
        pause_durations = []
        if audio_features:
            pause_durations = audio_features.get("pause_durations", [])
        
        # Compute pause statistics
        if pause_durations:
            avg_pause_duration = np.mean(pause_durations)
            total_pause_time = sum(pause_durations)
        else:
            avg_pause_duration = 0.0
            total_pause_time = 0.0
        
        # Get audio duration
        audio_duration = 0.0
        if audio_features:
            audio_duration = audio_features.get("duration", 0.0)
        
        # Pause ratio (pause time / total time)
        pause_ratio = total_pause_time / audio_duration if audio_duration > 0 else 0.0
        
        # Pause frequency (pauses per 100 words)
        word_count = len(transcript.split()) if transcript else 0
        pause_frequency = (len(pause_durations) / word_count * 100) if word_count > 0 else 0.0
        
        # Hesitation ratio
        hesitation_count = self.extract_hesitation_markers(transcript)
        hesitation_ratio = hesitation_count / word_count if word_count > 0 else 0.0
        
        # Sentence completeness
        sentence_completeness = self.compute_sentence_completeness(transcript)
        
        # Prosodic variability
        prosodic_variability = 0.0
        if audio_features:
            pitch = audio_features.get("pitch")
            energy = audio_features.get("energy")
            if pitch is not None and energy is not None:
                prosodic_variability = self.compute_prosodic_variability(
                    np.array(pitch),
                    np.array(energy)
                )
        
        return FluencyFeatures(
            speaking_rate=speaking_rate,
            hesitation_ratio=hesitation_ratio,
            avg_pause_duration=avg_pause_duration,
            pause_frequency=pause_frequency,
            pause_ratio=pause_ratio,
            sentence_completeness=sentence_completeness,
            prosodic_variability=prosodic_variability
        )
    
    def normalize_features(
        self,
        features: FluencyFeatures,
        normalization_stats: Optional[Dict] = None
    ) -> FluencyFeatures:
        """
        Normalize fluency features using provided statistics.
        
        Args:
            features: FluencyFeatures object
            normalization_stats: Dictionary with mean and std for each feature
            
        Returns:
            Normalized FluencyFeatures object
        """
        if normalization_stats is None:
            # Use default normalization (assume typical ranges)
            normalization_stats = {
                "speaking_rate": {"mean": 2.0, "std": 1.0},
                "hesitation_ratio": {"mean": 0.1, "std": 0.1},
                "avg_pause_duration": {"mean": 0.5, "std": 0.3},
                "pause_frequency": {"mean": 5.0, "std": 5.0},
                "pause_ratio": {"mean": 0.1, "std": 0.1},
                "sentence_completeness": {"mean": 0.8, "std": 0.2},
                "prosodic_variability": {"mean": 0.3, "std": 0.2}
            }
        
        normalized = {}
        for key, value in features.to_dict().items():
            if key in normalization_stats:
                stats = normalization_stats[key]
                mean = stats.get("mean", 0.0)
                std = stats.get("std", 1.0)
                if std > 0:
                    normalized[key] = (value - mean) / std
                else:
                    normalized[key] = 0.0
            else:
                normalized[key] = value
        
        return FluencyFeatures(**normalized)


def main():
    """
    Example usage of fluency feature extractor.
    """
    extractor = FluencyFeatureExtractor()
    
    # Example: extract features
    audio_features = {
        "duration": 5.0,
        "pause_durations": [0.2, 0.3, 0.1],
        "pitch": np.array([100, 120, 110, 105, 115]),
        "energy": np.array([0.5, 0.6, 0.55, 0.52, 0.58]),
        "speech_rate": 2.5
    }
    
    asr_result = {
        "transcript": "I um went to the store. You know, it was nice.",
        "speaking_speed": 2.5,
        "word_count": 12
    }
    
    features = extractor.extract(
        audio_features=audio_features,
        asr_result=asr_result
    )
    
    print(f"Fluency features: {features.to_dict()}")


if __name__ == "__main__":
    main()




