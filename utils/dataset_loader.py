"""
Dataset Loader for L2-ARTIC en_mdd Dataset

This module handles loading and validation of the L2-ARTIC dataset with:
- Audio file paths
- Canonical (native-like) phoneme sequences
- Spoken (actual) phoneme sequences
- Speaker nationality metadata

Author: SHL AI Research Team
"""

import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Nationality mapping from speaker ID suffix
NATIONALITY_MAP = {
    "A": "Arabic",
    "C": "Chinese",
    "I": "Indian",
    "K": "Korean",
    "S": "Spanish",
    "V": "Vietnamese"
}


@dataclass
class AudioSample:
    """
    Unified metadata object for a single audio sample.
    
    Attributes:
        sample_id: Unique identifier for the sample
        audio_path: Path to the audio file (.wav)
        speaker_id: Speaker identifier (e.g., "ABA", "THV")
        nationality: Speaker's L1 nationality
        canonical_phonemes: Native-like phoneme sequence (space-separated)
        spoken_phonemes: Actual spoken phoneme sequence (space-separated)
        split: Dataset split ("train", "dev", or "test")
        audio_exists: Whether the audio file exists at the specified path
    """
    sample_id: str
    audio_path: str
    speaker_id: str
    nationality: str
    canonical_phonemes: str
    spoken_phonemes: str
    split: str
    audio_exists: bool = False
    
    def __post_init__(self):
        """Validate audio file existence after initialization."""
        if os.path.exists(self.audio_path):
            self.audio_exists = True
        elif os.path.exists(self.audio_path + ".wav"):
            self.audio_path = self.audio_path + ".wav"
            self.audio_exists = True


class L2ArticDatasetLoader:
    """
    Dataset loader for L2-ARTIC en_mdd dataset.
    
    Handles:
    - Loading train/dev/test CSV files
    - Parsing speaker IDs and nationalities
    - Validating audio file paths
    - Creating unified metadata objects
    """
    
    def __init__(
        self,
        data_root: Union[str, Path],
        csv_train_path: Optional[str] = None,
        csv_dev_path: Optional[str] = None,
        csv_test_path: Optional[str] = None,
        audio_base_path: Optional[str] = None
    ):
        """
        Initialize the dataset loader.
        
        Args:
            data_root: Root directory containing dataset files
            csv_train_path: Path to training CSV (relative to data_root or absolute)
            csv_dev_path: Path to dev CSV (relative to data_root or absolute)
            csv_test_path: Path to test CSV (relative to data_root or absolute)
            audio_base_path: Base path for audio files (if different from data_root)
        """
        self.data_root = Path(data_root)
        
        # Set default CSV paths if not provided
        if csv_train_path is None:
            csv_train_path = self.data_root / "graph_dataset_l2_artic" / "graph_dataset_l2_artic" / "train.csv"
        else:
            csv_train_path = Path(csv_train_path)
            if not csv_train_path.is_absolute():
                csv_train_path = self.data_root / csv_train_path
        
        if csv_dev_path is None:
            csv_dev_path = self.data_root / "graph_dataset_l2_artic" / "graph_dataset_l2_artic" / "dev.csv"
        else:
            csv_dev_path = Path(csv_dev_path)
            if not csv_dev_path.is_absolute():
                csv_dev_path = self.data_root / csv_dev_path
        
        if csv_test_path is None:
            csv_test_path = self.data_root / "graph_dataset_l2_artic" / "graph_dataset_l2_artic" / "test.csv"
        else:
            csv_test_path = Path(csv_test_path)
            if not csv_test_path.is_absolute():
                csv_test_path = self.data_root / csv_test_path
        
        self.csv_train_path = csv_train_path
        self.csv_dev_path = csv_dev_path
        self.csv_test_path = csv_test_path
        
        # Audio base path
        if audio_base_path is None:
            self.audio_base_path = self.data_root
        else:
            self.audio_base_path = Path(audio_base_path)
        
        # Storage for loaded samples
        self.samples: Dict[str, List[AudioSample]] = {
            "train": [],
            "dev": [],
            "test": []
        }
        
        # Statistics
        self.stats = {
            "total_samples": 0,
            "samples_with_audio": 0,
            "samples_missing_audio": 0,
            "speakers": set(),
            "nationalities": defaultdict(int)
        }
    
    def _extract_speaker_id(self, path_str: str) -> Tuple[str, str]:
        """
        Extract speaker ID and nationality from path string.
        
        Path format: "L2_arctic_WAV/ABA_arctic_a0003"
        Speaker ID: "ABA" (last character indicates nationality)
        
        Args:
            path_str: Path string from CSV
            
        Returns:
            Tuple of (speaker_id, nationality_code)
        """
        # Pattern: L2_arctic_WAV/{SPEAKER_ID}_arctic_{SAMPLE_ID}
        match = re.search(r"L2_arctic_WAV/([A-Z]+)_arctic_", path_str)
        if match:
            speaker_id = match.group(1)
            nationality_code = speaker_id[-1]  # Last character indicates nationality
            return speaker_id, nationality_code
        
        # Fallback: try to extract from any path format
        parts = path_str.split("/")
        if len(parts) > 1:
            filename = parts[-1]
            match = re.search(r"([A-Z]+)_arctic_", filename)
            if match:
                speaker_id = match.group(1)
                nationality_code = speaker_id[-1]
                return speaker_id, nationality_code
        
        logger.warning(f"Could not extract speaker ID from path: {path_str}")
        return "UNKNOWN", "U"
    
    def _parse_phoneme_sequence(self, phoneme_str: str, remove_nationality_tags: bool = False) -> str:
        """
        Parse phoneme sequence from CSV format.
        
        Handles two formats:
        1. Space-separated phonemes: "f ao r dh ah t"
        2. Nationality-tagged phonemes: "f_A ao_A r_A dh_A ah_A"
        
        Args:
            phoneme_str: Phoneme string from CSV
            remove_nationality_tags: If True, remove nationality suffixes (e.g., "_A")
            
        Returns:
            Cleaned phoneme sequence (space-separated)
        """
        if pd.isna(phoneme_str) or not phoneme_str:
            return ""
        
        # Remove nationality tags if present and requested
        if remove_nationality_tags:
            # Pattern: phoneme_NATIONALITY (e.g., "f_A" -> "f")
            phoneme_str = re.sub(r"_([A-Z])$", "", phoneme_str)
            # Also handle cases like "f_A ao_A" -> "f ao"
            phoneme_str = re.sub(r"([a-z]+)_[A-Z]", r"\1", phoneme_str)
        
        # Split by space and filter out empty strings
        phonemes = [p.strip() for p in phoneme_str.split() if p.strip()]
        
        # Remove any phonemes containing "*" (marked as errors/uncertain)
        phonemes = [p for p in phonemes if "*" not in p]
        
        return " ".join(phonemes)
    
    def _resolve_audio_path(self, path_str: str) -> str:
        """
        Resolve full path to audio file.
        
        Args:
            path_str: Path string from CSV (e.g., "L2_arctic_WAV/ABA_arctic_a0003")
            
        Returns:
            Full path to audio file
        """
        # Remove .wav extension if present (we'll add it)
        if path_str.endswith(".wav"):
            path_str = path_str[:-4]
        
        # Try multiple possible locations
        possible_paths = [
            self.audio_base_path / f"{path_str}.wav",
            self.audio_base_path / path_str,
            self.data_root / f"{path_str}.wav",
            self.data_root / path_str,
        ]
        
        # Also try with L2_arctic_WAV prefix
        if "L2_arctic_WAV" not in path_str:
            possible_paths.extend([
                self.audio_base_path / "L2_arctic_WAV" / f"{path_str}.wav",
                self.data_root / "L2_arctic_WAV" / f"{path_str}.wav",
            ])
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        # Return the most likely path even if it doesn't exist
        return str(self.audio_base_path / f"{path_str}.wav")
    
    def load_split(self, split: str) -> List[AudioSample]:
        """
        Load samples from a specific split (train/dev/test).
        
        Args:
            split: Dataset split name ("train", "dev", or "test")
            
        Returns:
            List of AudioSample objects
        """
        if split not in ["train", "dev", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'dev', or 'test'")
        
        csv_path = {
            "train": self.csv_train_path,
            "dev": self.csv_dev_path,
            "test": self.csv_test_path
        }[split]
        
        if not csv_path.exists():
            logger.warning(f"CSV file not found: {csv_path}")
            return []
        
        logger.info(f"Loading {split} split from {csv_path}")
        
        # Load CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")
            return []
        
        # Validate required columns
        required_cols = ["Path", "Canonical", "Transcript"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns in CSV: {missing_cols}")
            return []
        
        samples = []
        for idx, row in df.iterrows():
            try:
                # Extract speaker ID and nationality
                speaker_id, nationality_code = self._extract_speaker_id(row["Path"])
                nationality = NATIONALITY_MAP.get(nationality_code, "Unknown")
                
                # Parse phoneme sequences
                canonical_phonemes = self._parse_phoneme_sequence(row["Canonical"], remove_nationality_tags=True)
                spoken_phonemes = self._parse_phoneme_sequence(row["Transcript"], remove_nationality_tags=True)
                
                # Resolve audio path
                audio_path = self._resolve_audio_path(row["Path"])
                
                # Create sample ID
                sample_id = f"{speaker_id}_{split}_{idx}"
                
                # Create AudioSample
                sample = AudioSample(
                    sample_id=sample_id,
                    audio_path=audio_path,
                    speaker_id=speaker_id,
                    nationality=nationality,
                    canonical_phonemes=canonical_phonemes,
                    spoken_phonemes=spoken_phonemes,
                    split=split
                )
                
                samples.append(sample)
                
                # Update statistics
                self.stats["speakers"].add(speaker_id)
                self.stats["nationalities"][nationality] += 1
                if sample.audio_exists:
                    self.stats["samples_with_audio"] += 1
                else:
                    self.stats["samples_missing_audio"] += 1
                    logger.debug(f"Audio file not found: {audio_path}")
                
            except Exception as e:
                logger.warning(f"Error processing row {idx} in {split} split: {e}")
                continue
        
        self.samples[split] = samples
        self.stats["total_samples"] = len(samples)
        
        logger.info(f"Loaded {len(samples)} samples from {split} split")
        return samples
    
    def load_all(self) -> Dict[str, List[AudioSample]]:
        """
        Load all splits (train, dev, test).
        
        Returns:
            Dictionary mapping split names to lists of AudioSample objects
        """
        logger.info("Loading all dataset splits...")
        
        for split in ["train", "dev", "test"]:
            self.load_split(split)
        
        return self.samples
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            "total_samples": sum(len(samples) for samples in self.samples.values()),
            "train_samples": len(self.samples["train"]),
            "dev_samples": len(self.samples["dev"]),
            "test_samples": len(self.samples["test"]),
            "samples_with_audio": self.stats["samples_with_audio"],
            "samples_missing_audio": self.stats["samples_missing_audio"],
            "unique_speakers": len(self.stats["speakers"]),
            "speakers": sorted(list(self.stats["speakers"])),
            "nationality_distribution": dict(self.stats["nationalities"]),
            "audio_coverage": (
                self.stats["samples_with_audio"] / max(self.stats["total_samples"], 1) * 100
            )
        }
        
        return stats
    
    def print_statistics(self):
        """Print dataset statistics to console."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(f"Total samples: {stats['total_samples']}")
        print(f"  - Train: {stats['train_samples']}")
        print(f"  - Dev: {stats['dev_samples']}")
        print(f"  - Test: {stats['test_samples']}")
        print(f"\nAudio files:")
        print(f"  - Found: {stats['samples_with_audio']}")
        print(f"  - Missing: {stats['samples_missing_audio']}")
        print(f"  - Coverage: {stats['audio_coverage']:.2f}%")
        print(f"\nUnique speakers: {stats['unique_speakers']}")
        print(f"\nNationality distribution:")
        for nationality, count in sorted(stats['nationality_distribution'].items()):
            print(f"  - {nationality}: {count}")
        print("="*60 + "\n")


def main():
    """
    Example usage of the dataset loader.
    """
    # Initialize loader
    loader = L2ArticDatasetLoader(
        data_root=".",
        csv_train_path="graph_dataset_l2_artic/graph_dataset_l2_artic/train.csv",
        csv_dev_path="graph_dataset_l2_artic/graph_dataset_l2_artic/dev.csv",
        csv_test_path="graph_dataset_l2_artic/graph_dataset_l2_artic/test.csv"
    )
    
    # Load all splits
    samples = loader.load_all()
    
    # Print statistics
    loader.print_statistics()
    
    # Example: Access a sample
    if samples["train"]:
        sample = samples["train"][0]
        print(f"\nExample sample:")
        print(f"  Sample ID: {sample.sample_id}")
        print(f"  Speaker: {sample.speaker_id} ({sample.nationality})")
        print(f"  Audio path: {sample.audio_path}")
        print(f"  Audio exists: {sample.audio_exists}")
        print(f"  Canonical phonemes: {sample.canonical_phonemes[:50]}...")
        print(f"  Spoken phonemes: {sample.spoken_phonemes[:50]}...")


if __name__ == "__main__":
    main()

