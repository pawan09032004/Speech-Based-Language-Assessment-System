"""
Audio Preprocessing Module

This module handles audio loading, preprocessing, and feature extraction for
speech-based language assessment.

Features extracted:
- MFCCs (Mel-frequency cepstral coefficients)
- Pitch (fundamental frequency)
- Energy
- Pause duration
- Speech rate

Author: SHL AI Research Team
"""

import os
import numpy as np
import librosa
import torch
import torchaudio
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    """
    Container for extracted audio features.
    
    Attributes:
        mfcc: MFCC features (n_mfcc, n_frames)
        pitch: Pitch contour (fundamental frequency) in Hz (n_frames,)
        energy: Energy contour (n_frames,)
        pause_durations: List of pause durations in seconds
        speech_rate: Words per second (if transcript available)
        duration: Total audio duration in seconds
        sample_rate: Audio sample rate
        raw_audio: Raw audio waveform (optional)
    """
    mfcc: np.ndarray
    pitch: np.ndarray
    energy: np.ndarray
    pause_durations: list
    speech_rate: Optional[float]
    duration: float
    sample_rate: int
    raw_audio: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "mfcc": self.mfcc.tolist() if isinstance(self.mfcc, np.ndarray) else self.mfcc,
            "pitch": self.pitch.tolist() if isinstance(self.pitch, np.ndarray) else self.pitch,
            "energy": self.energy.tolist() if isinstance(self.energy, np.ndarray) else self.energy,
            "pause_durations": self.pause_durations,
            "speech_rate": self.speech_rate,
            "duration": self.duration,
            "sample_rate": self.sample_rate
        }


class AudioProcessor:
    """
    Audio preprocessing and feature extraction pipeline.
    
    Handles:
    - Audio loading and resampling
    - Amplitude normalization
    - Silence trimming
    - Feature extraction (MFCC, pitch, energy, pauses)
    """
    
    def __init__(
        self,
        target_sr: int = 16000,
        n_mfcc: int = 13,
        hop_length: int = 512,
        frame_length: int = 2048,
        normalize_audio: bool = True,
        trim_silence: bool = True,
        vad_threshold: float = 0.01
    ):
        """
        Initialize audio processor.
        
        Args:
            target_sr: Target sample rate for resampling (Hz)
            n_mfcc: Number of MFCC coefficients to extract
            hop_length: Hop length for frame-based analysis (samples)
            frame_length: Frame length for analysis (samples)
            normalize_audio: Whether to normalize audio amplitude
            trim_silence: Whether to trim leading/trailing silence
            vad_threshold: Voice activity detection threshold (amplitude)
        """
        self.target_sr = target_sr
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.normalize_audio = normalize_audio
        self.trim_silence = trim_silence
        self.vad_threshold = vad_threshold
    
    def load_audio(
        self,
        audio_path: str,
        sr: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file using librosa.
        
        Args:
            audio_path: Path to audio file
            sr: Target sample rate (None uses self.target_sr)
            
        Returns:
            Tuple of (audio_waveform, sample_rate)
        """
        if sr is None:
            sr = self.target_sr
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Load audio with librosa (automatically resamples to target_sr)
            audio, original_sr = librosa.load(
                audio_path,
                sr=sr,
                mono=True  # Convert to mono if stereo
            )
            
            logger.debug(f"Loaded audio: {audio_path}, shape: {audio.shape}, sr: {original_sr} -> {sr}")
            
            return audio, sr
            
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            raise
    
    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio amplitude to [-1, 1] range.
        
        Args:
            audio: Audio waveform
            
        Returns:
            Normalized audio waveform
        """
        if len(audio) == 0:
            return audio
        
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude > 0:
            audio = audio / max_amplitude
        
        return audio
    
    def trim_silence_edges(
        self,
        audio: np.ndarray,
        sr: int,
        top_db: int = 20
    ) -> Tuple[np.ndarray, int, int]:
        """
        Trim leading and trailing silence from audio.
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            top_db: Silence threshold in dB below peak
            
        Returns:
            Tuple of (trimmed_audio, start_sample, end_sample)
        """
        if len(audio) == 0:
            return audio, 0, 0
        
        # Use librosa's trim function
        audio_trimmed, index = librosa.effects.trim(
            audio,
            top_db=top_db,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        
        start_sample, end_sample = index
        
        logger.debug(f"Trimmed silence: {start_sample} -> {end_sample} (removed {start_sample} + {len(audio) - end_sample} samples)")
        
        return audio_trimmed, start_sample, end_sample
    
    def extract_mfcc(
        self,
        audio: np.ndarray,
        sr: int
    ) -> np.ndarray:
        """
        Extract MFCC features.
        
        MFCCs capture spectral characteristics of speech and are commonly
        used for speech recognition and analysis.
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            
        Returns:
            MFCC features (n_mfcc, n_frames)
        """
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.frame_length
        )
        
        return mfccs
    
    def extract_pitch(
        self,
        audio: np.ndarray,
        sr: int,
        fmin: float = 50.0,
        fmax: float = 400.0
    ) -> np.ndarray:
        """
        Extract pitch (fundamental frequency) contour.
        
        Pitch is important for prosody and pronunciation assessment.
        Uses librosa's pyin algorithm for robust pitch tracking.
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            fmin: Minimum frequency for pitch search (Hz)
            fmax: Maximum frequency for pitch search (Hz)
            
        Returns:
            Pitch contour in Hz (n_frames,), with NaN for unvoiced frames
        """
        # Use pyin for pitch tracking (more robust than autocorrelation)
        pitch, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            fill_na=np.nan
        )
        
        return pitch
    
    def extract_energy(
        self,
        audio: np.ndarray,
        sr: int
    ) -> np.ndarray:
        """
        Extract energy contour (RMS energy per frame).
        
        Energy is useful for detecting pauses and speech segments.
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            
        Returns:
            Energy contour (n_frames,)
        """
        # Compute RMS energy per frame
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]  # Remove singleton dimension
        
        return rms
    
    def detect_pauses(
        self,
        energy: np.ndarray,
        sr: int,
        energy_threshold: Optional[float] = None,
        min_pause_duration: float = 0.1
    ) -> list:
        """
        Detect pauses in speech based on energy threshold.
        
        Pauses are important for fluency assessment.
        
        Args:
            energy: Energy contour
            sr: Sample rate
            energy_threshold: Energy threshold for pause detection (None = auto)
            min_pause_duration: Minimum pause duration in seconds
            
        Returns:
            List of pause durations in seconds
        """
        if energy_threshold is None:
            # Auto-threshold: use median energy as threshold
            energy_threshold = np.median(energy) * 0.3
        
        # Convert energy to binary: 1 = speech, 0 = pause
        is_speech = energy > energy_threshold
        
        # Find pause segments
        pause_durations = []
        in_pause = False
        pause_start = 0
        
        frame_duration = self.hop_length / sr  # Duration of each frame in seconds
        
        for i, speech_flag in enumerate(is_speech):
            if not speech_flag and not in_pause:
                # Start of pause
                in_pause = True
                pause_start = i
            elif speech_flag and in_pause:
                # End of pause
                pause_duration = (i - pause_start) * frame_duration
                if pause_duration >= min_pause_duration:
                    pause_durations.append(pause_duration)
                in_pause = False
        
        # Handle pause at the end
        if in_pause:
            pause_duration = (len(is_speech) - pause_start) * frame_duration
            if pause_duration >= min_pause_duration:
                pause_durations.append(pause_duration)
        
        return pause_durations
    
    def compute_speech_rate(
        self,
        audio: np.ndarray,
        transcript: Optional[str] = None,
        duration: Optional[float] = None
    ) -> Optional[float]:
        """
        Compute speech rate (words per second).
        
        Speech rate is a key fluency indicator.
        
        Args:
            audio: Audio waveform
            transcript: Text transcript (optional)
            duration: Audio duration in seconds (optional, computed if None)
            
        Returns:
            Speech rate in words per second, or None if transcript unavailable
        """
        if transcript is None:
            return None
        
        if duration is None:
            duration = len(audio) / self.target_sr
        
        # Simple word count (split by whitespace)
        words = transcript.strip().split()
        word_count = len(words)
        
        if duration > 0:
            speech_rate = word_count / duration
        else:
            speech_rate = 0.0
        
        return speech_rate
    
    def process(
        self,
        audio_path: str,
        transcript: Optional[str] = None,
        return_raw: bool = False
    ) -> AudioFeatures:
        """
        Complete audio processing pipeline.
        
        Args:
            audio_path: Path to audio file
            transcript: Optional text transcript for speech rate computation
            return_raw: Whether to include raw audio in features
            
        Returns:
            AudioFeatures object containing all extracted features
        """
        # Load audio
        audio, sr = self.load_audio(audio_path)
        original_duration = len(audio) / sr
        
        # Normalize amplitude
        if self.normalize_audio:
            audio = self.normalize(audio)
        
        # Trim silence
        if self.trim_silence:
            audio, start_idx, end_idx = self.trim_silence_edges(audio, sr)
        else:
            start_idx, end_idx = 0, len(audio)
        
        duration = len(audio) / sr
        
        # Extract features
        logger.debug(f"Extracting features from {audio_path}...")
        
        mfcc = self.extract_mfcc(audio, sr)
        pitch = self.extract_pitch(audio, sr)
        energy = self.extract_energy(audio, sr)
        pause_durations = self.detect_pauses(energy, sr)
        speech_rate = self.compute_speech_rate(audio, transcript, duration)
        
        # Prepare raw audio (optional)
        raw_audio = audio if return_raw else None
        
        features = AudioFeatures(
            mfcc=mfcc,
            pitch=pitch,
            energy=energy,
            pause_durations=pause_durations,
            speech_rate=speech_rate,
            duration=duration,
            sample_rate=sr,
            raw_audio=raw_audio
        )
        
        logger.debug(f"Extracted features: MFCC shape={mfcc.shape}, pitch shape={pitch.shape}, "
                    f"energy shape={energy.shape}, pauses={len(pause_durations)}")
        
        return features


def main():
    """
    Example usage of the audio processor.
    """
    processor = AudioProcessor(
        target_sr=16000,
        n_mfcc=13,
        normalize_audio=True,
        trim_silence=True
    )
    
    # Example: process an audio file
    # audio_path = "path/to/audio.wav"
    # features = processor.process(audio_path, transcript="Hello world")
    # print(f"Features: {features.to_dict()}")


if __name__ == "__main__":
    main()




