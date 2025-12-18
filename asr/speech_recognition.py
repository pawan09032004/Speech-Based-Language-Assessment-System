"""
Automatic Speech Recognition (ASR) Module

This module provides ASR functionality using multiple backends:
- Whisper (OpenAI)
- Wav2Vec 2.0 (Facebook)

The module is designed to be modular, allowing easy swapping of ASR models.

Author: SHL AI Research Team
"""

import os
import numpy as np
import torch
import torchaudio
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class ASRResult:
    """
    Container for ASR transcription results.
    
    Attributes:
        transcript: Transcribed text
        word_count: Number of words in transcript
        speaking_speed: Words per second
        pause_count: Number of detected pauses (from timing if available)
        confidence: Confidence score (if available from model)
        word_timings: List of (word, start_time, end_time) tuples (if available)
    """
    transcript: str
    word_count: int
    speaking_speed: float
    pause_count: int
    confidence: Optional[float] = None
    word_timings: Optional[List[Tuple[str, float, float]]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "transcript": self.transcript,
            "word_count": self.word_count,
            "speaking_speed": self.speaking_speed,
            "pause_count": self.pause_count,
            "confidence": self.confidence
        }


class BaseASR:
    """
    Base class for ASR models.
    
    All ASR implementations should inherit from this class.
    """
    
    def transcribe(
        self,
        audio_path: str,
        audio_duration: Optional[float] = None
    ) -> ASRResult:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            audio_duration: Duration of audio in seconds (optional, computed if None)
            
        Returns:
            ASRResult object
        """
        raise NotImplementedError("Subclasses must implement transcribe()")
    
    def transcribe_from_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        audio_duration: Optional[float] = None
    ) -> ASRResult:
        """
        Transcribe audio from numpy array.
        
        Args:
            audio_array: Audio waveform as numpy array
            sample_rate: Sample rate of audio
            audio_duration: Duration of audio in seconds (optional)
            
        Returns:
            ASRResult object
        """
        raise NotImplementedError("Subclasses must implement transcribe_from_array()")


class WhisperASR(BaseASR):
    """
    Whisper ASR model wrapper.
    
    Whisper is a state-of-the-art ASR model from OpenAI that provides
    high-quality transcriptions with word-level timings.
    """
    
    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        language: Optional[str] = "en"
    ):
        """
        Initialize Whisper ASR.
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            device: Device to run on ("cpu", "cuda", or None for auto)
            language: Language code (e.g., "en" for English)
        """
        try:
            import whisper
            self.whisper = whisper
        except ImportError:
            raise ImportError(
                "Whisper not installed. Install with: pip install openai-whisper"
            )
        
        self.model_size = model_size
        self.language = language
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load model
        logger.info(f"Loading Whisper model: {model_size} on {device}")
        self.model = self.whisper.load_model(model_size, device=device)
        logger.info("Whisper model loaded successfully")
    
    def _check_ffmpeg_available(self) -> bool:
        """Check if ffmpeg is available in the system."""
        import shutil
        return shutil.which("ffmpeg") is not None
    
    def transcribe(
        self,
        audio_path: str,
        audio_duration: Optional[float] = None
    ) -> ASRResult:
        """Transcribe audio file using Whisper."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Try to transcribe with file path first (uses ffmpeg)
        # If ffmpeg is not available, fall back to loading with librosa and saving to temp file
        try:
            # Transcribe (word timings are automatically included in segments by default)
            # Avoid passing word_timestamps parameter to prevent API compatibility issues
            try:
                result = self.model.transcribe(
                    audio_path,
                    language=self.language,
                    verbose=False
                )
            except (TypeError, ValueError) as e:
                # Catch API compatibility errors (e.g., word_timings parameter issues)
                if "word_timings" in str(e) or "DecodingOptions" in str(e):
                    logger.warning(f"Whisper API compatibility issue detected: {e}")
                    logger.warning("Attempting transcription with minimal parameters...")
                    # Try with even more minimal parameters
                    result = self.model.transcribe(audio_path, verbose=False)
                else:
                    raise
        except (FileNotFoundError, OSError) as e:
            # ffmpeg not found or other OS error, load audio with librosa
            logger.warning(f"ffmpeg not available ({e}), using librosa as fallback")
            try:
                import librosa
                
                # Load audio with librosa (doesn't require ffmpeg)
                audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
                
                # Pass numpy array directly to Whisper
                # Word timings are automatically included in segments by default
                try:
                    result = self.model.transcribe(
                        audio_array,
                        language=self.language,
                        verbose=False
                    )
                except (TypeError, ValueError) as e:
                    # Catch API compatibility errors (e.g., word_timings parameter issues)
                    if "word_timings" in str(e) or "DecodingOptions" in str(e):
                        logger.warning(f"Whisper API compatibility issue detected: {e}")
                        logger.warning("Attempting transcription with minimal parameters...")
                        # Try with even more minimal parameters
                        result = self.model.transcribe(audio_array, verbose=False)
                    else:
                        raise
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load audio. ffmpeg not found and librosa fallback failed: {e2}. "
                    "Please ensure ffmpeg is in PATH or librosa can load the audio file."
                ) from e2
        
        transcript = result["text"].strip()
        segments = result.get("segments", [])
        
        # Extract word timings
        word_timings = []
        for segment in segments:
            words = segment.get("words", [])
            for word_info in words:
                word = word_info.get("word", "").strip()
                start = word_info.get("start", 0.0)
                end = word_info.get("end", 0.0)
                if word:
                    word_timings.append((word, start, end))
        
        # Compute statistics
        word_count = len(transcript.split())
        
        if audio_duration is None:
            # Estimate from segments
            if segments:
                audio_duration = segments[-1].get("end", 0.0)
            else:
                audio_duration = 1.0  # Fallback
        
        speaking_speed = word_count / audio_duration if audio_duration > 0 else 0.0
        
        # Count pauses (gaps > 0.5 seconds between words)
        pause_count = 0
        if len(word_timings) > 1:
            for i in range(1, len(word_timings)):
                gap = word_timings[i][1] - word_timings[i-1][2]
                if gap > 0.5:
                    pause_count += 1
        
        # Get confidence (average of segment confidences)
        confidence = None
        if segments:
            confidences = [s.get("no_speech_prob", 0.0) for s in segments]
            if confidences:
                # Convert no_speech_prob to confidence (lower is better)
                confidence = 1.0 - np.mean(confidences)
        
        return ASRResult(
            transcript=transcript,
            word_count=word_count,
            speaking_speed=speaking_speed,
            pause_count=pause_count,
            confidence=confidence,
            word_timings=word_timings if word_timings else None
        )
    
    def transcribe_from_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        audio_duration: Optional[float] = None
    ) -> ASRResult:
        """Transcribe audio array using Whisper."""
        # Save to temporary file and transcribe
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            try:
                # Save audio
                import soundfile as sf
                sf.write(tmp_path, audio_array, sample_rate)
                
                # Transcribe
                result = self.transcribe(tmp_path, audio_duration)
                return result
            finally:
                # Clean up
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)


class Wav2Vec2ASR(BaseASR):
    """
    Wav2Vec 2.0 ASR model wrapper.
    
    Wav2Vec 2.0 is a self-supervised learning model from Facebook that
    can be fine-tuned for ASR tasks.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        device: Optional[str] = None
    ):
        """
        Initialize Wav2Vec 2.0 ASR.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ("cpu", "cuda", or None for auto)
        """
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        except ImportError:
            raise ImportError(
                "Transformers not installed. Install with: pip install transformers"
            )
        
        self.model_name = model_name
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load model and processor
        logger.info(f"Loading Wav2Vec2 model: {model_name} on {device}")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
        self.model.eval()
        logger.info("Wav2Vec2 model loaded successfully")
    
    def transcribe(
        self,
        audio_path: str,
        audio_duration: Optional[float] = None
    ) -> ASRResult:
        """Transcribe audio file using Wav2Vec2."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)
        
        return self.transcribe_from_array(audio, sr, audio_duration)
    
    def transcribe_from_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        audio_duration: Optional[float] = None
    ) -> ASRResult:
        """Transcribe audio array using Wav2Vec2."""
        # Resample if needed
        if sample_rate != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Process audio
        inputs = self.processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Transcribe
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcript = self.processor.decode(predicted_ids[0])
        
        # Clean transcript
        transcript = transcript.strip()
        
        # Compute statistics
        word_count = len(transcript.split())
        
        if audio_duration is None:
            audio_duration = len(audio_array) / sample_rate
        
        speaking_speed = word_count / audio_duration if audio_duration > 0 else 0.0
        
        # Wav2Vec2 doesn't provide word timings or confidence scores
        # Estimate pause count from transcript (count punctuation as pauses)
        pause_count = len(re.findall(r'[.,!?;:]', transcript))
        
        return ASRResult(
            transcript=transcript,
            word_count=word_count,
            speaking_speed=speaking_speed,
            pause_count=pause_count,
            confidence=None,
            word_timings=None
        )


class ASRFactory:
    """
    Factory class for creating ASR instances.
    """
    
    @staticmethod
    def create(
        backend: str = "whisper",
        **kwargs
    ) -> BaseASR:
        """
        Create an ASR instance.
        
        Args:
            backend: ASR backend ("whisper" or "wav2vec2")
            **kwargs: Additional arguments passed to ASR constructor
            
        Returns:
            ASR instance
        """
        if backend.lower() == "whisper":
            return WhisperASR(**kwargs)
        elif backend.lower() == "wav2vec2":
            return Wav2Vec2ASR(**kwargs)
        else:
            raise ValueError(f"Unknown ASR backend: {backend}. Choose 'whisper' or 'wav2vec2'.")


def main():
    """
    Example usage of ASR modules.
    """
    # Example: Use Whisper
    # asr = ASRFactory.create("whisper", model_size="base")
    # result = asr.transcribe("path/to/audio.wav")
    # print(f"Transcript: {result.transcript}")
    # print(f"Word count: {result.word_count}")
    # print(f"Speaking speed: {result.speaking_speed:.2f} words/sec")
    
    # Example: Use Wav2Vec2
    # asr = ASRFactory.create("wav2vec2", model_name="facebook/wav2vec2-base-960h")
    # result = asr.transcribe("path/to/audio.wav")
    # print(f"Transcript: {result.transcript}")


if __name__ == "__main__":
    main()

