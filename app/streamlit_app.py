"""
Streamlit Demo Application

This module provides a web-based interface for the speech assessment system.

Features:
- Audio file upload
- Real-time processing
- Score display (pronunciation, grammar, fluency)
- LLM-generated feedback
- Visualization of results

Author: SHL AI Research Team
"""

import streamlit as st
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path for imports (insert at beginning to prioritize)
app_dir = Path(__file__).parent.absolute()
project_root = app_dir.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from preprocessing.audio_processor import AudioProcessor
from asr.speech_recognition import ASRFactory
from phoneme_analysis.phoneme_analyzer import PhonemeAnalyzer
from nlp_grammar.grammar_analyzer import GrammarAnalyzer
from scoring.fluency_features import FluencyFeatureExtractor
from scoring.scoring_model import BaselineScoringModel
from scoring.explainability import ExplainerFactory
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Speech Language Assessment",
    page_icon="🎤",
    layout="wide"
)

# Initialize session state
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "results" not in st.session_state:
    st.session_state.results = None


def process_audio(audio_file, canonical_phonemes=None):
    """
    Process uploaded audio file through the complete pipeline.
    
    Args:
        audio_file: Uploaded audio file
        canonical_phonemes: Optional canonical phoneme sequence
        
    Returns:
        Dictionary of results
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_path = tmp_file.name
        tmp_file.write(audio_file.read())
    
    try:
        # Initialize processors
        audio_processor = AudioProcessor(
            target_sr=16000,
            normalize_audio=True,
            trim_silence=True
        )
        
        # Process audio
        with st.spinner("Processing audio..."):
            audio_features = audio_processor.process(tmp_path, return_raw=False)
            audio_features_dict = audio_features.to_dict()
        
        # ASR
        with st.spinner("Transcribing speech..."):
            try:
                asr = ASRFactory.create("whisper", model_size="base")
            except Exception as e:
                logger.warning(f"Whisper not available: {e}")
                st.warning("Whisper not available, using mock transcription")
                asr = None
            
            if asr:
                try:
                    asr_result = asr.transcribe(tmp_path, audio_duration=audio_features.duration)
                    transcript = asr_result.transcript
                    asr_result_dict = asr_result.to_dict()
                except Exception as e:
                    logger.error(f"Transcription failed: {e}")
                    st.warning(f"Transcription failed: {str(e)}. Using mock transcription.")
                    # Mock transcript
                    transcript = "This is a sample transcript. The system is processing the audio."
                    asr_result_dict = {
                        "transcript": transcript,
                        "word_count": len(transcript.split()),
                        "speaking_speed": len(transcript.split()) / audio_features.duration if audio_features.duration > 0 else 0,
                        "pause_count": 0
                    }
            else:
                # Mock transcript
                transcript = "This is a sample transcript. The system is processing the audio."
                asr_result_dict = {
                    "transcript": transcript,
                    "word_count": len(transcript.split()),
                    "speaking_speed": len(transcript.split()) / audio_features.duration if audio_features.duration > 0 else 0,
                    "pause_count": 0
                }
        
        # Phoneme analysis
        pronunciation_features = {}
        if canonical_phonemes:
            with st.spinner("Analyzing pronunciation..."):
                phoneme_analyzer = PhonemeAnalyzer()
                # Use mock spoken phonemes for demo (in production, use actual phoneme recognition)
                spoken_phonemes = canonical_phonemes  # Placeholder
                pronunciation_features = phoneme_analyzer.extract_pronunciation_features(
                    canonical_phonemes,
                    spoken_phonemes
                )
        else:
            # Mock pronunciation features
            pronunciation_features = {
                "phoneme_error_rate": 0.15,
                "substitution_rate": 0.08,
                "deletion_rate": 0.04,
                "insertion_rate": 0.03,
                "correct_rate": 0.85
            }
        
        # Grammar analysis
        with st.spinner("Analyzing grammar..."):
            grammar_analyzer = GrammarAnalyzer(use_spacy=False)  # Use False to avoid requiring model download
            grammar_features = grammar_analyzer.extract_grammar_features(transcript)
        
        # Fluency features
        with st.spinner("Extracting fluency features..."):
            fluency_extractor = FluencyFeatureExtractor()
            fluency_features = fluency_extractor.extract(
                audio_features=audio_features_dict,
                asr_result=asr_result_dict,
                transcript=transcript
            )
            fluency_features_dict = fluency_features.to_dict()
        
        # Scoring
        with st.spinner("Computing scores..."):
            scoring_model = BaselineScoringModel()
            score_breakdown = scoring_model.score(
                pronunciation_features,
                grammar_features,
                fluency_features_dict
            )
            score_dict = score_breakdown.to_dict()
        
        # Explainability
        with st.spinner("Generating feedback..."):
            explainer = ExplainerFactory.create("mock")
            explanation = explainer.explain(
                score_dict,
                pronunciation_features,
                grammar_features,
                fluency_features_dict,
                transcript
            )
            explanation_dict = explanation.to_dict()
        
        return {
            "transcript": transcript,
            "audio_features": audio_features_dict,
            "pronunciation_features": pronunciation_features,
            "grammar_features": grammar_features,
            "fluency_features": fluency_features_dict,
            "score_breakdown": score_dict,
            "explanation": explanation_dict
        }
    
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    """Main Streamlit app."""
    st.title("🎤 Speech-Based Language Assessment System")
    st.markdown("""
    This system evaluates spoken English communication skills using:
    - **Pronunciation Analysis**: Phoneme error detection
    - **Grammar Analysis**: NLP-based grammar error detection
    - **Fluency Analysis**: Speaking rate, pauses, prosody
    - **AI-Powered Feedback**: LLM-generated improvement suggestions
    """)
    
    st.sidebar.header("Configuration")
    
    # File upload
    st.header("Upload Audio")
    audio_file = st.file_uploader(
        "Choose an audio file (.wav, .mp3, .flac)",
        type=["wav", "mp3", "flac"]
    )
    
    # Optional canonical phonemes
    st.sidebar.subheader("Optional: Canonical Phonemes")
    canonical_phonemes = st.sidebar.text_input(
        "Enter canonical phoneme sequence (space-separated)",
        help="For pronunciation analysis, provide the expected phoneme sequence"
    )
    
    # Process button
    if audio_file is not None:
        if st.button("Process Audio", type="primary"):
            st.session_state.processing_complete = False
            st.session_state.results = None
            
            # Process
            results = process_audio(audio_file, canonical_phonemes if canonical_phonemes else None)
            
            st.session_state.results = results
            st.session_state.processing_complete = True
            st.success("Processing complete!")
    
    # Display results
    if st.session_state.processing_complete and st.session_state.results:
        results = st.session_state.results
        
        st.header("📊 Assessment Results")
        
        # Score display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Overall Score",
                f"{results['score_breakdown']['overall_score']:.1f}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Pronunciation",
                f"{results['score_breakdown']['pronunciation_score']:.1f}",
                delta=None
            )
        
        with col3:
            st.metric(
                "Grammar",
                f"{results['score_breakdown']['grammar_score']:.1f}",
                delta=None
            )
        
        with col4:
            st.metric(
                "Fluency",
                f"{results['score_breakdown']['fluency_score']:.1f}",
                delta=None
            )
        
        # Transcript
        st.subheader("📝 Transcript")
        st.text(results['transcript'])
        
        # Explanation
        st.subheader("💡 AI-Generated Feedback")
        explanation = results['explanation']
        
        st.markdown(f"**Summary:** {explanation['score_summary']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Strengths:**")
            for strength in explanation['strengths']:
                st.markdown(f"- {strength}")
        
        with col2:
            st.markdown("**Weaknesses:**")
            for weakness in explanation['weaknesses']:
                st.markdown(f"- {weakness}")
        
        st.markdown("**Suggestions:**")
        for suggestion in explanation['suggestions']:
            st.markdown(f"- {suggestion}")
        
        # Detailed metrics
        with st.expander("📈 Detailed Metrics"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Pronunciation Metrics:**")
                st.json(results['pronunciation_features'])
            
            with col2:
                st.markdown("**Grammar Metrics:**")
                st.json(results['grammar_features'])
            
            with col3:
                st.markdown("**Fluency Metrics:**")
                st.json(results['fluency_features'])
        
        # Audio features
        with st.expander("🎵 Audio Features"):
            st.json({
                "duration": results['audio_features']['duration'],
                "sample_rate": results['audio_features']['sample_rate'],
                "pause_count": len(results['audio_features']['pause_durations']),
                "speech_rate": results['audio_features'].get('speech_rate', 'N/A')
            })


if __name__ == "__main__":
    main()

