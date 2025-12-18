"""
Main Pipeline Script

This script demonstrates the complete assessment pipeline from audio input
to final scores and explanations.

Author: SHL AI Research Team
"""

import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules
from utils.dataset_loader import L2ArticDatasetLoader, AudioSample
from preprocessing.audio_processor import AudioProcessor
from asr.speech_recognition import ASRFactory
from phoneme_analysis.phoneme_analyzer import PhonemeAnalyzer
from nlp_grammar.grammar_analyzer import GrammarAnalyzer
from scoring.fluency_features import FluencyFeatureExtractor
from scoring.scoring_model import BaselineScoringModel
from scoring.explainability import ExplainerFactory
from fairness.bias_analyzer import BiasAnalyzer
from evaluation.metrics import MetricsEvaluator


def process_single_sample(
    sample: AudioSample,
    audio_processor: AudioProcessor,
    asr,
    phoneme_analyzer: PhonemeAnalyzer,
    grammar_analyzer: GrammarAnalyzer,
    fluency_extractor: FluencyFeatureExtractor,
    scoring_model: BaselineScoringModel,
    explainer
):
    """
    Process a single audio sample through the complete pipeline.
    
    Args:
        sample: AudioSample object
        audio_processor: AudioProcessor instance
        asr: ASR instance
        phoneme_analyzer: PhonemeAnalyzer instance
        grammar_analyzer: GrammarAnalyzer instance
        fluency_extractor: FluencyFeatureExtractor instance
        scoring_model: ScoringModel instance
        explainer: LLMExplainer instance
        
    Returns:
        Dictionary of results
    """
    logger.info(f"Processing sample: {sample.sample_id}")
    
    if not sample.audio_exists:
        logger.warning(f"Audio file not found: {sample.audio_path}")
        return None
    
    try:
        # Step 1: Process audio
        logger.info("Step 1: Processing audio...")
        audio_features = audio_processor.process(sample.audio_path)
        audio_features_dict = audio_features.to_dict()
        
        # Step 2: Transcribe
        logger.info("Step 2: Transcribing speech...")
        asr_result = asr.transcribe(sample.audio_path, audio_duration=audio_features.duration)
        transcript = asr_result.transcript
        asr_result_dict = asr_result.to_dict()
        
        # Step 3: Analyze phonemes
        logger.info("Step 3: Analyzing pronunciation...")
        pronunciation_features = phoneme_analyzer.extract_pronunciation_features(
            sample.canonical_phonemes,
            sample.spoken_phonemes
        )
        
        # Step 4: Analyze grammar
        logger.info("Step 4: Analyzing grammar...")
        grammar_features = grammar_analyzer.extract_grammar_features(transcript)
        
        # Step 5: Extract fluency features
        logger.info("Step 5: Extracting fluency features...")
        fluency_features = fluency_extractor.extract(
            audio_features=audio_features_dict,
            asr_result=asr_result_dict,
            transcript=transcript
        )
        fluency_features_dict = fluency_features.to_dict()
        
        # Step 6: Score
        logger.info("Step 6: Computing scores...")
        score_breakdown = scoring_model.score(
            pronunciation_features,
            grammar_features,
            fluency_features_dict
        )
        score_dict = score_breakdown.to_dict()
        
        # Step 7: Generate explanation
        logger.info("Step 7: Generating feedback...")
        explanation = explainer.explain(
            score_dict,
            pronunciation_features,
            grammar_features,
            fluency_features_dict,
            transcript
        )
        explanation_dict = explanation.to_dict()
        
        return {
            "sample_id": sample.sample_id,
            "speaker_id": sample.speaker_id,
            "nationality": sample.nationality,
            "transcript": transcript,
            "pronunciation_features": pronunciation_features,
            "grammar_features": grammar_features,
            "fluency_features": fluency_features_dict,
            "score_breakdown": score_dict,
            "explanation": explanation_dict
        }
    
    except Exception as e:
        logger.error(f"Error processing sample {sample.sample_id}: {e}")
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Speech-Based Language Assessment Pipeline"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=".",
        help="Root directory containing dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "dev", "test"],
        default="dev",
        help="Dataset split to process"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--asr_backend",
        type=str,
        choices=["whisper", "wav2vec2"],
        default="whisper",
        help="ASR backend to use"
    )
    parser.add_argument(
        "--explainer_backend",
        type=str,
        choices=["openai", "mock"],
        default="mock",
        help="Explainer backend to use"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize components
    logger.info("Initializing components...")
    
    # Dataset loader
    loader = L2ArticDatasetLoader(data_root=args.data_root)
    samples = loader.load_split(args.split)
    
    logger.info(f"Loaded {len(samples)} samples from {args.split} split")
    
    # Limit samples
    samples = samples[:args.max_samples]
    
    # Processors
    audio_processor = AudioProcessor()
    
    try:
        asr = ASRFactory.create(args.asr_backend, model_size="base")
    except Exception as e:
        logger.warning(f"Could not initialize {args.asr_backend}: {e}")
        logger.warning("Falling back to mock ASR")
        asr = None
    
    phoneme_analyzer = PhonemeAnalyzer()
    grammar_analyzer = GrammarAnalyzer(use_spacy=False)  # Set to True if spaCy model is available
    fluency_extractor = FluencyFeatureExtractor()
    scoring_model = BaselineScoringModel()
    explainer = ExplainerFactory.create(args.explainer_backend)
    
    # Process samples
    logger.info(f"Processing {len(samples)} samples...")
    results = []
    scores = []
    nationalities = []
    
    for i, sample in enumerate(samples):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing sample {i+1}/{len(samples)}")
        logger.info(f"{'='*60}")
        
        result = process_single_sample(
            sample,
            audio_processor,
            asr,
            phoneme_analyzer,
            grammar_analyzer,
            fluency_extractor,
            scoring_model,
            explainer
        )
        
        if result:
            results.append(result)
            scores.append(result["score_breakdown"]["overall_score"])
            nationalities.append(result["nationality"])
            
            # Print results
            logger.info(f"\nResults for {result['sample_id']}:")
            logger.info(f"  Speaker: {result['speaker_id']} ({result['nationality']})")
            logger.info(f"  Overall Score: {result['score_breakdown']['overall_score']:.1f}/100")
            logger.info(f"    - Pronunciation: {result['score_breakdown']['pronunciation_score']:.1f}/100")
            logger.info(f"    - Grammar: {result['score_breakdown']['grammar_score']:.1f}/100")
            logger.info(f"    - Fluency: {result['score_breakdown']['fluency_score']:.1f}/100")
            logger.info(f"  Transcript: {result['transcript'][:100]}...")
    
    # Fairness analysis
    if len(scores) > 0 and len(set(nationalities)) > 1:
        logger.info("\n" + "="*60)
        logger.info("FAIRNESS ANALYSIS")
        logger.info("="*60)
        
        bias_analyzer = BiasAnalyzer()
        fairness_report = bias_analyzer.analyze(scores, nationalities)
        bias_analyzer.print_report(fairness_report)
    
    # Evaluation metrics
    if len(scores) > 0:
        logger.info("\n" + "="*60)
        logger.info("EVALUATION METRICS")
        logger.info("="*60)
        
        evaluator = MetricsEvaluator()
        
        # Component scores
        component_scores = {
            "pronunciation": [r["score_breakdown"]["pronunciation_score"] for r in results],
            "grammar": [r["score_breakdown"]["grammar_score"] for r in results],
            "fluency": [r["score_breakdown"]["fluency_score"] for r in results]
        }
        
        metrics = evaluator.evaluate(scores, component_scores=component_scores)
        evaluator.print_report(metrics)
    
    logger.info("\n" + "="*60)
    logger.info("Pipeline completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    main()




