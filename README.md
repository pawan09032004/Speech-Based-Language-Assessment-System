# Speech-Based Grammar, Fluency, and Pronunciation Scoring Engine for Spoken Language Assessment

Deployed app link: https://pawan09032004-speech-based-language-ass-appstreamlit-app-pfjxtp.streamlit.app/

A comprehensive research-grade system for evaluating spoken English communication skills using advanced speech processing, NLP, and deep learning techniques. This project provides multi-dimensional assessment of pronunciation, grammar, and fluency with explainable AI, fairness analysis, and an interactive web interface.

## 🎯 Problem Motivation

### People Science & Hiring Context

In modern hiring and assessment contexts, evaluating spoken language proficiency is critical for:

1. **Global Talent Acquisition**: Assessing English communication skills for non-native speakers in international roles
2. **Fair Assessment**: Providing objective, standardized evaluation that reduces human bias
3. **Scalability**: Enabling automated assessment at scale for large applicant pools
4. **Consistency**: Ensuring uniform evaluation criteria across all candidates
5. **Actionable Feedback**: Delivering specific, constructive feedback to help candidates improve

Traditional human evaluation is:
- **Time-consuming**: Requires expert linguists for each assessment
- **Subjective**: Prone to inter-rater variability and implicit bias
- **Expensive**: Not scalable for high-volume hiring
- **Inconsistent**: Different evaluators may apply different standards

This system addresses these challenges by providing:
- **Automated Assessment**: Fast, consistent evaluation
- **Multi-dimensional Analysis**: Pronunciation, grammar, and fluency
- **Explainable AI**: Transparent scoring with actionable feedback
- **Fairness Analysis**: Bias detection and mitigation strategies

## ✨ Key Features

### Core Functionalities

1. **Audio Processing & Feature Extraction**
   - Audio resampling and normalization
   - Silence trimming and VAD (Voice Activity Detection)
   - MFCC (Mel-frequency cepstral coefficients) extraction
   - Pitch and prosody analysis
   - Energy and pause detection
   - Speech rate calculation

2. **Speech Recognition (ASR)**
   - **Whisper Integration**: OpenAI's state-of-the-art speech recognition
   - **Wav2Vec2 Support**: Alternative ASR backend
   - Automatic transcription with word-level timestamps
   - Multiple model sizes (tiny, base, small, medium, large)
   - Fallback to mock transcription for testing

3. **Pronunciation Analysis**
   - Phoneme Error Rate (PER) computation
   - Substitution, deletion, and insertion detection
   - Canonical vs. spoken phoneme alignment
   - Pronunciation accuracy scoring (0-100 scale)
   - Support for ARPA phoneme set (40 phonemes, expandable to 240)

4. **Grammar Analysis**
   - NLP-based grammar error detection
   - Sentence complexity analysis
   - Part-of-speech (POS) tagging and dependency parsing
   - Grammar error density calculation
   - Sentence completeness evaluation
   - spaCy integration (optional)

5. **Fluency Assessment**
   - Speaking rate analysis (words per second)
   - Hesitation marker detection ("um", "uh", "er")
   - Pause ratio and duration analysis
   - Sentence completeness scoring
   - Prosodic variability measurement
   - Combined audio-text fluency features

6. **Scoring Models**
   - **Baseline Model**: Interpretable linear combination with configurable weights
     - Pronunciation weight: 0.4
     - Grammar weight: 0.3
     - Fluency weight: 0.3
   - **MLP Model**: Deep learning model with non-linear feature learning
     - Multi-layer perceptron architecture
     - Configurable hidden layers (default: [64, 32])
     - Dropout regularization (0.2)
     - Output: 4 scores (overall, pronunciation, grammar, fluency)
   - Score interpretation: 0-40 (Beginner), 40-60 (Intermediate), 60-80 (Advanced), 80-100 (Expert/Native-like)

7. **Explainability & Feedback**
   - **LLM-Powered Explanations**: OpenAI GPT or Anthropic Claude integration
   - **Mock Explainer**: For testing without API keys
   - Score summary generation
   - Strength identification
   - Weakness analysis
   - Actionable improvement suggestions
   - Detailed feedback text

8. **Fairness & Bias Analysis**
   - Nationality-based score statistics
   - Mean score difference analysis
   - Statistical significance testing (t-tests)
   - Effect size calculation (Cohen's d)
   - Bias indicator identification
   - Mitigation strategy suggestions
   - Data imbalance detection

9. **Evaluation Metrics**
   - Score distribution statistics (mean, std, median, IQR)
   - Correlation analysis (Pearson, Spearman)
   - Error distribution analysis (MAE, RMSE)
   - Component correlation analysis
   - Limitation identification
   - Comprehensive evaluation reports

10. **Interactive Web Application (Streamlit)**
    - Audio file upload (.wav, .mp3, .flac)
    - Real-time processing with progress indicators
    - Score visualization (overall, pronunciation, grammar, fluency)
    - Transcript display
    - AI-generated feedback display
    - Detailed metrics viewer
    - Audio features visualization
    - Optional canonical phoneme input

11. **Command-Line Interface**
    - Batch processing of dataset splits
    - Configurable ASR backend selection
    - Explainer backend selection
    - Sample limit configuration
    - Output directory management
    - Comprehensive logging

12. **Dataset Management**
    - L2-ARTIC dataset loader
    - CSV parsing and validation
    - Audio file existence checking
    - Metadata extraction (speaker ID, nationality)
    - Statistics generation
    - Support for train/dev/test splits

## 📊 Dataset Description

### L2-ARTIC en_mdd Dataset

The system uses the **L2-ARTIC** (L2 Arctic) dataset with nationality categorization:

- **Audio Files**: `.wav` format, recorded speech samples
- **Canonical Phoneme Sequences**: Native-like phoneme transcriptions (reference)
- **Spoken Phoneme Sequences**: Actual phoneme transcriptions (from speakers)
- **Speaker Metadata**: Nationality information (Arabic, Chinese, Indian, Korean, Spanish, Vietnamese)

### Dataset Statistics

- **Training Set**: ~2,700 samples
- **Development Set**: ~449 samples
- **Test Set**: ~450 samples
- **Speakers**: Multiple speakers per nationality group
- **Phoneme Inventory**: 40 standard ARPA phonemes, expandable to 240 with nationality tags

### Data Format

CSV files contain:
- `Path`: Audio file path
- `Canonical`: Reference phoneme sequence (space-separated)
- `Transcript`: Spoken phoneme sequence (space-separated)

## 🔬 Methodology

### Pipeline Overview

The assessment pipeline consists of 11 sequential steps:

1. **Dataset Loading**: CSV parsing, audio validation, metadata extraction
2. **Audio Preprocessing**: Resampling, normalization, silence trimming, feature extraction
3. **Speech-to-Text (ASR)**: Transcription using Whisper or Wav2Vec2
4. **Phoneme Error Analysis**: PER computation, substitution/deletion/insertion detection
5. **NLP Grammar Analysis**: Grammar error detection, complexity analysis
6. **Fluency Feature Engineering**: Combining audio and text features
7. **Scoring Model**: Baseline regression + deep learning (MLP)
8. **Explainability**: LLM-generated feedback and suggestions
9. **Fairness Analysis**: Bias detection across nationality groups
10. **Evaluation**: Metrics and correlation analysis
11. **Demo Application**: Streamlit web interface

### Feature Extraction

#### Audio Features
- **MFCCs**: Mel-frequency cepstral coefficients (spectral characteristics)
- **Pitch**: Fundamental frequency contour (prosody)
- **Energy**: RMS energy per frame (pause detection)
- **Pause Durations**: Detected silence segments
- **Speech Rate**: Words per second

#### Text Features
- **Grammar Errors**: Detected grammar mistakes
- **Sentence Complexity**: POS diversity, dependency depth
- **Hesitation Markers**: "um", "uh", "er" counts
- **Sentence Completeness**: Ratio of complete sentences

#### Phoneme Features
- **Phoneme Error Rate (PER)**: Overall pronunciation accuracy
- **Substitution Rate**: Phoneme substitutions
- **Deletion Rate**: Missing phonemes
- **Insertion Rate**: Extra phonemes

## 🏗️ Model Design

### Scoring Architecture

#### Baseline Model (Rule-Based)

Linear combination of component scores:

```
Overall Score = w₁ × Pronunciation + w₂ × Grammar + w₃ × Fluency
```

Where:
- `w₁ = 0.4` (pronunciation weight)
- `w₂ = 0.3` (grammar weight)
- `w₃ = 0.3` (fluency weight)

**Component Scoring Logic**:

1. **Pronunciation** (0-100):
   - Based on Phoneme Error Rate (PER)
   - Penalties for substitutions, deletions, insertions
   - Formula: `(1 - PER) × 100 - error_penalties`

2. **Grammar** (0-100):
   - Based on grammar error density
   - Rewards sentence complexity (with moderation)
   - Formula: `100 - (error_density × 10) + complexity_bonus`

3. **Fluency** (0-100):
   - Speaking rate (optimal: 2-3 words/sec)
   - Hesitation and pause penalties
   - Sentence completeness bonus
   - Prosodic variability bonus

#### Deep Learning Model (MLP)

Multi-layer perceptron with:
- **Input**: 20-dimensional feature vector (pronunciation + grammar + fluency features)
- **Hidden Layers**: [64, 32] neurons with ReLU activation
- **Dropout**: 0.2 for regularization
- **Output**: 4 scores (overall, pronunciation, grammar, fluency)
- **Activation**: Sigmoid (scaled to 0-100)
- **Training**: Adam optimizer, MSE loss, validation split support

### Score Interpretation

- **0-40**: Beginner level
- **40-60**: Intermediate level
- **60-80**: Advanced level
- **80-100**: Expert/Native-like level

## ⚖️ Fairness Considerations

### Bias Detection

The system includes comprehensive fairness analysis:

1. **Mean Score Differences**: Pairwise comparisons across nationality groups
2. **Statistical Significance**: T-tests to identify significant differences
3. **Effect Size**: Cohen's d for practical significance
4. **Accent Bias Risks**: Identification of groups with systematically lower scores
5. **Data Imbalance Detection**: Automatic detection of imbalanced datasets

### Mitigation Strategies

1. **Data Balancing**: Ensure balanced representation across nationality groups
2. **Calibration**: Nationality-aware scoring thresholds if needed
3. **Human Validation**: Expert evaluation to validate score differences
4. **Adversarial Training**: Fairness constraints in model training
5. **Transparency**: Clear documentation of scoring criteria

### Ethical Considerations

- **Accent Neutrality**: System should evaluate proficiency, not accent
- **Cultural Sensitivity**: Avoid penalizing non-standard but valid expressions
- **Transparency**: Clear explanation of scoring logic
- **Continuous Monitoring**: Regular fairness audits
- **Human Oversight**: Expert review for high-stakes decisions

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster processing)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd SHL_Assignment
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download spaCy model** (for grammar analysis):
```bash
python -m spacy download en_core_web_sm
```

4. **Download Whisper model** (automatic on first use):
   - Models are downloaded automatically when first used
   - Available sizes: tiny, base, small, medium, large

### Optional: OpenAI API Key

For LLM explainability, set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

## 💻 Usage

### Command Line

#### Dataset Loading and Exploration

```python
from utils.dataset_loader import L2ArticDatasetLoader

loader = L2ArticDatasetLoader(
    data_root=".",
    csv_train_path="graph_dataset_l2_artic/graph_dataset_l2_artic/train.csv",
    csv_dev_path="graph_dataset_l2_artic/graph_dataset_l2_artic/dev.csv",
    csv_test_path="graph_dataset_l2_artic/graph_dataset_l2_artic/test.csv"
)

samples = loader.load_all()
loader.print_statistics()
```

#### Complete Assessment Pipeline

```python
from preprocessing.audio_processor import AudioProcessor
from asr.speech_recognition import ASRFactory
from phoneme_analysis.phoneme_analyzer import PhonemeAnalyzer
from nlp_grammar.grammar_analyzer import GrammarAnalyzer
from scoring.fluency_features import FluencyFeatureExtractor
from scoring.scoring_model import BaselineScoringModel
from scoring.explainability import ExplainerFactory

# 1. Process audio
audio_processor = AudioProcessor()
audio_features = audio_processor.process("path/to/audio.wav")

# 2. Transcribe
asr = ASRFactory.create("whisper", model_size="base")
asr_result = asr.transcribe("path/to/audio.wav")

# 3. Analyze phonemes
phoneme_analyzer = PhonemeAnalyzer()
pronunciation_features = phoneme_analyzer.extract_pronunciation_features(
    canonical_phonemes, spoken_phonemes
)

# 4. Analyze grammar
grammar_analyzer = GrammarAnalyzer()
grammar_features = grammar_analyzer.extract_grammar_features(asr_result.transcript)

# 5. Extract fluency features
fluency_extractor = FluencyFeatureExtractor()
fluency_features = fluency_extractor.extract(
    audio_features=audio_features.to_dict(),
    asr_result=asr_result.to_dict()
)

# 6. Score
scoring_model = BaselineScoringModel()
score_breakdown = scoring_model.score(
    pronunciation_features,
    grammar_features,
    fluency_features.to_dict()
)

# 7. Generate explanation
explainer = ExplainerFactory.create("mock")  # or "openai" with API key
explanation = explainer.explain(
    score_breakdown.to_dict(),
    pronunciation_features,
    grammar_features,
    fluency_features.to_dict(),
    asr_result.transcript
)
```

#### Running Main Pipeline

```bash
python main.py --split dev --max_samples 10 --asr_backend whisper --explainer_backend mock
```

**Command-line Arguments**:
- `--data_root`: Root directory containing dataset (default: ".")
- `--split`: Dataset split to process (train/dev/test, default: "dev")
- `--max_samples`: Maximum number of samples to process (default: 10)
- `--asr_backend`: ASR backend to use (whisper/wav2vec2, default: "whisper")
- `--explainer_backend`: Explainer backend (openai/mock, default: "mock")
- `--output_dir`: Output directory for results (default: "results")

#### Fairness Analysis

```python
from fairness.bias_analyzer import BiasAnalyzer

analyzer = BiasAnalyzer()
report = analyzer.analyze(scores, nationalities)
analyzer.print_report(report)
```

#### Evaluation Metrics

```python
from evaluation.metrics import MetricsEvaluator

evaluator = MetricsEvaluator()
metrics = evaluator.evaluate(scores, component_scores=component_scores)
evaluator.print_report(metrics)
```

### Streamlit Web Application

Launch the interactive web interface:

```bash
streamlit run app/streamlit_app.py
```

**Features**:
- Audio file upload (.wav, .mp3, .flac)
- Real-time processing with progress indicators
- Score visualization (overall, pronunciation, grammar, fluency)
- Transcript display
- AI-generated feedback (summary, strengths, weaknesses, suggestions)
- Detailed metrics viewer (expandable sections)
- Audio features display
- Optional canonical phoneme input for pronunciation analysis

**Usage**:
1. Upload an audio file using the file uploader
2. (Optional) Enter canonical phoneme sequence in the sidebar
3. Click "Process Audio" button
4. View results: scores, transcript, feedback, and detailed metrics

## 📁 Project Structure

```
SHL_Assignment/
├── app/
│   ├── __init__.py
│   └── streamlit_app.py           # Streamlit web application
├── asr/
│   ├── __init__.py
│   └── speech_recognition.py      # ASR (Whisper/Wav2Vec2)
├── phoneme_analysis/
│   ├── __init__.py
│   └── phoneme_analyzer.py        # Phoneme error analysis
├── nlp_grammar/
│   ├── __init__.py
│   └── grammar_analyzer.py        # Grammar analysis
├── scoring/
│   ├── __init__.py
│   ├── fluency_features.py        # Fluency feature extraction
│   ├── scoring_model.py           # Baseline + MLP models
│   └── explainability.py          # LLM explainability
├── fairness/
│   ├── __init__.py
│   └── bias_analyzer.py           # Fairness and bias analysis
├── evaluation/
│   ├── __init__.py
│   └── metrics.py                 # Evaluation metrics
├── preprocessing/
│   ├── __init__.py
│   └── audio_processor.py         # Audio preprocessing and feature extraction
├── utils/
│   ├── __init__.py
│   └── dataset_loader.py          # Dataset loading utilities
├── graph_dataset_l2_artic/         # Dataset directory
│   └── graph_dataset_l2_artic/
│       ├── train.csv
│       ├── dev.csv
│       └── test.csv
├── graph_dataset_l2_artic_241_trans_41_canno/  # Alternative dataset
├── data/                          # Data directory
├── results/                       # Output directory for results
├── main.py                        # Main pipeline script
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore file
└── README.md                      # This file
```

## 🔮 Future Work

### Multimodal Assessment

- **Video Integration**: Analyze facial expressions, gestures, and body language
- **Eye Tracking**: Attention and engagement metrics
- **Emotion Recognition**: Sentiment and emotional expression analysis

### Advanced NLP

- **Transformer-Based Grammar Checking**: Fine-tuned BERT/GPT models for grammar
- **Semantic Analysis**: Meaning and coherence evaluation
- **Discourse Analysis**: Conversation flow and coherence

### Enhanced Explainability

- **RAG (Retrieval-Augmented Generation)**: Context-aware feedback using knowledge base
- **Visual Explanations**: Feature importance visualizations
- **Interactive Feedback**: Real-time suggestions during speaking

### Model Improvements

- **End-to-End Learning**: Joint training of all components
- **Transfer Learning**: Pre-trained models on large speech corpora
- **Few-Shot Learning**: Adaptation to new languages/dialects
- **Active Learning**: Intelligent data collection strategies

### Production Enhancements

- **API Deployment**: RESTful API for integration
- **Batch Processing**: Efficient processing of large datasets
- **Model Serving**: Scalable model deployment (TorchServe, TensorFlow Serving)
- **Monitoring**: Real-time performance and fairness monitoring

### Research Directions

- **Cross-Lingual Transfer**: Leverage multilingual models
- **Unsupervised Learning**: Reduce reliance on labeled data
- **Causal Inference**: Understand causal relationships in language learning
- **Personalization**: Adaptive assessment based on learner profile

## 📚 References

### Datasets
- L2-ARTIC Dataset: https://www.kaggle.com/datasets/davidthomastran/l2-artic-en-mdd-with-nationality-catogorization
- ARPA Phoneme Set: Standard phoneme inventory

### Models
- Whisper: Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision", 2022
- Wav2Vec 2.0: Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations", 2020

### Libraries
- librosa: McFee et al., "librosa: Audio and Music Analysis in Python", 2015
- spaCy: Explosion AI, "spaCy: Industrial-strength Natural Language Processing"
- PyTorch: Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library", 2019

### Fairness
- Mehrabi et al., "A Survey on Bias and Fairness in Machine Learning", 2021
- Bender & Friedman, "Data Statements for Natural Language Processing", 2018

## 👥 Authors

Pawan Meena

## 🙏 Acknowledgments

- L2-ARTIC dataset creators
- OpenAI for Whisper model
- Facebook AI Research for Wav2Vec2
- Open-source community for excellent tools and libraries
