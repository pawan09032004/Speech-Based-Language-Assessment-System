"""
Scoring Model Module

This module implements scoring models for overall language assessment:
- Baseline regression model (linear combination of features)
- Deep learning model (MLP)
- Score explanation and interpretability

Scores are on a 0-100 scale where:
- 0-40: Beginner
- 40-60: Intermediate
- 60-80: Advanced
- 80-100: Expert/Native-like

Author: SHL AI Research Team
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle

logger = logging.getLogger(__name__)


@dataclass
class ScoreBreakdown:
    """
    Breakdown of overall score into components.
    
    Attributes:
        overall_score: Overall score (0-100)
        pronunciation_score: Pronunciation component (0-100)
        grammar_score: Grammar component (0-100)
        fluency_score: Fluency component (0-100)
        weights: Weights used for combining components
    """
    overall_score: float
    pronunciation_score: float
    grammar_score: float
    fluency_score: float
    weights: Dict[str, float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "overall_score": self.overall_score,
            "pronunciation_score": self.pronunciation_score,
            "grammar_score": self.grammar_score,
            "fluency_score": self.fluency_score,
            "weights": self.weights
        }


class BaselineScoringModel:
    """
    Baseline scoring model using linear combination of features.
    
    This is a simple, interpretable model that combines:
    - Pronunciation features (from phoneme analysis)
    - Grammar features (from NLP analysis)
    - Fluency features (from audio + text)
    """
    
    def __init__(
        self,
        pronunciation_weight: float = 0.4,
        grammar_weight: float = 0.3,
        fluency_weight: float = 0.3
    ):
        """
        Initialize baseline scoring model.
        
        Args:
            pronunciation_weight: Weight for pronunciation component
            grammar_weight: Weight for grammar component
            fluency_weight: Weight for fluency component
        """
        # Normalize weights
        total = pronunciation_weight + grammar_weight + fluency_weight
        self.pronunciation_weight = pronunciation_weight / total
        self.grammar_weight = grammar_weight / total
        self.fluency_weight = fluency_weight / total
        
        logger.info(
            f"Initialized baseline model with weights: "
            f"pronunciation={self.pronunciation_weight:.2f}, "
            f"grammar={self.grammar_weight:.2f}, "
            f"fluency={self.fluency_weight:.2f}"
        )
    
    def _score_pronunciation(self, pronunciation_features: Dict) -> float:
        """
        Score pronunciation based on phoneme error features.
        
        Args:
            pronunciation_features: Dictionary of pronunciation features
            
        Returns:
            Pronunciation score (0-100)
        """
        # Extract key features
        per = pronunciation_features.get("phoneme_error_rate", 1.0)
        substitution_rate = pronunciation_features.get("substitution_rate", 0.0)
        deletion_rate = pronunciation_features.get("deletion_rate", 0.0)
        insertion_rate = pronunciation_features.get("insertion_rate", 0.0)
        correct_rate = pronunciation_features.get("correct_rate", 0.0)
        
        # Score based on error rates (lower errors = higher score)
        # PER of 0.0 = perfect (100), PER of 1.0 = all errors (0)
        per_score = (1.0 - per) * 100
        
        # Also consider individual error types
        error_penalty = (substitution_rate + deletion_rate + insertion_rate) * 50
        
        # Final score: weighted combination
        pronunciation_score = per_score - error_penalty
        
        # Clamp to 0-100
        pronunciation_score = max(0.0, min(100.0, pronunciation_score))
        
        return pronunciation_score
    
    def _score_grammar(self, grammar_features: Dict) -> float:
        """
        Score grammar based on grammar analysis features.
        
        Args:
            grammar_features: Dictionary of grammar features
            
        Returns:
            Grammar score (0-100)
        """
        # Extract key features
        error_density = grammar_features.get("grammar_error_density", 0.0)
        complexity_score = grammar_features.get("complexity_score", 0.0)
        avg_sentence_length = grammar_features.get("avg_sentence_length", 0.0)
        
        # Score based on error density (lower errors = higher score)
        # Error density of 0 = perfect (100), high error density = lower score
        error_score = max(0.0, 100.0 - error_density * 10.0)
        
        # Reward complexity (but not too much - simple correct > complex incorrect)
        complexity_bonus = min(complexity_score * 10.0, 10.0)
        
        # Final score
        grammar_score = error_score + complexity_bonus
        
        # Clamp to 0-100
        grammar_score = max(0.0, min(100.0, grammar_score))
        
        return grammar_score
    
    def _score_fluency(self, fluency_features: Dict) -> float:
        """
        Score fluency based on fluency features.
        
        Args:
            fluency_features: Dictionary of fluency features
            
        Returns:
            Fluency score (0-100)
        """
        # Extract key features
        speaking_rate = fluency_features.get("speaking_rate", 0.0)
        hesitation_ratio = fluency_features.get("hesitation_ratio", 0.0)
        pause_ratio = fluency_features.get("pause_ratio", 0.0)
        sentence_completeness = fluency_features.get("sentence_completeness", 0.0)
        prosodic_variability = fluency_features.get("prosodic_variability", 0.0)
        
        # Speaking rate: optimal around 2-3 words/sec
        # Too slow (< 1.5) or too fast (> 4) is penalized
        if 1.5 <= speaking_rate <= 4.0:
            rate_score = 100.0
        elif speaking_rate < 1.5:
            rate_score = (speaking_rate / 1.5) * 100.0
        else:
            rate_score = max(0.0, 100.0 - (speaking_rate - 4.0) * 20.0)
        
        # Hesitation penalty
        hesitation_penalty = hesitation_ratio * 50.0
        
        # Pause penalty (too many pauses = disfluent)
        pause_penalty = min(pause_ratio * 100.0, 30.0)
        
        # Sentence completeness bonus
        completeness_bonus = sentence_completeness * 20.0
        
        # Prosodic variability bonus (more expressive = better)
        prosodic_bonus = prosodic_variability * 10.0
        
        # Final score
        fluency_score = (
            rate_score * 0.4 +
            (100.0 - hesitation_penalty) * 0.2 +
            (100.0 - pause_penalty) * 0.2 +
            completeness_bonus * 0.1 +
            prosodic_bonus * 0.1
        )
        
        # Clamp to 0-100
        fluency_score = max(0.0, min(100.0, fluency_score))
        
        return fluency_score
    
    def score(
        self,
        pronunciation_features: Dict,
        grammar_features: Dict,
        fluency_features: Dict
    ) -> ScoreBreakdown:
        """
        Compute overall score from all feature sets.
        
        Args:
            pronunciation_features: Pronunciation features
            grammar_features: Grammar features
            fluency_features: Fluency features
            
        Returns:
            ScoreBreakdown object
        """
        # Score each component
        pronunciation_score = self._score_pronunciation(pronunciation_features)
        grammar_score = self._score_grammar(grammar_features)
        fluency_score = self._score_fluency(fluency_features)
        
        # Combine with weights
        overall_score = (
            pronunciation_score * self.pronunciation_weight +
            grammar_score * self.grammar_weight +
            fluency_score * self.fluency_weight
        )
        
        return ScoreBreakdown(
            overall_score=overall_score,
            pronunciation_score=pronunciation_score,
            grammar_score=grammar_score,
            fluency_score=fluency_score,
            weights={
                "pronunciation": self.pronunciation_weight,
                "grammar": self.grammar_weight,
                "fluency": self.fluency_weight
            }
        )


class MLPScoringModel(nn.Module):
    """
    Deep learning scoring model using Multi-Layer Perceptron (MLP).
    
    This model learns non-linear relationships between features and scores.
    """
    
    def __init__(
        self,
        input_dim: int = 20,  # Total feature dimension
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.2
    ):
        """
        Initialize MLP scoring model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super(MLPScoringModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer: 4 outputs (overall, pronunciation, grammar, fluency)
        layers.append(nn.Linear(prev_dim, 4))
        layers.append(nn.Sigmoid())  # Output in [0, 1], scale to [0, 100]
        
        self.model = nn.Sequential(*layers)
        
        logger.info(f"Initialized MLP model: {input_dim} -> {hidden_dims} -> 4")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (batch_size, input_dim)
            
        Returns:
            Scores (batch_size, 4): [overall, pronunciation, grammar, fluency]
        """
        scores = self.model(x)
        # Scale from [0, 1] to [0, 100]
        scores = scores * 100.0
        return scores


class ScoringModelTrainer:
    """
    Trainer for MLP scoring model.
    """
    
    def __init__(
        self,
        model: MLPScoringModel,
        learning_rate: float = 0.001,
        device: Optional[str] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: MLP model to train
            learning_rate: Learning rate for optimizer
            device: Device to train on ("cpu" or "cuda")
        """
        self.model = model
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Feature scaler
        self.scaler = StandardScaler()
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2
    ):
        """
        Train the model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target scores (n_samples, 4): [overall, pronunciation, grammar, fluency]
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data to use for validation
        """
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split train/validation
        n_val = int(len(X_scaled) * validation_split)
        X_train = X_scaled[:-n_val]
        y_train = y[:-n_val]
        X_val = X_scaled[-n_val:]
        y_val = y[-n_val:]
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for i in range(0, len(X_train_t), batch_size):
                batch_X = X_train_t[i:i+batch_size]
                batch_y = y_train_t[i:i+batch_size]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = self.criterion(val_outputs, y_val_t).item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"train_loss={train_loss/len(X_train_t)*batch_size:.4f}, "
                    f"val_loss={val_loss:.4f}"
                )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict scores.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted scores (n_samples, 4)
        """
        self.model.eval()
        
        # Normalize features
        X_scaled = self.scaler.transform(X)
        
        # Convert to tensor
        X_t = torch.FloatTensor(X_scaled).to(self.device)
        
        # Predict
        with torch.no_grad():
            scores = self.model(X_t)
        
        return scores.cpu().numpy()
    
    def save(self, model_path: str, scaler_path: str):
        """Save model and scaler."""
        torch.save(self.model.state_dict(), model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Model saved to {model_path}")
    
    def load(self, model_path: str, scaler_path: str):
        """Load model and scaler."""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")


def main():
    """
    Example usage of scoring models.
    """
    # Baseline model
    baseline = BaselineScoringModel()
    
    pronunciation_features = {
        "phoneme_error_rate": 0.2,
        "substitution_rate": 0.1,
        "deletion_rate": 0.05,
        "insertion_rate": 0.05,
        "correct_rate": 0.8
    }
    
    grammar_features = {
        "grammar_error_density": 2.0,
        "complexity_score": 0.6,
        "avg_sentence_length": 12.0
    }
    
    fluency_features = {
        "speaking_rate": 2.5,
        "hesitation_ratio": 0.1,
        "pause_ratio": 0.15,
        "sentence_completeness": 0.9,
        "prosodic_variability": 0.4
    }
    
    score = baseline.score(pronunciation_features, grammar_features, fluency_features)
    print(f"Baseline score: {score.to_dict()}")


if __name__ == "__main__":
    main()




