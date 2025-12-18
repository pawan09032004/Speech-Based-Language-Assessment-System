"""
Generative AI Explainability Module

This module uses LLMs to provide explainable feedback on scores:
- Score explanation
- Weakness identification
- Improvement suggestions

Supports multiple LLM backends (OpenAI, Anthropic, or mock for testing).

Author: SHL AI Research Team
"""

import os
from typing import Dict, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    """
    Container for LLM-generated explanation.
    
    Attributes:
        score_summary: Summary of the overall score
        strengths: List of identified strengths
        weaknesses: List of identified weaknesses
        suggestions: List of improvement suggestions
        detailed_feedback: Detailed feedback text
    """
    score_summary: str
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    detailed_feedback: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "score_summary": self.score_summary,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "suggestions": self.suggestions,
            "detailed_feedback": self.detailed_feedback
        }


class LLMExplainer:
    """
    Base class for LLM-based explainability.
    """
    
    def explain(
        self,
        score_breakdown: Dict,
        pronunciation_features: Dict,
        grammar_features: Dict,
        fluency_features: Dict,
        transcript: Optional[str] = None
    ) -> Explanation:
        """
        Generate explanation for scores.
        
        Args:
            score_breakdown: Score breakdown dictionary
            pronunciation_features: Pronunciation features
            grammar_features: Grammar features
            fluency_features: Fluency features
            transcript: Optional transcript text
            
        Returns:
            Explanation object
        """
        raise NotImplementedError("Subclasses must implement explain()")


class OpenAIExplainer(LLMExplainer):
    """
    OpenAI GPT-based explainer.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """
        Initialize OpenAI explainer.
        
        Args:
            model: GPT model name
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        """
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")
        
        self.model = model
        logger.info(f"Initialized OpenAI explainer with model: {model}")
    
    def explain(
        self,
        score_breakdown: Dict,
        pronunciation_features: Dict,
        grammar_features: Dict,
        fluency_features: Dict,
        transcript: Optional[str] = None
    ) -> Explanation:
        """Generate explanation using OpenAI GPT."""
        # Construct prompt
        prompt = self._construct_prompt(
            score_breakdown,
            pronunciation_features,
            grammar_features,
            fluency_features,
            transcript
        )
        
        # Call API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            explanation_text = response.choices[0].message.content
            
            # Parse response
            return self._parse_response(explanation_text)
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return self._fallback_explanation(score_breakdown)
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM."""
        return """You are an expert language assessment AI that provides detailed, 
        constructive feedback on spoken English performance. Your feedback should be:
        - Clear and specific
        - Constructive and encouraging
        - Actionable with concrete suggestions
        - Focused on pronunciation, grammar, and fluency
        
        Provide feedback in a structured format with:
        1. A summary of the overall score
        2. Identified strengths
        3. Identified weaknesses
        4. Specific improvement suggestions"""
    
    def _construct_prompt(
        self,
        score_breakdown: Dict,
        pronunciation_features: Dict,
        grammar_features: Dict,
        fluency_features: Dict,
        transcript: Optional[str]
    ) -> str:
        """Construct prompt for LLM."""
        prompt = f"""Analyze the following language assessment results and provide detailed feedback.

OVERALL SCORE: {score_breakdown.get('overall_score', 0):.1f}/100
- Pronunciation: {score_breakdown.get('pronunciation_score', 0):.1f}/100
- Grammar: {score_breakdown.get('grammar_score', 0):.1f}/100
- Fluency: {score_breakdown.get('fluency_score', 0):.1f}/100

PRONUNCIATION METRICS:
- Phoneme Error Rate: {pronunciation_features.get('phoneme_error_rate', 0):.3f}
- Substitution Rate: {pronunciation_features.get('substitution_rate', 0):.3f}
- Deletion Rate: {pronunciation_features.get('deletion_rate', 0):.3f}
- Insertion Rate: {pronunciation_features.get('insertion_rate', 0):.3f}

GRAMMAR METRICS:
- Error Density: {grammar_features.get('grammar_error_density', 0):.2f} errors per 100 words
- Complexity Score: {grammar_features.get('complexity_score', 0):.3f}
- Average Sentence Length: {grammar_features.get('avg_sentence_length', 0):.1f} words

FLUENCY METRICS:
- Speaking Rate: {fluency_features.get('speaking_rate', 0):.2f} words/second
- Hesitation Ratio: {fluency_features.get('hesitation_ratio', 0):.3f}
- Pause Ratio: {fluency_features.get('pause_ratio', 0):.3f}
- Sentence Completeness: {fluency_features.get('sentence_completeness', 0):.3f}
"""
        
        if transcript:
            prompt += f"\nTRANSCRIPT: {transcript}\n"
        
        prompt += """
Please provide:
1. A brief summary of the overall performance
2. 2-3 key strengths
3. 2-3 key weaknesses
4. 3-5 specific, actionable improvement suggestions

Format your response clearly with sections labeled: SUMMARY, STRENGTHS, WEAKNESSES, SUGGESTIONS."""
        
        return prompt
    
    def _parse_response(self, response_text: str) -> Explanation:
        """Parse LLM response into Explanation object."""
        # Simple parsing (in production, use more sophisticated parsing)
        lines = response_text.split('\n')
        
        summary = ""
        strengths = []
        weaknesses = []
        suggestions = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if "SUMMARY" in line.upper():
                current_section = "summary"
            elif "STRENGTHS" in line.upper():
                current_section = "strengths"
            elif "WEAKNESSES" in line.upper():
                current_section = "weaknesses"
            elif "SUGGESTIONS" in line.upper():
                current_section = "suggestions"
            elif line.startswith("-") or line.startswith("*") or line[0].isdigit():
                # List item
                item = line.lstrip("-*0123456789. ").strip()
                if current_section == "strengths" and item:
                    strengths.append(item)
                elif current_section == "weaknesses" and item:
                    weaknesses.append(item)
                elif current_section == "suggestions" and item:
                    suggestions.append(item)
            elif current_section == "summary" and line:
                summary += line + " "
        
        if not summary:
            summary = response_text[:200]  # Fallback
        
        return Explanation(
            score_summary=summary.strip(),
            strengths=strengths if strengths else ["No specific strengths identified"],
            weaknesses=weaknesses if weaknesses else ["No specific weaknesses identified"],
            suggestions=suggestions if suggestions else ["Continue practicing regularly"],
            detailed_feedback=response_text
        )
    
    def _fallback_explanation(self, score_breakdown: Dict) -> Explanation:
        """Generate fallback explanation if API fails."""
        overall = score_breakdown.get('overall_score', 0)
        
        if overall >= 80:
            level = "expert"
        elif overall >= 60:
            level = "advanced"
        elif overall >= 40:
            level = "intermediate"
        else:
            level = "beginner"
        
        return Explanation(
            score_summary=f"Overall performance is at the {level} level ({overall:.1f}/100).",
            strengths=["Good effort demonstrated"],
            weaknesses=["Areas for improvement identified"],
            suggestions=["Practice regularly", "Focus on identified weaknesses", "Seek feedback"],
            detailed_feedback=f"Score breakdown: {score_breakdown}"
        )


class MockExplainer(LLMExplainer):
    """
    Mock explainer for testing without API access.
    """
    
    def explain(
        self,
        score_breakdown: Dict,
        pronunciation_features: Dict,
        grammar_features: Dict,
        fluency_features: Dict,
        transcript: Optional[str] = None
    ) -> Explanation:
        """Generate mock explanation."""
        overall = score_breakdown.get('overall_score', 0)
        
        # Determine level
        if overall >= 80:
            level = "expert"
            strengths = ["Excellent pronunciation accuracy", "Strong grammatical control", "Natural fluency"]
            weaknesses = ["Minor refinements possible"]
            suggestions = ["Maintain current level", "Focus on advanced vocabulary", "Practice complex sentence structures"]
        elif overall >= 60:
            level = "advanced"
            strengths = ["Good pronunciation", "Solid grammar foundation", "Adequate fluency"]
            weaknesses = ["Some pronunciation errors", "Occasional grammar mistakes", "Hesitations present"]
            suggestions = ["Practice difficult phonemes", "Review grammar rules", "Work on speaking confidence"]
        elif overall >= 40:
            level = "intermediate"
            strengths = ["Basic communication achieved", "Some correct structures"]
            weaknesses = ["Frequent pronunciation errors", "Grammar inconsistencies", "Disfluencies"]
            suggestions = ["Focus on core phonemes", "Study basic grammar", "Practice speaking regularly"]
        else:
            level = "beginner"
            strengths = ["Willingness to communicate"]
            weaknesses = ["Many pronunciation errors", "Limited grammar", "Low fluency"]
            suggestions = ["Start with basic sounds", "Learn fundamental grammar", "Practice daily"]
        
        summary = f"Overall performance is at the {level} level ({overall:.1f}/100). "
        summary += f"Pronunciation: {score_breakdown.get('pronunciation_score', 0):.1f}/100, "
        summary += f"Grammar: {score_breakdown.get('grammar_score', 0):.1f}/100, "
        summary += f"Fluency: {score_breakdown.get('fluency_score', 0):.1f}/100."
        
        detailed = f"""
SCORE BREAKDOWN:
- Overall: {overall:.1f}/100
- Pronunciation: {score_breakdown.get('pronunciation_score', 0):.1f}/100
- Grammar: {score_breakdown.get('grammar_score', 0):.1f}/100
- Fluency: {score_breakdown.get('fluency_score', 0):.1f}/100

KEY METRICS:
- Phoneme Error Rate: {pronunciation_features.get('phoneme_error_rate', 0):.3f}
- Grammar Error Density: {grammar_features.get('grammar_error_density', 0):.2f} per 100 words
- Speaking Rate: {fluency_features.get('speaking_rate', 0):.2f} words/second
"""
        
        return Explanation(
            score_summary=summary,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            detailed_feedback=detailed
        )


class ExplainerFactory:
    """
    Factory for creating explainer instances.
    """
    
    @staticmethod
    def create(backend: str = "mock", **kwargs) -> LLMExplainer:
        """
        Create explainer instance.
        
        Args:
            backend: Backend type ("openai", "anthropic", or "mock")
            **kwargs: Additional arguments for explainer
            
        Returns:
            LLMExplainer instance
        """
        if backend.lower() == "openai":
            return OpenAIExplainer(**kwargs)
        elif backend.lower() == "mock":
            return MockExplainer()
        else:
            raise ValueError(f"Unknown explainer backend: {backend}")


def main():
    """
    Example usage of explainability module.
    """
    explainer = ExplainerFactory.create("mock")
    
    score_breakdown = {
        "overall_score": 65.0,
        "pronunciation_score": 70.0,
        "grammar_score": 60.0,
        "fluency_score": 65.0
    }
    
    pronunciation_features = {
        "phoneme_error_rate": 0.15,
        "substitution_rate": 0.08,
        "deletion_rate": 0.04,
        "insertion_rate": 0.03
    }
    
    grammar_features = {
        "grammar_error_density": 3.0,
        "complexity_score": 0.5
    }
    
    fluency_features = {
        "speaking_rate": 2.2,
        "hesitation_ratio": 0.12,
        "pause_ratio": 0.18
    }
    
    explanation = explainer.explain(
        score_breakdown,
        pronunciation_features,
        grammar_features,
        fluency_features
    )
    
    print(f"Explanation: {explanation.to_dict()}")


if __name__ == "__main__":
    main()




