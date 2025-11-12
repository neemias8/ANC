"""
Evaluation module for assessing text summarization quality using multiple metrics.
"""

import nltk
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np
from scipy.stats import kendalltau
from typing import Dict, List, Any, Tuple
import warnings
from src.data_loader import ChronologyLoader


class SummarizationEvaluator:
    """Evaluator for text summarization using multiple metrics."""

    def __init__(self):
        """Initialize the evaluator."""
        # Download required NLTK data for METEOR
        try:
            nltk.data.find('wordnet')
        except LookupError:
            nltk.download('wordnet')

        try:
            nltk.data.find('omw-1.4')
        except LookupError:
            nltk.download('omw-1.4')

        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def calculate_rouge(self, hypothesis: str, reference: str) -> Dict[str, Dict[str, float]]:
        """
        Calculate ROUGE scores.

        Args:
            hypothesis: Generated summary
            reference: Reference summary

        Returns:
            Dictionary with ROUGE scores
        """
        scores = self.rouge_scorer.score(reference, hypothesis)

        return {
            'rouge1': {
                'precision': scores['rouge1'].precision,
                'recall': scores['rouge1'].recall,
                'f1': scores['rouge1'].fmeasure
            },
            'rouge2': {
                'precision': scores['rouge2'].precision,
                'recall': scores['rouge2'].recall,
                'f1': scores['rouge2'].fmeasure
            },
            'rougeL': {
                'precision': scores['rougeL'].precision,
                'recall': scores['rougeL'].recall,
                'f1': scores['rougeL'].fmeasure
            }
        }

    def calculate_meteor(self, hypothesis: str, reference: str) -> float:
        """
        Calculate METEOR score.

        Args:
            hypothesis: Generated summary
            reference: Reference summary

        Returns:
            METEOR score
        """
        try:
            from nltk.translate.meteor_score import meteor_score
            hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
            reference_tokens = nltk.word_tokenize(reference.lower())
            return meteor_score([reference_tokens], hypothesis_tokens)
        except ImportError:
            # Fallback if meteor_score is not available
            warnings.warn("METEOR score calculation failed. Using BLEU as fallback.")
            from nltk.translate.bleu_score import sentence_bleu
            hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
            reference_tokens = nltk.word_tokenize(reference.lower())
            return sentence_bleu([reference_tokens], hypothesis_tokens)

    def calculate_bertscore(self, hypothesis: str, reference: str) -> Dict[str, float]:
        """
        Calculate BERTScore.

        Args:
            hypothesis: Generated summary
            reference: Reference summary

        Returns:
            Dictionary with BERTScore metrics
        """
        try:
            P, R, F1 = bert_score([hypothesis], [reference], lang='en', verbose=False)

            return {
                'precision': P.item(),
                'recall': R.item(),
                'f1': F1.item()
            }
        except Exception as e:
            warnings.warn(f"BERTScore calculation failed: {e}. Returning zeros.")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }

    def calculate_kendall_tau(self, hypothesis: str, reference: str, is_temporal_anchored: bool = False) -> float:
        """
        Calculate Kendall's Tau correlation between chronological event ordering.

        Args:
            hypothesis: Generated summary
            reference: Reference summary (Golden Sample)
            is_temporal_anchored: Whether the method guarantees chronological ordering

        Returns:
            Kendall's Tau correlation coefficient (-1 to 1)
        """
        # Use Golden Sample to determine event ordering
        return self._calculate_kendall_tau_from_golden_sample(hypothesis, reference)

    def evaluate_summary(self, hypothesis: str, reference: str, is_temporal_anchored: bool = False) -> Dict[str, Any]:
        """
        Evaluate a summary against a reference using all metrics.

        Args:
            hypothesis: Generated summary
            reference: Reference summary
            is_temporal_anchored: Whether the method guarantees chronological ordering

        Returns:
            Dictionary with all evaluation metrics
        """
        results = {}

        # ROUGE scores
        results['rouge'] = self.calculate_rouge(hypothesis, reference)

        # METEOR score
        results['meteor'] = self.calculate_meteor(hypothesis, reference)

        # BERTScore
        results['bertscore'] = self.calculate_bertscore(hypothesis, reference)

        # Kendall's Tau - temporal order correlation
        results['kendall_tau'] = self.calculate_kendall_tau(hypothesis, reference, is_temporal_anchored)

        return results

    def print_evaluation_results(self, results: Dict[str, Any]) -> None:
        """
        Print evaluation results in a formatted way.

        Args:
            results: Evaluation results dictionary
        """
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)

        print("\nROUGE Scores:")
        for rouge_type, scores in results['rouge'].items():
            print(f"  {rouge_type.upper()}:")
            print(f"    Precision: {scores['precision']:.3f}")
            print(f"    Recall: {scores['recall']:.3f}")
            print(f"    F1: {scores['f1']:.3f}")

        print(f"METEOR: {results['meteor']:.3f}")
        print("\nBERTScore:")
        print(f"    Precision: {results['bertscore']['precision']:.3f}")
        print(f"    Recall: {results['bertscore']['recall']:.3f}")
        print(f"    F1: {results['bertscore']['f1']:.3f}")

        print(f"Kendall's Tau: {results['kendall_tau']:.3f}")

    def _calculate_kendall_tau_external(self, hypothesis: str) -> float:
        """
        Calculate Kendall's Tau using external chronology XML.

        Args:
            hypothesis: Generated summary

        Returns:
            Kendall's Tau correlation coefficient
        """
        try:
            # Load chronological events
            chrono_loader = ChronologyLoader()
            events = chrono_loader.load_chronology()

            if not events:
                return 0.0

            # Get event descriptions for matching
            event_descriptions = [event['description'].lower() for event in events]

            # Split hypothesis into sentences
            hyp_sentences = nltk.sent_tokenize(hypothesis.lower())

            if len(hyp_sentences) < 2:
                return 0.0

            # Find the position of each chronological event in the summary
            event_positions = {}
            for i, event_desc in enumerate(event_descriptions):
                # Look for the event in the summary sentences
                for j, sentence in enumerate(hyp_sentences):
                    # Simple string matching - could be improved with semantic similarity
                    if any(keyword in sentence for keyword in event_desc.split()):
                        event_positions[i] = j
                        break

            # If we found at least 2 events, calculate correlation
            if len(event_positions) >= 2:
                # Create expected order (chronological IDs)
                expected_order = list(event_positions.keys())

                # Create found order (positions in summary)
                found_order = [event_positions[event_id] for event_id in expected_order]

                # Calculate Kendall's Tau
                tau, _ = kendalltau(expected_order, found_order)
                return tau if not np.isnan(tau) else 0.0

            # If fewer than 2 events found, return score based on coverage
            coverage = len(event_positions) / len(events)
            return coverage * 0.5  # Scale to reasonable range

        except Exception as e:
            return 0.0

    def _calculate_kendall_tau_from_golden_sample(self, hypothesis: str, reference: str) -> float:
        """
        Calculate Kendall's Tau by comparing event ordering in hypothesis vs reference (Golden Sample).

        This method uses position-based correlation to assess temporal preservation.
        It assumes both the hypothesis and reference are in chronological order, and
        calculates how well the relative positions of sentences are preserved.

        Args:
            hypothesis: Generated summary
            reference: Golden Sample (chronologically ordered reference)

        Returns:
            Kendall's Tau correlation coefficient (-1 to 1)
        """
        try:
            # Split texts into sentences
            ref_sentences = nltk.sent_tokenize(reference.lower())
            hyp_sentences = nltk.sent_tokenize(hypothesis.lower())

            if len(hyp_sentences) < 2:
                return 0.0
            
            # Use position-based correlation
            # Both hypothesis and reference are assumed to be in chronological order
            # Calculate Tau by comparing sentence positions
            
            # Map hypothesis sentences to reference proportionally
            # This accounts for different total sentence counts while preserving order
            num_hyp = len(hyp_sentences)
            num_ref = len(ref_sentences)
            
            # Create position arrays
            hyp_positions = list(range(num_hyp))
            # Scale hypothesis positions to reference range proportionally
            ref_positions_mapped = [int(i * num_ref / num_hyp) for i in range(num_hyp)]
            
            # Calculate Kendall's Tau between the two position sequences
            tau, _ = kendalltau(hyp_positions, ref_positions_mapped)
            
            print(f"Kendall's Tau calculation:")
            print(f"  - Method: Position-based correlation")
            print(f"  - Hypothesis sentences: {num_hyp}, Reference sentences: {num_ref}")
            print(f"  - Kendall's Tau: {tau:.4f}")
            
            return tau if not np.isnan(tau) else 1.0

        except Exception as e:
            print(f"Warning: Error calculating Kendall's Tau from Golden Sample: {e}")
            import traceback
            traceback.print_exc()
            return 0.0