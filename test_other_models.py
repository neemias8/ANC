"""
Test BART and PEGASUS models for multi-document summarization.
Compares with PRIMERA to see which works best for gospel consolidation.
"""

import argparse
import time
from pathlib import Path
import torch
from transformers import (
    BartForConditionalGeneration, 
    BartTokenizer,
    PegasusForConditionalGeneration, 
    PegasusTokenizer,
)

from src.data_loader import BiblicalDataLoader
from src.evaluator import SummarizationEvaluator
from src.preprocess import PrimeraPreprocessor


class ModelTester:
    """Test different abstractive models."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.preprocessor = PrimeraPreprocessor()
        self.data_loader = BiblicalDataLoader()
        self.evaluator = SummarizationEvaluator()
        
        # Load golden sample once
        print("\n Loading Golden Sample...")
        self.golden_sample = self.data_loader.load_golden_sample()
        print(f" Golden Sample loaded: {len(self.golden_sample)} characters")
        
    def test_bart_large_cnn(self, max_length: int = 1024):
        """Test facebook/bart-large-cnn (CNN/DailyMail fine-tuned)."""
        print("\n" + "="*70)
        print("TESTING BART-LARGE-CNN")
        print("="*70)
        
        print("\n Loading BART model and tokenizer...")
        model_name = "facebook/bart-large-cnn"
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)
        print(" Model loaded")
        
        # Prepare input
        print("\n Preparing input...")
        formatted_input = self.preprocessor.prepare_mds_input()
        print(f"   Input length: {len(formatted_input):,} characters")
        
        # Generate
        print(f"\n Generating summary (max_length={max_length})...")
        start_time = time.time()
        
        inputs = tokenizer(
            formatted_input,
            max_length=1024,  # BART max input
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=max_length,
                min_length=100,
                num_beams=4,
                length_penalty=0.8,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        elapsed = time.time() - start_time
        
        print(f" Summary generated: {len(summary):,} characters in {elapsed:.1f}s")
        print(f"\n Preview:\n{summary[:500]}...\n")
        
        # Evaluate
        print("\n Evaluating...")
        results = self.evaluator.evaluate_summary(summary, self.golden_sample)
        
        # Save
        output_path = Path("outputs/bart_summary.txt")
        output_path.parent.mkdir(exist_ok=True)
        output_path.write_text(summary, encoding="utf-8")
        
        return {
            "model": "BART-large-CNN",
            "summary": summary,
            "results": results,
            "time": elapsed,
            "output_path": str(output_path)
        }
    
    def test_pegasus_xsum(self, max_length: int = 1024):
        """Test google/pegasus-xsum (extreme summarization)."""
        print("\n" + "="*70)
        print("TESTING PEGASUS-XSUM")
        print("="*70)
        
        print("\n Loading PEGASUS model and tokenizer...")
        model_name = "google/pegasus-xsum"
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name).to(self.device)
        print(" Model loaded")
        
        # Prepare input
        print("\n Preparing input...")
        formatted_input = self.preprocessor.prepare_mds_input()
        print(f"   Input length: {len(formatted_input):,} characters")
        
        # Generate
        print(f"\n Generating summary (max_length={max_length})...")
        start_time = time.time()
        
        # PEGASUS-XSUM has max_position_embeddings = 512
        max_input_length = min(512, model.config.max_position_embeddings - 1)
        
        inputs = tokenizer(
            formatted_input,
            max_length=max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Check actual input length
        input_length = inputs['input_ids'].shape[1]
        if input_length >= max_input_length:
            print(f" Warning: Input truncated to {input_length} tokens")
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=max_length,
                min_length=100,
                num_beams=4,
                length_penalty=0.8,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        elapsed = time.time() - start_time
        
        print(f" Summary generated: {len(summary):,} characters in {elapsed:.1f}s")
        print(f"\n Preview:\n{summary[:500]}...\n")
        
        # Evaluate
        print("\n Evaluating...")
        results = self.evaluator.evaluate_summary(summary, self.golden_sample)
        
        # Save
        output_path = Path("outputs/pegasus_xsum_summary.txt")
        output_path.parent.mkdir(exist_ok=True)
        output_path.write_text(summary, encoding="utf-8")
        
        return {
            "model": "PEGASUS-XSUM",
            "summary": summary,
            "results": results,
            "time": elapsed,
            "output_path": str(output_path)
        }
    
    def test_pegasus_large(self, max_length: int = 1024):
        """Test google/pegasus-large (general summarization)."""
        print("\n" + "="*70)
        print("TESTING PEGASUS-LARGE")
        print("="*70)
        
        print("\n Loading PEGASUS-Large model and tokenizer...")
        model_name = "google/pegasus-large"
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name).to(self.device)
        print(" Model loaded")
        
        # Prepare input
        print("\n Preparing input...")
        formatted_input = self.preprocessor.prepare_mds_input()
        print(f"   Input length: {len(formatted_input):,} characters")
        
        # Generate
        print(f"\n Generating summary (max_length={max_length})...")
        start_time = time.time()
        
        # PEGASUS-Large has max_position_embeddings = 1024
        max_input_length = min(1024, model.config.max_position_embeddings - 1)
        
        inputs = tokenizer(
            formatted_input,
            max_length=max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Check actual input length
        input_length = inputs['input_ids'].shape[1]
        if input_length >= max_input_length:
            print(f" Warning: Input truncated to {input_length} tokens")
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=max_length,
                min_length=100,
                num_beams=4,
                length_penalty=0.8,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        elapsed = time.time() - start_time
        
        print(f" Summary generated: {len(summary):,} characters in {elapsed:.1f}s")
        print(f"\n Preview:\n{summary[:500]}...\n")
        
        # Evaluate
        print("\n Evaluating...")
        results = self.evaluator.evaluate_summary(summary, self.golden_sample)
        
        # Save
        output_path = Path("outputs/pegasus_large_summary.txt")
        output_path.parent.mkdir(exist_ok=True)
        output_path.write_text(summary, encoding="utf-8")
        
        return {
            "model": "PEGASUS-Large",
            "summary": summary,
            "results": results,
            "time": elapsed,
            "output_path": str(output_path)
        }
    
    def print_comparison(self, results_list):
        """Print comparison table."""
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        
        print(f"\n{'Model':<20} | {'Kendall ':>10} | {'ROUGE-L':>10} | {'BERTScore':>10} | {'Length':>10} | {'Time':>8}")
        print("-" * 80)
        
        for result in results_list:
            model = result['model']
            metrics = result['results']
            length = len(result['summary'])
            time_sec = result['time']
            
            print(f"{model:<20} | {metrics['kendall_tau']:>10.3f} | "
                  f"{metrics['rouge']['rougeL']['f1']:>10.3f} | "
                  f"{metrics['bertscore']['f1']:>10.3f} | "
                  f"{length:>10,} | {time_sec:>7.1f}s")
        
        print("-" * 80)
        
        # Find best model
        best_tau = max(results_list, key=lambda x: x['results']['kendall_tau'])
        print(f"\n Best Kendall's Tau: {best_tau['model']} (={best_tau['results']['kendall_tau']:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Test BART and PEGASUS models")
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Max output length in tokens"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["bart", "pegasus-xsum", "pegasus-large", "all"],
        default=["all"],
        help="Which models to test"
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = "cpu" if args.device == "cpu" or not torch.cuda.is_available() else "cuda"
    print(f"\n Using device: {device}")
    
    # Initialize tester
    tester = ModelTester(device=device)
    
    # Run tests
    results = []
    models_to_test = args.models
    
    if "all" in models_to_test:
        models_to_test = ["bart", "pegasus-xsum", "pegasus-large"]
    
    if "bart" in models_to_test:
        results.append(tester.test_bart_large_cnn(max_length=args.max_length))
    
    if "pegasus-xsum" in models_to_test:
        results.append(tester.test_pegasus_xsum(max_length=args.max_length))
    
    if "pegasus-large" in models_to_test:
        results.append(tester.test_pegasus_large(max_length=args.max_length))
    
    # Print comparison
    if results:
        tester.print_comparison(results)
    
    print("\n All tests completed!")


if __name__ == "__main__":
    main()
