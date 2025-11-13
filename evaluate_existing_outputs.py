#!/usr/bin/env python3
"""
Evaluate existing output files against Golden Sample.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.evaluator import SummarizationEvaluator
from src.data_loader import BiblicalDataLoader

def main():
    """Evaluate existing outputs."""
    
    # Load Golden Sample
    print("\nLoading Golden Sample...")
    loader = BiblicalDataLoader()
    golden_sample = loader.load_golden_sample()
    print(f"Golden Sample: {len(golden_sample)} characters")
    
    # Initialize evaluator
    evaluator = SummarizationEvaluator()
    
    # Output files to evaluate
    outputs = [
        ("TAEG (LEXRANK-TA)", "outputs/taeg_summary_lexrank-ta.txt"),
        ("BART", "outputs/bart_summary.txt"),
        ("PEGASUS", "outputs/pegasus_summary.txt"),
        ("PRIMERA MDS", "outputs/primera_standard_mds.txt"),
        ("PRIMERA Event Consolidation", "outputs/primera_event_by_event.txt"),
    ]
    
    results = []
    
    for name, filepath in outputs:
        path = Path(filepath)
        if not path.exists():
            print(f"\n{name}: File not found - {filepath}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Evaluating: {name}")
        print(f"{'='*70}")
        
        # Load output
        hypothesis = path.read_text(encoding='utf-8')
        print(f"Length: {len(hypothesis)} characters, {len(hypothesis.split())} words")
        
        # Evaluate
        eval_results = evaluator.evaluate_summary(hypothesis, golden_sample)
        
        # Print results
        print(f"\nROUGE-L F1: {eval_results['rouge']['rougeL']['f1']:.4f}")
        print(f"METEOR: {eval_results['meteor']:.4f}")
        print(f"BERTScore F1: {eval_results['bertscore']['f1']:.4f}")
        print(f"Kendall's Tau: {eval_results['kendall_tau']:.4f}")
        
        results.append({
            'name': name,
            'rouge_l': eval_results['rouge']['rougeL']['f1'],
            'meteor': eval_results['meteor'],
            'bertscore': eval_results['bertscore']['f1'],
            'kendall_tau': eval_results['kendall_tau'],
            'length': len(hypothesis)
        })
    
    # Print comparison table
    print(f"\n{'='*100}")
    print("COMPARISON TABLE")
    print(f"{'='*100}")
    print(f"{'Method':<35} | {'Kendall':<8} | {'ROUGE-L':<8} | {'METEOR':<8} | {'BERTScore':<10} | {'Length':<10}")
    print(f"{'-'*100}")
    
    for r in results:
        print(f"{r['name']:<35} | {r['kendall_tau']:>8.4f} | {r['rouge_l']:>8.4f} | {r['meteor']:>8.4f} | {r['bertscore']:>10.4f} | {r['length']:>10}")
    
    print(f"{'-'*100}")
    
    # Find best in each metric
    best_kendall = max(results, key=lambda x: x['kendall_tau'])
    best_rouge = max(results, key=lambda x: x['rouge_l'])
    best_meteor = max(results, key=lambda x: x['meteor'])
    best_bert = max(results, key=lambda x: x['bertscore'])
    
    print(f"\nBest Kendall's Tau: {best_kendall['name']} ({best_kendall['kendall_tau']:.4f})")
    print(f"Best ROUGE-L: {best_rouge['name']} ({best_rouge['rouge_l']:.4f})")
    print(f"Best METEOR: {best_meteor['name']} ({best_meteor['meteor']:.4f})")
    print(f"Best BERTScore: {best_bert['name']} ({best_bert['bertscore']:.4f})")

if __name__ == "__main__":
    main()
