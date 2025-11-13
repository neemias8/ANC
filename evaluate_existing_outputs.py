#!/usr/bin/env python3
"""
Evaluate existing output files against Golden Sample.
"""

import sys
import re
from pathlib import Path

# Add src directory to path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from src.evaluator import SummarizationEvaluator
from src.data_loader import BiblicalDataLoader

def extract_numbered_events(text: str) -> list:
    """
    Extract numbered event blocks from Golden Sample or summary.
    Returns list of tuples: (event_id, event_text)
    """
    # Match number followed by space and optionally quotes, then capital letter
    # Works even if text is all on one line
    pattern = re.compile(r'\b(\d+)\s+"?([A-Z])')
    events = []
    
    matches = list(pattern.finditer(text))
    
    for i, match in enumerate(matches):
        event_id = int(match.group(1))
        # Start position is the number
        start_pos = match.start()
        
        # End position is the start of next event (or end of text)
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(text)
        
        # Extract event text
        event_text = text[start_pos:end_pos].strip()
        events.append((event_id, event_text))
    
    return events

def main():
    """Evaluate existing outputs."""
    
    # Load Golden Sample
    print("\nLoading Golden Sample...")
    loader = BiblicalDataLoader()
    golden_sample = loader.load_golden_sample()
    print(f"Golden Sample: {len(golden_sample)} characters")
    
    # Extract numbered events from Golden Sample
    golden_events = extract_numbered_events(golden_sample)
    print(f"Extracted {len(golden_events)} numbered events from Golden Sample")
    
    # Initialize evaluator
    evaluator = SummarizationEvaluator()
    
    # Output files to evaluate
    outputs = [
        ("TAEG (LEXRANK-TA)", "outputs/taeg_summary_lexrank-ta.txt"),
        ("BART", "outputs/bart_summary.txt"),
        ("PEGASUS-XSUM", "outputs/pegasus_xsum_summary.txt"),
        ("PEGASUS-Large", "outputs/pegasus_large_summary.txt"),
        ("PRIMERA Standard MDS", "outputs/primera_standard_mds.txt"),
        ("PRIMERA Event-by-Event", "outputs/primera_event_by_event.txt"),
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
        
        # Extract events from hypothesis
        hypothesis_events = extract_numbered_events(hypothesis)
        print(f"Extracted {len(hypothesis_events)} numbered events from summary")
        
        # Evaluate
        eval_results = evaluator.evaluate_summary(
            hypothesis, 
            golden_sample,
            reference_events=golden_events,
            hypothesis_events=hypothesis_events
        )
        
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
